"""Where does the encoder's device time go, and how much HBM traffic does each fusion move?

Runs the jitted Laya forward on synthetic bucketed inputs, dumps the optimized HLO, and reads xprof's
per-op stats (device time, HBM bytes) so ops / kernels can be ranked by HBM traffic.

  python benchmarks/hbm_traffic.py [--model laya] [--rows 64] [--seq 256] [--impl auto] [--out out.json] [--tracer-level 2]

xprof has no HBM counters for Pallas custom calls, so with impl=pallas the kernels' traffic is derived
from their BlockSpecs (`fused_hbm_bytes`). The fused path wraps each layer in a jax.named_scope (visible
in the trace viewer); --tracer-level sets jax.profiler.ProfileOptions.device_tracer_level.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def synthetic(model, rows: int, seq: int, options: int = 4):
    """Bucketed inputs of a fixed shape: real tokens fill the row, the last `options` positions are markers."""
    rng = np.random.default_rng(0)
    vocab = model.params["encoder.embeddings.tok_embeddings.weight"].shape[0]
    ids = rng.integers(5, vocab - 1, (rows, seq), np.int32)
    marker_pos = np.tile(np.arange(seq - options, seq, dtype=np.int32), (rows, 1))
    return {
        "input_ids": jnp.asarray(ids),
        "attention_mask": jnp.ones((rows, seq), jnp.int32),
        "marker_pos": jnp.asarray(marker_pos),
        "marker_mask": jnp.ones((rows, options), bool),
        "qtype": jnp.zeros((rows,), jnp.int32),
    }


def op_stats(logdir: str, calls: int):
    """Per-op device time and HBM traffic from xprof's framework_op_stats (XLA's cost model; the
    Pallas custom calls report 0 bytes there, see `fused_hbm_bytes` for them)."""
    from xprof.convert import raw_to_tool_data

    pb = glob.glob(logdir + "/plugins/profile/*/*.xplane.pb")[0]
    t = json.loads(raw_to_tool_data.xspace_to_tool_data([pb], "framework_op_stats", {})[0])
    t = t[0] if isinstance(t, list) else t
    cols = [c["id"] for c in t["cols"]]
    rows = [dict(zip(cols, [c.get("v") if c else None for c in r["c"]])) for r in t["rows"]]
    out = []
    for r in rows:
        if r["host_or_device"] != "Device":
            continue
        ms = r["total_self_time"] / 1000 / calls
        bw = r["measured_memory_bw"] or 0.0  # GB/s
        out.append(
            {
                "op": r["operation"],
                "type": r["type"],
                "ms": ms,
                "hbm_gb_per_s": bw,
                "hbm_mb": bw * ms,
                "tflops": (r["model_flop_rate"] or 0.0) / 1000,
                "bound_by": r["bound_by"],
            }
        )
    return sorted(out, key=lambda o: -o["ms"])


def fused_hbm_bytes(model, rows: int, seq: int) -> dict[str, float]:
    """HBM bytes per call of the fused path, from the kernels' BlockSpecs (xprof has no counters
    for custom calls): each input block is fetched once per change of its block index, each output
    block written once; weights have constant indices, so they are fetched once per kernel."""
    from deciax.kernels.attention.pallas import BLOCK
    from deciax.kernels.fused import LAYER_MAX_SEQ

    cfg = model.config.encoder
    d = cfg.hidden_size
    r = rows * seq
    fp = model.fused
    bf, f32 = 2, 4

    def attention(window: int) -> float:
        tq = min(BLOCK if window else 2 * BLOCK, seq)
        tk = min(BLOCK // 2 if window else 2 * BLOCK, seq)
        nq, nk = seq // tq, seq // tk
        if window:
            visited = sum(min(((i + 1) * tq - 1 + window) // tk, nk - 1) - max((i * tq - window) // tk, 0) + 1 for i in range(nq))
        else:
            visited = nq * nk
        return rows * (seq * d * bf + 2 * visited * tk * d * bf + seq * d * bf)  # q once, k and v per visited block, out

    def layer(k: str, window: int, act: str) -> float:
        wqkv = fp[f"{k}.WqkvT" if "encoder" in k else f"{k}.in_projT"]
        wi, wo2 = (fp[f"{k}.WiT"], fp[f"{k}.Wo2T"]) if act == "gated_gelu" else (fp[f"{k}.linear1T"], fp[f"{k}.linear2T"])
        weights = (wqkv.size + d * d + wi.size + wo2.size) * bf
        if seq <= LAYER_MAX_SEQ:  # fused.layer: x in, mask, weights, x' out; qkv and att stay in VMEM scratch
            return r * d * f32 + rows * seq * 4 + weights + r * d * f32
        qkv = r * d * f32 + wqkv.size * bf + r * 3 * d * bf
        mlp = r * d * bf + r * d * f32 + (d * d + wi.size + wo2.size) * bf + r * d * f32
        return qkv + attention(window) + mlp

    enc = sum(layer(f"encoder.layers.{i}", cfg.window if t == "sliding_attention" else 0, "gated_gelu") for i, t in enumerate(cfg.layer_types))
    head = sum(layer(f"head.layers.{i}", 0, "relu") for i in range(model.config.head_layers))
    embed = r * d * f32 * 4  # gather + embedding norm + final norm + type_emb add, read / write of the fp32 stream (XLA ops, approximate)
    return {"encoder": enc, "heads": head, "xla_glue": embed, "total": enc + head + embed}


def hlo_fusions(dump_dir: str):
    """Fusion kinds and counts from the optimized HLO module of the forward."""
    files = sorted(glob.glob(dump_dir + "/*_apply*after_optimizations.txt")) or sorted(glob.glob(dump_dir + "/*after_optimizations.txt"))
    if not files:
        return {}
    text = Path(files[-1]).read_text()
    kinds = {}
    for m in re.finditer(r"^\s*%?(\S+) = \S+ fusion\(.*?kind=(k\w+)", text, re.MULTILINE):
        kinds[m.group(2)] = kinds.get(m.group(2), 0) + 1
    custom = len(re.findall(r"custom-call\(", text))
    return {"file": files[-1], "fusions": kinds, "custom_calls": custom, "lines": text.count("\n")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--impl", default="auto")
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--tracer-level", type=int, default=2, help="jax.profiler.ProfileOptions.device_tracer_level")
    args = ap.parse_args()

    dump = tempfile.mkdtemp(prefix="hlo_")
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={dump} --xla_dump_hlo_pass_re=.*after_optimizations"

    import deciax

    d = deciax.load(args.model, impl=args.impl)
    arrays = synthetic(d.model, args.rows, args.seq)
    fwd = lambda: jax.block_until_ready(d.model.forward(arrays))
    for _ in range(5):
        fwd()
    ms = []
    for _ in range(20):
        t = time.perf_counter()
        fwd()
        ms.append((time.perf_counter() - t) * 1000)
    p50 = float(np.median(ms))

    logdir = tempfile.mkdtemp(prefix="xprof_")
    opts = jax.profiler.ProfileOptions()
    opts.device_tracer_level = args.tracer_level
    with jax.profiler.trace(logdir, profiler_options=opts):
        for _ in range(args.calls):
            fwd()
    ops = op_stats(logdir, args.calls)
    if d.model.impl == "pallas":  # one row per kernel kind: the named scopes split them per layer
        agg = {}
        for o in ops:
            key = o["op"].split("/")[-2] if "/pallas_call" in o["op"] else o["op"]
            a = agg.setdefault(key, {**o, "op": key, "ms": 0.0, "hbm_mb": 0.0, "n": 0})
            a["ms"] += o["ms"]
            a["hbm_mb"] += o["hbm_mb"]
            a["n"] += 1
        ops = sorted(agg.values(), key=lambda o: -o["ms"])
    busy = sum(o["ms"] for o in ops if o["type"] != "IDLE")
    hbm = sum(o["hbm_mb"] for o in ops)
    fused_bytes = fused_hbm_bytes(d.model, args.rows, args.seq) if d.model.impl == "pallas" else None
    if fused_bytes:
        hbm = fused_bytes["total"] / 1e6
    dev = jax.devices()[0]
    res = {
        "model": args.model,
        "impl": args.impl,
        "rows": args.rows,
        "seq": args.seq,
        "device": dev.device_kind,
        "p50_ms": p50,
        "device_busy_ms": busy,
        "idle_ms": sum(o["ms"] for o in ops if o["type"] == "IDLE"),
        "hbm_traffic_mb_per_call": hbm,
        "hbm_traffic_source": "kernels: BlockSpec accounting (fused_hbm_bytes); XLA ops: xprof"
        if fused_bytes
        else "xprof framework_op_stats measured bandwidth x self time",
        "fused_hbm_bytes": fused_bytes,
        "hbm_avg_gb_per_s": hbm / p50 if p50 else 0,
        "hlo": hlo_fusions(dump),
        "ops": ops[:40],
    }
    print(
        f"{args.model} rows={args.rows} seq={args.seq} impl={args.impl}: p50 {p50:.2f} ms, device busy {busy:.2f} ms, HBM traffic {hbm:.0f} MB/call = {hbm / p50:.0f} GB/s average"
        + (
            f" (kernels by BlockSpec accounting: encoder {fused_bytes['encoder'] / 1e6:.0f} MB, heads {fused_bytes['heads'] / 1e6:.0f} MB, XLA glue ~{fused_bytes['xla_glue'] / 1e6:.0f} MB)"
            if fused_bytes
            else " (xprof)"
        )
    )
    print(f"HLO: {res['hlo']}")
    print(f"{'op':52s} {'ms':>7s} {'share':>6s} {'HBM MB':>8s} {'GB/s':>6s} {'TF/s':>6s} bound")
    for o in ops[:16]:
        print(
            f"{o['op'][-52:]:52s} {o['ms']:7.2f} {o['ms'] / busy:6.0%} {o['hbm_mb']:8.0f} {o['hbm_gb_per_s']:6.0f} {o['tflops']:6.0f} {o['bound_by']}"
            + (f"  x{o['n']}" if o.get("n", 1) > 1 else "")
        )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
