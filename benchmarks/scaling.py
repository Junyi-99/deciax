"""Forward latency vs batch rows and sequence length, impl="jax" vs impl="pallas", on synthetic
bucketed inputs (benchmarks/hbm_traffic.py builds them). Reports p50 ms, tokens / s and "MFU" =
matmul + full-attention FLOPs / time / bf16 peak; the attention term counts every layer as full
attention, so at long sequences (where the sliding layers skip most key blocks) it overstates
the useful work. Results: benchmarks/results/<chip>/scaling_<model>.txt.

  python benchmarks/scaling.py [--model laya] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from benchmarks.hbm_traffic import synthetic

ROWS = (64, 128, 256)  # at seq 256
SEQS = (256, 512, 1024, 2048, 4096, 8192, 16384)  # at a token budget
TOKENS = 64 * 1024  # rows * seq for the sequence sweep, where rows >= 1
JAX_SCORE_LIMIT = 8 << 30  # skip impl="jax" when its [B, H, S, S] fp32 score tensor would exceed this


def peak_tflops(kind: str) -> float:
    try:
        from jax.experimental.pallas import tpu as pltpu

        return pltpu.get_tpu_info().bf16_ops_per_second / 1e12
    except Exception:  # noqa: BLE001
        return float("nan")


def dense_flops(model, rows: int, seq: int) -> float:
    """2 * tokens * params of the matmul weights (encoder + head layers), plus attention QK / PV."""
    cfg = model.config.encoder
    p = model.params
    mats = sum(v.size for k, v in p.items() if v.ndim == 2 and "embeddings" not in k and "type_emb" not in k)
    layers = len(cfg.layer_types) + model.config.head_layers
    attn = 4 * cfg.hidden_size * seq * layers  # per token: 2 * (QK^T + PV) over S keys
    return 2.0 * rows * seq * mats + rows * seq * attn


def run(model, rows: int, seq: int, reps: int = 10) -> float:
    arrays = synthetic(model, rows, seq)
    fwd = lambda: jax.block_until_ready(model.forward(arrays))
    fwd()
    fwd()
    ms = []
    for _ in range(reps):
        t = time.perf_counter()
        fwd()
        ms.append((time.perf_counter() - t) * 1000)
    return float(np.median(ms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya")
    ap.add_argument("--out", default="")
    ap.add_argument("--impls", default="jax,pallas")
    args = ap.parse_args()

    import deciax

    models = {impl: deciax.load(args.model, impl=impl).model for impl in args.impls.split(",")}
    kind = jax.devices()[0].device_kind
    peak = peak_tflops(kind)
    heads = next(iter(models.values())).config.encoder.num_attention_heads
    shapes = list(dict.fromkeys([(r, 256) for r in ROWS] + [(max(TOKENS // s, 1), s) for s in SEQS]))
    res = []
    print(f"{args.model} on {kind} (bf16 peak {peak:.0f} TFLOP/s)")
    print(f"{'rows':>5s} {'seq':>6s} {'tokens':>7s} | " + " | ".join(f"{impl:>7s} ms {'tok/s':>8s} {'MFU':>4s}" for impl in models) + " | speedup")
    for rows, seq in shapes:
        row = {"rows": rows, "seq": seq, "tokens": rows * seq}
        for impl, m in models.items():
            if impl == "jax" and rows * heads * seq * seq * 4 > JAX_SCORE_LIMIT:
                row[impl] = None
                continue
            try:
                p50 = run(m, rows, seq)
            except Exception as e:  # noqa: BLE001 -- OOM / unsupported shape: record and continue
                row[impl] = {"error": str(e).splitlines()[0][:120]}
                continue
            row[impl] = {"p50_ms": p50, "tokens_per_s": rows * seq / p50 * 1000, "mfu": dense_flops(m, rows, seq) / (p50 / 1000) / 1e12 / peak}
        res.append(row)
        cells = []
        for impl in models:
            r = row[impl]
            cells.append(
                f"{r['p50_ms']:10.1f} {r['tokens_per_s']:8.0f} {r['mfu']:4.0%}"
                if isinstance(r, dict) and "p50_ms" in r
                else f"{'skip' if r is None else 'ERR':>24s}"
            )
        sp = row.get("jax"), row.get("pallas")
        spd = f"{sp[0]['p50_ms'] / sp[1]['p50_ms']:.2f}x" if all(isinstance(x, dict) and "p50_ms" in x for x in sp) else "-"
        print(f"{rows:5d} {seq:6d} {rows * seq:7d} | " + " | ".join(cells) + f" | {spd}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "device": kind, "peak_tflops": peak, "results": res}, indent=1))


if __name__ == "__main__":
    main()
