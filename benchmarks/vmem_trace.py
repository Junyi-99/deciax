"""VMEM occupancy and HBM traffic of one fused forward, as a self-contained HTML timeline.

- VMEM per kernel: Mosaic does not report its allocation, so each kernel instance is compiled with a
  binary-searched `vmem_limit_bytes`; the smallest limit that compiles is what the kernel needs
  (double-buffered blocks + scratch + the kernel body's temporaries). The analytic block / scratch
  part is listed next to it; the remainder is the body's temporaries.
- Timeline: one jitted forward under jax.profiler.trace; the device "XLA Ops" events give start /
  duration per kernel and per XLA op, `bytes_accessed` the HBM bytes of the XLA ops; the kernels'
  HBM bytes come from their BlockSpecs (xprof has no counters for custom calls).

  python -m benchmarks.vmem_trace [--model laya] [--rows 64] [--seq 256] [--out vmem_trace.html]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import html
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.hbm_traffic import synthetic
from deciax.kernels import fused
from deciax.kernels.attention import pallas

MB = 1 << 20
F32, BF16 = 4, 2
KINDS = ("norm_linear_rope", "flash_attention_w64", "flash_attention_w0", "out_mlp_gated_gelu", "norm_linear", "out_mlp_relu")
COLOR = {
    "norm_linear_rope": "#1E40AF",
    "norm_linear": "#60A5FA",
    "flash_attention_w64": "#B91C1C",
    "flash_attention_w0": "#F87171",
    "out_mlp_gated_gelu": "#15803D",
    "out_mlp_relu": "#86EFAC",
    "xla": "#94A3B8",
}


@dataclass
class Kernel:
    kind: str
    grid_steps: int
    blocks_in: dict[str, int]  # name -> bytes per block (double-buffered by the pipeline)
    blocks_out: dict[str, int]
    scratch: int
    hbm_read: int  # bytes per call of this kernel instance
    hbm_write: int
    stays_in_vmem: list[str]
    vmem_measured: int = 0
    substeps: list[tuple[str, int]] = field(default_factory=list)  # (label, live temporaries in bytes on top of resident)
    call: object = None  # (fn, args): fn(*args) builds the pallas_call on representative inputs; args are jit parameters, not constants
    layers: int = 0

    @property
    def blocks_bytes(self) -> int:
        return 2 * (sum(self.blocks_in.values()) + sum(self.blocks_out.values()))

    @property
    def weights_bytes(self) -> int:  # constant-index operands: resident, single-buffered
        return sum(v for n, v in self.blocks_in.items() if "^T" in n or n.startswith(("norm", "bias", "cos")))

    @property
    def stream_bytes(self) -> int:  # row / q / k / v blocks: double-buffered by the pipeline
        return 2 * (sum(self.blocks_in.values()) - self.weights_bytes + sum(self.blocks_out.values()))

    @property
    def resident(self) -> int:
        return self.weights_bytes + self.stream_bytes + self.scratch


def kernels_for(model, rows: int, seq: int) -> dict[str, Kernel]:
    cfg = model.config.encoder
    d, heads, eps = cfg.hidden_size, cfg.num_attention_heads, cfg.norm_eps
    r = rows * seq
    tm = min(fused.TM, r)
    fp = model.fused
    k0, h0 = "encoder.layers.1", "head.layers.0"  # layer 1: has attn_norm (layer 0 skips it)
    x = jnp.zeros((r, d), jnp.float32)
    att = jnp.zeros((r, d), jnp.bfloat16)
    qkv = jnp.zeros((rows, seq, 3 * d), jnp.bfloat16)
    mask = jnp.ones((rows, seq), jnp.int32)
    zeros, f_zeros = fp["zeros"], jnp.zeros((fp[f"{k0}.WiT"].shape[1],), jnp.float32)
    rope = fused.rope_tables(jnp.ones((32,), jnp.float32), seq)
    n_full = sum(t == "full_attention" for t in cfg.layer_types)
    n_slide = len(cfg.layer_types) - n_full
    out: dict[str, Kernel] = {}

    def norm_linear(kind: str, w_t, bias, with_rope: bool, nw, nb, layers: int):
        n = w_t.shape[1]
        blocks_in = {"x f32": tm * d * F32, "norm w, b": 2 * d * F32, "W^T bf16": d * n * BF16, "bias": n * F32}
        if with_rope:
            blocks_in["cos, sin"] = 2 * tm * 128 * F32
        out[kind] = Kernel(
            kind,
            r // tm,
            blocks_in,
            {"qkv bf16": tm * n * BF16},
            0,
            r * d * F32 + d * n * BF16,
            r * n * BF16,
            ["normalized rows (bf16 [tm, D])", "fp32 matmul result before the bf16 cast", "RoPE rotated copy" if with_rope else "bias add"],
            substeps=[
                ("DMA x block [tm, D] fp32 (prefetched)", 0),
                ("LayerNorm -> xn bf16 [tm, D]", tm * d * (F32 + BF16)),
                *[
                    (
                        f"chunk {j}: xn @ W^T[:, {j}D:{j + 1}D] -> bf16" + (" -> RoPE (2 rolls + tiles)" if with_rope and j < 2 else ""),
                        tm * d * BF16 + tm * d * (F32 + BF16) + (tm * d * (BF16 + 2 * F32) if with_rope and j < 2 else 0),
                    )
                    for j in range(n // d)
                ],
                ("store out block [tm, N] bf16 -> HBM", tm * d * BF16),
            ],
            call=(
                lambda x, w_t, bias, nw, nb: fused.norm_linear(
                    x, nw, nb, eps, w_t, bias, rope=rope if with_rope else None, rope_blocks=2 if with_rope else 0, tn=d
                ),
                (x, w_t, bias, nw, nb),
            ),
            layers=layers,
        )

    norm_linear("norm_linear_rope", fp[f"{k0}.WqkvT"], jnp.zeros((3 * d,), jnp.float32), True, fp[f"{k0}.attn_norm"], zeros, len(cfg.layer_types))
    norm_linear(
        "norm_linear",
        fp[f"{h0}.in_projT"],
        model.params[f"{h0}.self_attn.in_proj_bias"],
        False,
        model.params[f"{h0}.norm1.weight"],
        model.params[f"{h0}.norm1.bias"],
        model.config.head_layers,
    )

    def flash(kind: str, window: int, layers: int):
        tq = min(pallas.BLOCK if window else 2 * pallas.BLOCK, seq)
        tk = min(pallas.BLOCK // 2 if window else 2 * pallas.BLOCK, seq)
        nq, nk = seq // tq, seq // tk
        if window:
            steps = min(nk, (tq - 1 + 2 * window) // tk + 2)
            visited = sum(min(((i + 1) * tq - 1 + window) // tk, nk - 1) - max((i * tq - window) // tk, 0) + 1 for i in range(nq))
        else:
            steps, visited = nk, nq * nk
        out[kind] = Kernel(
            kind,
            rows * nq * steps,
            {"q bf16": tq * d * BF16, "k bf16": tk * d * BF16, "v bf16": tk * d * BF16, "mask": tk * 4},
            {"att bf16": tq * d * BF16},
            2 * heads * tq * F32 + heads * 64 * tq * F32,
            rows * (seq * d * BF16 + 2 * visited * tk * d * BF16),
            rows * seq * d * BF16,
            [f"scores s^T fp32 [{tk}, {tq}] per head", "probabilities p (fp32 then bf16)", "running max / sum / accumulator (scratch)", "mask + window bias"],
            substeps=[
                ("DMA q, k, v blocks bf16 + mask", 0),
                ("kk == 0: zero m / l / acc scratch", 0),
                (f"mask + window -> bias fp32 [{tk}, {tq}]", tk * tq * F32),
                (f"per head (x{heads}): s^T = k q^T fp32 [{tk}, {tq}] + bias; m, alpha", tk * tq * F32 * 2),
                ("per head: p = exp(s^T - m) -> bf16; l += sum p; acc += v^T p^T", tk * tq * (F32 * 2 + BF16)),
                ("last key block: o = (acc / l)^T -> out block bf16 -> HBM", tk * tq * F32 + 64 * tq * F32 * 2),
            ],
            call=(lambda qkv, mask: pallas.flash_attention(qkv, mask, heads=heads, window=window), (qkv, mask)),
            layers=layers,
        )

    flash("flash_attention_w64", cfg.window, n_slide)
    flash("flash_attention_w0", 0, n_full + model.config.head_layers)

    def out_mlp(kind: str, k: str, wo_t, bo, nw, nb, wi_t, bi, wo2_t, b2, act: str, layers: int):
        f, fp_ = wi_t.shape[1], wo2_t.shape[0]
        out[kind] = Kernel(
            kind,
            r // tm,
            {
                "att bf16": tm * d * BF16,
                "x f32": tm * d * F32,
                "Wo^T": d * d * BF16,
                "Wi^T": d * f * BF16,
                "Wo2^T": fp_ * d * BF16,
                "norm w, b + biases": (3 * d + f) * F32,
            },
            {"x' f32": tm * d * F32},
            0,
            r * d * (BF16 + F32) + (d * d + d * f + fp_ * d) * BF16,
            r * d * F32,
            [
                "y = att @ Wo + x (fp32 [tm, D])",
                "normalized y (bf16)",
                f"MLP hidden h fp32 [tm, {f}]",
                "gelu(a) * b / relu(h) (bf16)" if act == "gated_gelu" else "relu(h) (bf16)",
            ],
            substeps=[
                ("DMA att block bf16 + x block fp32 (weights resident)", 0),
                ("y = att @ Wo^T + bo + x  fp32 [tm, D]", tm * d * F32),
                ("z = LayerNorm(y) -> bf16 [tm, D]", tm * d * (F32 + BF16)),
                (f"h = z @ Wi^T + bi  fp32 [tm, {f}]", tm * d * (F32 + BF16) + tm * f * F32),
                (
                    f"g = gelu(h[:, :{fp_}]) * h[:, {fp_}:] -> bf16 [tm, {fp_}]" if act == "gated_gelu" else f"g = relu(h) -> bf16 [tm, {fp_}]",
                    tm * d * F32 + tm * f * F32 + tm * fp_ * (F32 + BF16),
                ),
                ("out = g @ Wo2^T + b2 + y -> out block fp32 -> HBM", tm * d * F32 + tm * fp_ * BF16),
            ],
            call=(lambda *a: fused.out_mlp(*a, act=act), (att, x, wo_t, bo, nw, nb, eps, wi_t, bi, wo2_t, b2)),
            layers=layers,
        )

    out_mlp(
        "out_mlp_gated_gelu",
        k0,
        fp[f"{k0}.WoT"],
        zeros,
        fp[f"{k0}.mlp_norm"],
        zeros,
        fp[f"{k0}.WiT"],
        f_zeros,
        fp[f"{k0}.Wo2T"],
        zeros,
        "gated_gelu",
        len(cfg.layer_types),
    )
    p = model.params
    out_mlp(
        "out_mlp_relu",
        h0,
        fp[f"{h0}.out_projT"],
        p[f"{h0}.self_attn.out_proj.bias"],
        p[f"{h0}.norm2.weight"],
        p[f"{h0}.norm2.bias"],
        fp[f"{h0}.linear1T"],
        p[f"{h0}.linear1.bias"],
        fp[f"{h0}.linear2T"],
        p[f"{h0}.linear2.bias"],
        "relu",
        model.config.head_layers,
    )
    return out


def measure_vmem(k: Kernel, lo: int = MB // 2, hi: int = 110 * MB, step: int = MB // 4) -> int:
    """Smallest vmem_limit_bytes (to `step`) at which the kernel compiles."""

    def compiles(limit: int) -> bool:
        fused.VMEM_LIMIT = pallas.VMEM_LIMIT = limit
        jax.clear_caches()  # the jit tracing cache is keyed on the function; a cached trace keeps the old limit
        fn, args = k.call
        try:
            jax.jit(fn, static_argnums=[i for i, a in enumerate(args) if isinstance(a, float)]).lower(*args).compile()
            return True
        except Exception as e:
            if "vmem" not in str(e).lower():
                raise
            return False

    assert compiles(hi), k.kind
    while hi - lo > step:
        mid = (lo + hi) // 2
        if compiles(mid):
            hi = mid
        else:
            lo = mid
    fused.VMEM_LIMIT = pallas.VMEM_LIMIT = 96 * MB
    return hi


def trace_forward(model, arrays) -> list[dict]:
    """Device ops of the last forward call: name, kind, start_us, dur_us, hbm_bytes (XLA ops only)."""
    fwd = lambda: jax.block_until_ready(model.forward(arrays))
    fwd()
    fwd()
    logdir = tempfile.mkdtemp(prefix="xprof_vmem_")
    opts = jax.profiler.ProfileOptions()
    opts.device_tracer_level = 2
    with jax.profiler.trace(logdir, profiler_options=opts):
        fwd()
    with gzip.open(glob.glob(logdir + "/plugins/profile/*/*.trace.json.gz")[0]) as f:
        ev = json.load(f)["traceEvents"]
    pids = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    tids = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    tpu = [e for e in ev if e.get("ph") == "X" and "TPU:0" in pids.get(e["pid"], "")]
    mods = [e for e in tpu if tids.get((e["pid"], e["tid"])) == "XLA Modules" and "_apply" in e["name"]]
    m = max(mods, key=lambda e: e["ts"])
    t0, t1 = m["ts"], m["ts"] + m["dur"]
    ops = []
    for e in tpu:
        if tids.get((e["pid"], e["tid"])) != "XLA Ops" or not (t0 <= e["ts"] <= t1):
            continue
        kind = next((k for k in KINDS if e["name"].startswith(k) and not e["name"].startswith(k + "_")), "xla")
        prefetch = kind == "xla" and any(e["name"].startswith(a) for a in ("copy-start", "copy-done", "slice-start", "slice-done"))
        # async copy / slice ops are XLA prefetching kernel operands (weights, norm vectors, biases) into VMEM (memory
        # space S(1)); the kernels' BlockSpec accounting already counts those bytes, so they are not added again
        ops.append(
            {
                "name": e["name"],
                "kind": kind,
                "prefetch": prefetch,
                "start_us": e["ts"] - t0,
                "dur_us": e["dur"],
                "hbm_bytes": 0 if prefetch else int(e.get("args", {}).get("bytes_accessed", 0) or 0),
            }
        )
    ops.sort(key=lambda o: o["start_us"])
    return ops


def svg_flow(model, rows: int, seq: int, kernels: dict[str, Kernel]) -> str:
    """Architecture as a flow: one box per step of the forward with a 0-128 MiB VMEM bar next to it
    (weights resident | double-buffered blocks | scratch | live temporaries) and the HBM tensors on
    the arrows; then every kernel expanded into the sub-steps of one grid step with the analytic
    live set at each point."""
    cfg = model.config.encoder
    d, r = cfg.hidden_size, rows * seq
    cap = 128 * MB
    bw, bx, x0, row = 560, 430, 16, 34  # bar width, bar x, box x, row height
    seg_color = {"weights": "#334155", "blocks x2": "#64748B", "scratch": "#CBD5E1", "live temporaries": "#F59E0B"}
    kl = kernels

    def bar(y: int, k: Kernel | None, temps: int, note: str = "") -> str:
        if k is None:
            return f'<text x="{bx}" y="{y + 17}" fill="#64748B" font-size="11">XLA op: buffers managed by XLA</text>'
        segs = [("weights", k.weights_bytes), ("blocks x2", k.stream_bytes), ("scratch", k.scratch), ("live temporaries", temps)]
        x, out = bx, [f'<rect x="{bx}" y="{y + 4}" width="{bw}" height="18" fill="#F8FAFC" stroke="#E2E8F0"/>']
        for name, b in segs:
            w = bw * b / cap
            if w > 0:
                out.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="18" fill="{seg_color[name]}"><title>{name}: {b / MB:.1f} MB</title></rect>')
                x += w
        total = sum(b for _, b in segs)
        if k.vmem_measured:
            mx = bx + bw * k.vmem_measured / cap
            out.append(
                f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y}" y2="{y + 26}" stroke="#B91C1C" stroke-width="2"><title>measured kernel VMEM {k.vmem_measured / MB:.1f} MB</title></line>'
            )
        out.append(f'<text x="{bx + bw + 6}" y="{y + 17}" font-size="11" fill="#0F172A">{total / MB:.0f} MB {note}</text>')
        return "".join(out)

    def box(y: int, label: str, color: str, sub: str = "") -> str:
        w = bx - x0 - 12
        return (
            f'<g><title>{html.escape(sub or label)}</title><rect x="{x0}" y="{y}" width="{w}" height="26" rx="4" fill="{color}" opacity="0.15" stroke="{color}"/>'
            f'<text x="{x0 + 8}" y="{y + 17}" font-size="11" font-weight="600" textLength="{min(w - 16, 6.2 * len(label)):.0f}" lengthAdjust="spacingAndGlyphs">{html.escape(label)}</text></g>'
        )

    def arrow(y: int, label: str) -> str:
        return f'<line x1="{x0 + 60}" x2="{x0 + 60}" y1="{y - 8}" y2="{y + 4}" stroke="#94A3B8" marker-end="url(#ah)"/><text x="{x0 + 70}" y="{y + 1}" font-size="10" fill="#B45309">{html.escape(label)}</text>'

    temps = lambda k: max(k.vmem_measured - k.resident, 0)
    n_enc = len(cfg.layer_types)
    steps = [
        ("embedding gather (XLA, SparseCore offload)", None, "", f"x fp32 {r * d * F32 / 1e6:.0f} MB"),
        ("embedding LayerNorm (XLA fusion)", None, "", f"x fp32 {r * d * F32 / 1e6:.0f} MB"),
        (f"encoder layer i (x{n_enc}): norm_linear_rope", kl["norm_linear_rope"], "LN -> @Wqkv -> RoPE", f"qkv bf16 {r * 3 * d * BF16 / 1e6:.0f} MB"),
        (
            "    flash_attention (w64 sliding / w0 full)",
            kl["flash_attention_w64"],
            "online softmax over key blocks; key mask and sliding window built in VMEM",
            f"att bf16 {r * d * BF16 / 1e6:.0f} MB (+ x fp32 {r * d * F32 / 1e6:.0f} MB re-read)",
        ),
        ("    out_mlp_gated_gelu", kl["out_mlp_gated_gelu"], "@Wo + x -> LN -> @Wi -> gelu*b -> @Wo2 + y", f"x fp32 {r * d * F32 / 1e6:.0f} MB -> next layer"),
        ("final LayerNorm + type_emb add (XLA)", None, "", f"x fp32 {r * d * F32 / 1e6:.0f} MB"),
        (f"head layer j (x{model.config.head_layers}): norm_linear", kl["norm_linear"], "LN -> @in_proj + bias", f"qkv bf16 {r * 3 * d * BF16 / 1e6:.0f} MB"),
        ("    flash_attention_w0", kl["flash_attention_w0"], "full attention", f"att bf16 {r * d * BF16 / 1e6:.0f} MB"),
        ("    out_mlp_relu", kl["out_mlp_relu"], "@out_proj + x -> LN -> @linear1 -> relu -> @linear2 + y", f"x fp32 {r * d * F32 / 1e6:.0f} MB"),
        ("scorer + act head (XLA, marker rows only)", None, "", "option / act logits"),
    ]
    parts, y = [], 44
    for label, k, sub, hbm in steps:
        color = COLOR[k.kind] if k else COLOR["xla"]
        parts.append(box(y, label, color, sub))
        parts.append(bar(y, k, temps(k) if k else 0, "measured" if k else ""))
        y += row
        parts.append(arrow(y, "HBM: " + hbm))
        y += 16
    for kind in KINDS:
        k = kl[kind]
        y += 26
        parts.append(
            f'<text x="{x0}" y="{y}" font-size="13" font-weight="600" fill="{COLOR[kind]}">{kind}: one grid step of {k.grid_steps} per call, {k.layers} layers. '
            f"resident = weights {k.weights_bytes / MB:.1f} + blocks x2 {k.stream_bytes / MB:.1f} + scratch {k.scratch / MB:.1f} MB; measured peak {k.vmem_measured / MB:.1f} MB (red tick)</text>"
        )
        y += 10
        for i, (label, live) in enumerate(k.substeps):
            parts.append(box(y, f"{i + 1}. {label}", COLOR[kind]))
            parts.append(bar(y, k, live, f"(live {live / MB:.1f} MB)"))
            y += row
    legend = "".join(
        f'<rect x="{bx + i * 130}" y="12" width="12" height="12" fill="{c}"/><text x="{bx + i * 130 + 16}" y="22" font-size="11">{n}</text>'
        for i, (n, c) in enumerate(seg_color.items())
    )
    legend += f'<line x1="{bx + 530}" x2="{bx + 530}" y1="10" y2="26" stroke="#B91C1C" stroke-width="2"/><text x="{bx + 536}" y="22" font-size="11">measured peak</text>'
    return (
        f'<svg width="1180" height="{y + 20}" font-family="-apple-system,Helvetica,Arial,sans-serif">'
        '<defs><marker id="ah" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto"><path d="M0,0 L6,0 L3,6 z" fill="#94A3B8"/></marker></defs>'
        f'<text x="{x0}" y="22" font-size="13" font-weight="600">Forward pass as a flow; VMEM bar per step, 0 - 128 MiB</text>{legend}'
        + "".join(parts)
        + "</svg>"
    )


def render(model, rows: int, seq: int, kernels: dict[str, Kernel], ops: list[dict], device: str) -> str:
    total_us = max(o["start_us"] + o["dur_us"] for o in ops)
    cap = 128 * MB
    for o in ops:
        k = kernels.get(o["kind"])
        o["vmem"] = k.vmem_measured if k else 0
        if k:
            o["hbm_bytes"] = k.hbm_read + k.hbm_write
    kernel_us = {kind: sum(o["dur_us"] for o in ops if o["kind"] == kind) for kind in KINDS}
    xla_us = sum(o["dur_us"] for o in ops if o["kind"] == "xla")
    n_prefetch = sum(o.get("prefetch", False) for o in ops)
    xla_fusion = [o for o in ops if o["kind"] == "xla" and not o.get("prefetch")]
    hbm_total = sum(o["hbm_bytes"] for o in ops)

    W, LEFT, TOP = 1180, 70, 56
    H1, H2, GAP = 220, 140, 50

    def svg_timeline(lo_us: float, hi_us: float, width: int, h1: int, h2: int, title: str) -> str:
        sx = (width - LEFT - 20) / (hi_us - lo_us)
        parts = [f'<svg width="{width}" height="{TOP + h1 + GAP + h2 + 60}" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="12">']
        parts.append(f'<text x="{LEFT}" y="16" font-weight="600" font-size="14">{html.escape(title)}</text>')
        # VMEM track
        y0, y1 = TOP, TOP + h1
        sy = h1 / cap
        for mb in (0, 32, 64, 96, 128):
            y = y1 - mb * MB * sy
            parts.append(
                f'<line x1="{LEFT}" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E2E8F0"/><text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#475569">{mb} MB</text>'
            )
        parts.append(f'<text x="{LEFT}" y="{y0 - 6}" fill="#0F172A" font-weight="600">VMEM in use (measured per kernel)</text>')
        ylim = y1 - 96 * MB * sy
        parts.append(
            f'<line x1="{LEFT}" x2="{width - 20}" y1="{ylim:.1f}" y2="{ylim:.1f}" stroke="#D97706" stroke-dasharray="4 3"/><text x="{width - 22}" y="{ylim - 3:.1f}" text-anchor="end" fill="#D97706">vmem_limit_bytes 96 MB</text>'
        )
        parts.append(f'<text x="{width - 22}" y="{y0 - 3}" text-anchor="end" fill="#475569">capacity 128 MiB</text>')
        for o in ops:
            s, e = max(o["start_us"], lo_us), min(o["start_us"] + o["dur_us"], hi_us)
            if e <= s:
                continue
            x, w = LEFT + (s - lo_us) * sx, max((e - s) * sx, 0.6)
            if o["vmem"]:
                hh = o["vmem"] * sy
                parts.append(
                    f'<rect x="{x:.2f}" y="{y1 - hh:.1f}" width="{w:.2f}" height="{hh:.1f}" fill="{COLOR[o["kind"]]}" stroke="white" stroke-width="0.3"><title>{html.escape(o["name"])}: {o["dur_us"]:.1f} us, VMEM {o["vmem"] / MB:.1f} MB, HBM {o["hbm_bytes"] / 1e6:.0f} MB</title></rect>'
                )
            else:
                parts.append(
                    f'<rect x="{x:.2f}" y="{y1 - 3}" width="{w:.2f}" height="3" fill="{COLOR["xla"]}"><title>{html.escape(o["name"])} (XLA): {o["dur_us"]:.1f} us, HBM {o["hbm_bytes"] / 1e6:.0f} MB</title></rect>'
                )
        # HBM track
        y2, y3 = y1 + GAP, y1 + GAP + h2
        hbm_max = max(o["hbm_bytes"] for o in ops) or 1
        sy2 = h2 / hbm_max
        for frac in (0, 0.5, 1):
            y = y3 - frac * h2
            parts.append(
                f'<line x1="{LEFT}" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E2E8F0"/><text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#475569">{frac * hbm_max / 1e6:.0f} MB</text>'
            )
        parts.append(
            f'<text x="{LEFT}" y="{y2 - 8}" fill="#0F172A" font-weight="600">HBM bytes per op (kernels: BlockSpec accounting; XLA ops: bytes_accessed)</text>'
        )
        for o in ops:
            s, e = max(o["start_us"], lo_us), min(o["start_us"] + o["dur_us"], hi_us)
            if e <= s or not o["hbm_bytes"]:
                continue
            x, w = LEFT + (s - lo_us) * sx, max((e - s) * sx, 0.6)
            hh = o["hbm_bytes"] * sy2
            parts.append(
                f'<rect x="{x:.2f}" y="{y3 - hh:.1f}" width="{w:.2f}" height="{hh:.1f}" fill="{COLOR[o["kind"]]}" opacity="0.85"><title>{html.escape(o["name"])}: HBM {o["hbm_bytes"] / 1e6:.0f} MB in {o["dur_us"]:.1f} us = {o["hbm_bytes"] / 1e3 / max(o["dur_us"], 1e-3):.0f} GB/s</title></rect>'
            )
        # time axis
        for t in np.linspace(lo_us, hi_us, 9):
            x = LEFT + (t - lo_us) * sx
            parts.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y3}" y2="{y3 + 5}" stroke="#94A3B8"/><text x="{x:.1f}" y="{y3 + 18}" text-anchor="middle" fill="#475569">{t / 1000:.2f} ms</text>'
            )
        parts.append("</svg>")
        return "".join(parts)

    # per-layer zoom: layers 0-2 of the encoder = first three out_mlp_gated_gelu ends
    ends = [o["start_us"] + o["dur_us"] for o in ops if o["kind"] == "out_mlp_gated_gelu"]
    zoom_hi = ends[2] if len(ends) > 2 else total_us
    legend = " ".join(f'<span class="sw" style="background:{COLOR[k]}"></span>{k}' for k in [*KINDS, "xla"])

    rows_html = []
    for kind in KINDS:
        k = kernels[kind]
        temps = k.vmem_measured - k.blocks_bytes - k.scratch  # negative = the pipeline does not double-buffer everything
        blocks = "; ".join(f"{n} {b / MB:.2f}" for n, b in {**k.blocks_in, **k.blocks_out}.items())
        rows_html.append(
            f"<tr><td><span class='sw' style='background:{COLOR[kind]}'></span>{kind}</td><td>{k.layers}</td><td>{k.grid_steps}</td>"
            f"<td><b>{k.vmem_measured / MB:.1f}</b></td><td>{k.blocks_bytes / MB:.1f}<br><small>{blocks} (MB per block, x2 buffers)</small></td>"
            f"<td>{k.scratch / MB:.1f}</td><td>{temps / MB:.1f}</td><td>{k.hbm_read / 1e6:.0f} / {k.hbm_write / 1e6:.0f}</td>"
            f"<td>{kernel_us[kind] / 1000:.2f} ms ({kernel_us[kind] / total_us:.0%})</td><td>{'<br>'.join(html.escape(s) for s in k.stays_in_vmem)}</td></tr>"
        )
    d = model.config.encoder.hidden_size
    r = rows * seq
    gone = [
        ("normalized rows (both norms), bf16", 2 * r * d * BF16),
        ("qkv projection fp32 + bf16 re-cast", r * 3 * d * (F32 + BF16)),
        ("attention scores fp32 [B, H, S, S]", rows * model.config.encoder.num_attention_heads * seq * seq * F32),
        ("attention probabilities bf16", rows * model.config.encoder.num_attention_heads * seq * seq * BF16),
        ("attention output fp32 (now bf16)", r * d * (F32 - BF16)),
        (
            "MLP hidden fp32 + gated output bf16",
            r * (model.fused["encoder.layers.0.WiT"].shape[1] * F32 + model.fused["encoder.layers.0.Wo2T"].shape[0] * BF16),
        ),
    ]
    gone_html = "".join(f"<tr><td>{n}</td><td>{b / 1e6:.0f}</td><td>{2 * b / 1e6:.0f}</td></tr>" for n, b in gone)
    still = [
        (
            "qkv bf16 [R, 3D]",
            r * 3 * d * BF16,
            "norm_linear -> flash_attention: needed because attention of a query block reads every key block of its sequence, and the projection of all positions must exist before the first query block runs",
        ),
        (
            "att bf16 [R, D]",
            r * d * BF16,
            "flash_attention -> out_mlp: a per-row-block kernel boundary; fusable into out_mlp's prologue (the attention output of a row block only depends on that block's queries)",
        ),
        ("residual x fp32 read twice", r * d * F32, "norm_linear and out_mlp both read x; one fused layer kernel would read it once"),
    ]
    still_html = "".join(f"<tr><td>{n}</td><td>{b / 1e6:.0f}</td><td>{html.escape(why)}</td></tr>" for n, b, why in still)
    # one-kernel-per-layer VMEM estimate at this seq (per sequence of `seq` rows, all heads)
    one = (
        seq * d * F32 * 2
        + seq * 3 * d * BF16
        + model.config.encoder.num_attention_heads * seq * seq * F32
        + seq * model.fused["encoder.layers.0.WiT"].shape[1] * F32
        + (d * d * 2 + d * 3 * d + d * model.fused["encoder.layers.0.WiT"].shape[1] + model.fused["encoder.layers.0.Wo2T"].shape[0] * d) * BF16
    )
    vmem_peak = max(k.vmem_measured for k in kernels.values())

    css = """body{font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif;color:#0F172A;max-width:1200px;margin:24px auto;padding:0 16px}
h1{font-size:22px}h2{font-size:17px;margin-top:36px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #E2E8F0;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#F1F5F9}td:nth-child(n+2):nth-child(-n+4),td:nth-child(6),td:nth-child(7){white-space:nowrap}td:nth-child(5){min-width:220px}td:last-child{min-width:260px}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.legend{font-size:12px;color:#475569;margin:6px 0 14px}.so{border-left:4px solid #1E40AF;background:#EFF6FF;padding:8px 14px;margin:12px 0}small{color:#475569}"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VMEM / HBM trace: {model.config.raw.get("model", "laya")} {rows}x{seq} on {device}</title><style>{css}</style></head><body>
<h1>VMEM occupancy and HBM traffic, one fused forward ({rows} rows x {seq} tokens, {device})</h1>
<p class="so"><b>So what:</b> the forward runs {len([o for o in ops if o["kind"] != "xla"])} Pallas kernels in {total_us / 1000:.1f} ms; the kernels hold {total_us and (total_us - xla_us) / total_us:.0%} of the time. Peak VMEM is {vmem_peak / MB:.0f} MB of 128 MiB (out_mlp: {kernels["out_mlp_gated_gelu"].vmem_measured / MB:.0f} MB = the three resident weight matrices, {sum(v for n, v in kernels["out_mlp_gated_gelu"].blocks_in.items() if n.startswith("W")) / MB:.0f} MB, plus double-buffered row blocks and the fp32 MLP hidden of one row block); the kernels never come near the 96 MB limit, so there is headroom for bigger row blocks or a one-kernel-per-layer design at this sequence length. HBM traffic is {hbm_total / 1e9:.1f} GB per call ({hbm_total / 1e3 / total_us:.0f} GB/s average); every per-token intermediate of a layer stays in VMEM, what still crosses HBM is the three cross-kernel tensors per layer (qkv, att, the residual stream) and the weights.</p>
<div class="legend">{legend} &nbsp;|&nbsp; VMEM = smallest <code>vmem_limit_bytes</code> that compiles (binary search, 0.25 MB); XLA ops (grey, no VMEM shown) are the glue between kernels (embedding gather, norms, scorer).</div>
<h2>Architecture: the forward as a flow, VMEM per step</h2>
<p><small>Top: the model-level flow; each step's bar is the VMEM the kernel holds while it runs (resident weights, the pipeline's double-buffered activation blocks, scratch, and the body's live temporaries = measured minus the rest); the orange labels on the arrows are the tensors that cross HBM between steps. Below: each kernel expanded into the sub-steps of one grid step with the analytic live set at that point. The analytic live set is an upper bound (Mosaic frees dead values and streams matmul results), so the measured peak, the red tick, sits below the largest sub-step.</small></p>
{svg_flow(model, rows, seq, kernels)}
<h2>Timeline</h2>
{svg_timeline(0, total_us, W, H1, H2, "Whole forward")}
{svg_timeline(0, zoom_hi, W, H1, H2, "Zoom: encoder layers 0-2 (norm_linear_rope -> flash_attention -> out_mlp per layer)")}
<h2>Per kernel</h2>
<table><tr><th>kernel</th><th>layers</th><th>grid steps / call</th><th>VMEM measured (MB)</th><th>blocks, if all double-buffered (MB)</th><th>scratch (MB)</th><th>temporaries = rest (MB)</th><th>HBM read / write per call (MB)</th><th>time per forward</th><th>stays in VMEM</th></tr>{"".join(rows_html)}</table>
<p><small>Blocks: the "if all double-buffered" column assumes two buffers per input / output block; the measured numbers come out below it for norm_linear and out_mlp, so the pipeline keeps a single buffer for the weights whose block index never changes (the temporaries column goes negative by about one copy of the weights). Temporaries = measured - blocks - scratch is a residual: Mosaic does not expose its allocation of the body's live values (fp32 matmul results, LayerNorm intermediates, scores and probabilities, the MLP hidden).</small></p>
<p><small>Around the kernels XLA runs {n_prefetch} async copy / slice ops that prefetch kernel operands (weights, norm vectors, biases) into VMEM (memory space S(1)) ahead of each custom call, so part of a kernel's operand set is already resident when it starts; their bytes are the same weight reads the BlockSpec accounting counts and are not added to the HBM total. The remaining {len(xla_fusion)} XLA ops (embedding gather, embedding / final norms, type_emb add, scorer) take {sum(o["dur_us"] for o in xla_fusion) / 1000:.2f} ms and move {sum(o["hbm_bytes"] for o in xla_fusion) / 1e6:.0f} MB.</small></p>
<h2>Intermediates that no longer touch HBM (per encoder layer, vs the XLA path)</h2>
<table><tr><th>intermediate</th><th>size (MB)</th><th>HBM traffic saved per layer, write + read (MB)</th></tr>{gone_html}</table>
<h2>What still crosses HBM between kernels (per encoder layer)</h2>
<table><tr><th>tensor</th><th>size (MB)</th><th>why</th></tr>{still_html}</table>
<p>At seq {seq}, one kernel per layer per sequence (grid over sequences, all heads in VMEM) would need about {one / MB:.0f} MB of VMEM for its live values plus weights (residual, qkv, all scores, the MLP hidden, the four weight matrices): it fits in 128 MiB up to about seq 512 and would remove the qkv / att round trips and the second residual read, i.e. ~{(r * 3 * d * BF16 * 2 + r * d * BF16 * 2 + r * d * F32) / 1e6:.0f} MB of the ~{sum(k.hbm_read + k.hbm_write for k in (kernels["norm_linear_rope"], kernels["flash_attention_w64"], kernels["out_mlp_gated_gelu"])) / 1e6:.0f} MB a layer moves now. Beyond that the attention must stay a separate flash kernel.</p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", default="vmem_trace.html")
    args = ap.parse_args()

    import deciax

    d = deciax.load(args.model, impl="pallas")
    kernels = kernels_for(d.model, args.rows, args.seq)
    for k in kernels.values():
        k.vmem_measured = measure_vmem(k)
        print(
            f"{k.kind:22s} VMEM {k.vmem_measured / MB:6.2f} MB = blocks x2 {k.blocks_bytes / MB:6.2f} + scratch {k.scratch / MB:5.2f} + temporaries {(k.vmem_measured - k.blocks_bytes - k.scratch) / MB:6.2f}; HBM r/w {k.hbm_read / 1e6:.0f}/{k.hbm_write / 1e6:.0f} MB per call"
        )
    ops = trace_forward(d.model, synthetic(d.model, args.rows, args.seq))
    page = render(d.model, args.rows, args.seq, kernels, ops, jax.devices()[0].device_kind)
    Path(args.out).write_text(page)
    print(f"{len(ops)} device ops, wrote {args.out}")


if __name__ == "__main__":
    main()
