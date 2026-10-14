"""VMEM occupancy and HBM traffic of one fused forward, as a self-contained HTML page.

- VMEM per kernel: Mosaic does not report its allocation, so each kernel instance is compiled with a
  binary-searched `vmem_limit_bytes`; the smallest limit that compiles is what the kernel needs
  (resident weights + pipelined blocks + scratch + the kernel body's temporaries). The analytic
  block / scratch part is listed next to it, every block with dtype, shape and the meaning of each
  dimension; the remainder is the body's temporaries.
- Timeline: one jitted forward under jax.profiler.trace; the device "XLA Ops" events give start /
  duration per kernel and per XLA op, `bytes_accessed` the HBM bytes of the XLA ops; the kernels'
  HBM bytes come from their BlockSpecs (xprof has no counters for custom calls).
- Architecture: the forward as a flow with a VMEM bar per step, and each kernel expanded into the
  sub-steps of one grid step with the analytic live set.

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
SIZE = {"f32": 4, "bf16": 2, "i32": 4}
# seq <= fused.LAYER_MAX_SEQ: one kernel per layer (encoder sliding / full, head); longer sequences: three kernels per layer
KINDS_LAYER = ("layer_gated_gelu_w64", "layer_gated_gelu_w0", "layer_relu_w0")
KINDS_THREE = ("norm_linear_rope", "flash_attention_w64", "flash_attention_w0", "out_mlp_gated_gelu", "norm_linear", "out_mlp_relu")
ALL_KINDS = KINDS_LAYER + KINDS_THREE
COLOR = {
    "layer_gated_gelu_w64": "#7C3AED",
    "layer_gated_gelu_w0": "#1E40AF",
    "layer_relu_w0": "#15803D",
    "norm_linear_rope": "#1E40AF",
    "norm_linear": "#60A5FA",
    "flash_attention_w64": "#B91C1C",
    "flash_attention_w0": "#F87171",
    "out_mlp_gated_gelu": "#15803D",
    "out_mlp_relu": "#86EFAC",
    "xla": "#94A3B8",
}

# Dimension names used in every shape below; the page's Notation table lists them with the run's values.
DIMS = {
    "B": "sequences in the bucket (one per question; the batch)",
    "S": "positions per sequence (bucketed, padded)",
    "R": "flattened rows = B * S (one row per token; the kernels' row axis)",
    "D": "hidden size = H * Dh",
    "H": "attention heads",
    "Dh": "head size, 64 (fixed by the kernels)",
    "N": "QKV projection width = 3 * D (q | k | v, each H heads of Dh)",
    "F": "MLP hidden width as computed: 2 * Fp for the gated GELU (a | b), Fp for ReLU",
    "Fp": "MLP intermediate width after padding to a multiple of 128 lanes",
    "tm": "rows per grid step: nseq * S for layer (whole sequences), min(512, R) for norm_linear / out_mlp",
    "nseq": "sequences per grid step of layer: min(B, 512 // S)",
    "tq": "query block of flash_attention: positions per grid step",
    "tk": "key block of flash_attention: positions per key step",
    "L": "cos / sin table rows: S for layer; tm if a row block holds whole sequences, else S, for norm_linear",
}
SHORT = {n: m.split(":")[0].split(" (")[0].split(" = ")[0] for n, m in DIMS.items()}  # short meaning for inline shapes


@dataclass(frozen=True)
class Tensor:
    """A block or tensor: dtype and shape as ((dim name, value, meaning), ...)."""

    dtype: str
    dims: tuple[tuple[str, int, str], ...]

    @property
    def bytes(self) -> int:
        n = SIZE[self.dtype]
        for _, v, _ in self.dims:
            n *= v
        return n

    @property
    def shape(self) -> str:  # f32[tm=512, D=1024]
        return f"{self.dtype}[{', '.join(f'{n}={v}' for n, v, _ in self.dims)}]"

    @property
    def long(self) -> str:  # f32[tm=512 (rows per grid step), D=1024 (hidden size)]
        return f"{self.dtype}[{', '.join(f'{n}={v} ({m})' for n, v, m in self.dims)}]"


def dim(name: str, value: int, meaning: str | None = None) -> tuple[str, int, str]:
    return (name, value, meaning if meaning is not None else SHORT[name])


@dataclass
class Kernel:
    kind: str
    grid_steps: int
    blocks: dict[str, Tensor]  # every pipelined operand / result block of one grid step
    resident_names: tuple[str, ...]  # blocks with a constant block index: fetched once, single-buffered
    out_names: tuple[str, ...]
    scratch: dict[str, Tensor]
    hbm_read: int  # bytes per call of this kernel instance
    hbm_write: int
    stays_in_vmem: list[tuple[str, Tensor]]  # (what, tensor) the kernel body keeps on chip
    substeps: list[tuple[str, int]] = field(default_factory=list)  # (label, live temporaries in bytes on top of `resident`)
    vmem_measured: int = 0
    call: object = None  # (fn, args): fn(*args) builds the pallas_call on representative inputs; args are jit parameters
    layers: int = 0

    @property
    def weights_bytes(self) -> int:
        return sum(self.blocks[n].bytes for n in self.resident_names)

    @property
    def stream_bytes(self) -> int:  # the pipeline double-buffers the blocks whose index changes
        return 2 * sum(t.bytes for n, t in self.blocks.items() if n not in self.resident_names)

    @property
    def scratch_bytes(self) -> int:
        return sum(t.bytes for t in self.scratch.values())

    @property
    def resident(self) -> int:
        return self.weights_bytes + self.stream_bytes + self.scratch_bytes


def kernels_for(model, rows: int, seq: int) -> dict[str, Kernel]:
    cfg = model.config.encoder
    d, heads, eps = cfg.hidden_size, cfg.num_attention_heads, cfg.norm_eps
    r = rows * seq
    tm = min(fused.TM, r)
    fp = model.fused
    p = model.params
    k0, h0 = "encoder.layers.1", "head.layers.0"  # layer 1 has an attn_norm (layer 0 skips it); head layer 0
    x = jnp.zeros((r, d), jnp.float32)
    att = jnp.zeros((r, d), jnp.bfloat16)
    qkv = jnp.zeros((rows, seq, 3 * d), jnp.bfloat16)
    mask = jnp.ones((rows, seq), jnp.int32)
    zeros, f_zeros = fp["zeros"], jnp.zeros((fp[f"{k0}.WiT"].shape[1],), jnp.float32)
    rope = fused.rope_tables(jnp.ones((32,), jnp.float32), seq)
    n_full = sum(t == "full_attention" for t in cfg.layer_types)
    n_slide = len(cfg.layer_types) - n_full
    out: dict[str, Kernel] = {}
    TM, Dd, Hh, DH = dim("tm", tm), dim("D", d), dim("H", heads), dim("Dh", 64)

    def norm_linear(kind: str, w_t, bias, with_rope: bool, nw, nb, layers: int):
        n = w_t.shape[1]
        NN = dim("N", n)
        blocks = {
            "x": Tensor("f32", (TM, Dd)),
            "norm w, b": Tensor("f32", (dim("2", 2, "weight and bias"), Dd)),
            "W^T": Tensor("bf16", (Dd, NN)),
            "bias": Tensor("f32", (NN,)),
            "out qkv": Tensor("bf16", (TM, NN)),
        }
        resident = ("norm w, b", "W^T", "bias")
        if with_rope:
            blocks["cos, sin"] = Tensor(
                "f32", (dim("2", 2, "cos and sin"), dim("L", max(tm, seq)), dim("128", 128, "one head's rotate-half pattern, duplicated for 2 heads per vreg"))
            )
            resident += ("cos, sin",)
        out[kind] = Kernel(
            kind,
            r // tm,
            blocks,
            resident,
            ("out qkv",),
            {},
            r * d * 4 + d * n * 2,
            r * n * 2,
            [
                ("normalized rows xn", Tensor("bf16", (TM, Dd))),
                ("fp32 matmul result of one chunk, before the bf16 cast", Tensor("f32", (TM, Dd))),
                ("RoPE rotated copy and cos / sin tiles" if with_rope else "bias add", Tensor("bf16" if with_rope else "f32", (TM, Dd))),
            ],
            substeps=[
                (f"DMA x block {blocks['x'].shape} (prefetched)", 0),
                (f"LayerNorm -> xn {Tensor('bf16', (TM, Dd)).shape}", tm * d * 6),
                *[
                    (
                        f"chunk {j}: xn @ W^T[:, {j}D:{j + 1}D] -> bf16[tm, D]" + (" -> RoPE (2 rolls + tiles)" if with_rope and j < 2 else ""),
                        tm * d * 2 + tm * d * 6 + (tm * d * 10 if with_rope and j < 2 else 0),
                    )
                    for j in range(n // d)
                ],
                (f"store out block {blocks['out qkv'].shape} -> HBM", tm * d * 2),
            ],
            call=(
                lambda x, w_t, bias, nw, nb: fused.norm_linear(
                    x, nw, nb, eps, w_t, bias, rope=rope if with_rope else None, rope_blocks=2 if with_rope else 0, tn=d
                ),
                (x, w_t, bias, nw, nb),
            ),
            layers=layers,
        )

    three = seq > fused.LAYER_MAX_SEQ
    if three:
        norm_linear("norm_linear_rope", fp[f"{k0}.WqkvT"], jnp.zeros((3 * d,), jnp.float32), True, fp[f"{k0}.attn_norm"], zeros, len(cfg.layer_types))
        norm_linear(
            "norm_linear",
            fp[f"{h0}.in_projT"],
            p[f"{h0}.self_attn.in_proj_bias"],
            False,
            p[f"{h0}.norm1.weight"],
            p[f"{h0}.norm1.bias"],
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
        TQ, TK = dim("tq", tq), dim("tk", tk)
        blocks = {
            "q": Tensor("bf16", (TQ, Dd)),
            "k": Tensor("bf16", (TK, Dd)),
            "v": Tensor("bf16", (TK, Dd)),
            "mask": Tensor("i32", (dim("1", 1, "row"), TK)),
            "out att": Tensor("bf16", (TQ, Dd)),
        }
        scratch = {
            "m (running max)": Tensor("f32", (Hh, dim("1", 1, "row"), TQ)),
            "l (running sum)": Tensor("f32", (Hh, dim("1", 1, "row"), TQ)),
            "acc (o^T accumulator)": Tensor("f32", (Hh, DH, TQ)),
        }
        out[kind] = Kernel(
            kind,
            rows * nq * steps,
            blocks,
            (),
            ("out att",),
            scratch,
            rows * (seq * d * 2 + 2 * visited * tk * d * 2),
            rows * seq * d * 2,
            [
                ("scores s^T, one head", Tensor("f32", (TK, TQ))),
                ("probabilities p^T, one head (fp32, then bf16 for the MXU)", Tensor("f32", (TK, TQ))),
                ("mask + window bias (shared by all heads)", Tensor("f32", (TK, TQ))),
            ],
            substeps=[
                ("DMA q, k, v blocks bf16 + mask", 0),
                ("kk == 0: zero m / l / acc scratch", 0),
                (f"mask + window -> bias {Tensor('f32', (TK, TQ)).shape}", tk * tq * 4),
                (f"per head (x{heads}): s^T = k q^T {Tensor('f32', (TK, TQ)).shape} + bias; m, alpha", tk * tq * 8),
                ("per head: p = exp(s^T - m) -> bf16; l += sum p; acc += v^T p^T", tk * tq * 10),
                (f"last key block: o = (acc / l)^T -> out block {blocks['out att'].shape} -> HBM", tk * tq * 4 + 64 * tq * 8),
            ],
            call=(lambda qkv, mask: pallas.flash_attention(qkv, mask, heads=heads, window=window), (qkv, mask)),
            layers=layers,
        )

    if three:
        flash("flash_attention_w64", cfg.window, n_slide)
        flash("flash_attention_w0", 0, n_full + model.config.head_layers)

    def out_mlp(kind: str, wo_t, bo, nw, nb, wi_t, bi, wo2_t, b2, act: str, layers: int):
        f, fp_ = wi_t.shape[1], wo2_t.shape[0]
        Ff, Fp = dim("F", f), dim("Fp", fp_)
        blocks = {
            "att": Tensor("bf16", (TM, Dd)),
            "x": Tensor("f32", (TM, Dd)),
            "Wo^T": Tensor("bf16", (Dd, Dd)),
            "Wi^T": Tensor("bf16", (Dd, Ff)),
            "Wo2^T": Tensor("bf16", (Fp, Dd)),
            "norm w, b + biases": Tensor("f32", (dim("3D+F", 3 * d + f, "norm weight, norm bias, bo, b2 and bi"),)),
            "out x'": Tensor("f32", (TM, Dd)),
        }
        g_label = (
            f"g = gelu(h[:, :Fp]) * h[:, Fp:] -> {Tensor('bf16', (TM, Fp)).shape}"
            if act == "gated_gelu"
            else f"g = relu(h) -> {Tensor('bf16', (TM, Fp)).shape}"
        )
        out[kind] = Kernel(
            kind,
            r // tm,
            blocks,
            ("Wo^T", "Wi^T", "Wo2^T", "norm w, b + biases"),
            ("out x'",),
            {},
            r * d * 6 + (d * d + d * f + fp_ * d) * 2,
            r * d * 4,
            [
                ("y = att @ Wo + bo + x", Tensor("f32", (TM, Dd))),
                ("normalized y", Tensor("bf16", (TM, Dd))),
                ("MLP hidden h", Tensor("f32", (TM, Ff))),
                ("gelu(a) * b" if act == "gated_gelu" else "relu(h)", Tensor("bf16", (TM, Fp))),
            ],
            substeps=[
                ("DMA att block bf16 + x block fp32 (weights resident)", 0),
                (f"y = att @ Wo^T + bo + x  {Tensor('f32', (TM, Dd)).shape}", tm * d * 4),
                (f"z = LayerNorm(y) -> {Tensor('bf16', (TM, Dd)).shape}", tm * d * 6),
                (f"h = z @ Wi^T + bi  {Tensor('f32', (TM, Ff)).shape}", tm * d * 6 + tm * f * 4),
                (g_label, tm * d * 4 + tm * f * 4 + tm * fp_ * 6),
                (f"out = g @ Wo2^T + b2 + y -> out block {blocks['out x' + chr(39)].shape} -> HBM", tm * d * 4 + tm * fp_ * 2),
            ],
            call=(lambda *a: fused.out_mlp(*a, act=act), (att, x, wo_t, bo, nw, nb, eps, wi_t, bi, wo2_t, b2)),
            layers=layers,
        )

    if three:
        out_mlp(
            "out_mlp_gated_gelu",
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
        out_mlp(
            "out_mlp_relu",
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

    def layer_kernel(kind: str, window: int, act: str, with_rope: bool, w: tuple, layers: int):
        """`fused.layer`: one grid step = nseq whole sequences; every intermediate of the layer stays in VMEM."""
        nw1, nb1, wqkv_t, bqkv, wo_t, bo, nw2, nb2, wi_t, bi, wo2_t, b2 = w
        n, f, fp_ = wqkv_t.shape[1], wi_t.shape[1], wo2_t.shape[0]
        nseq = min(rows, max(1, fused.TM // seq))
        tml = nseq * seq
        TML, NN, Ff, Fp, SS = dim("tm", tml), dim("N", n), dim("F", f), dim("Fp", fp_), dim("S", seq)
        blocks = {
            "x": Tensor("f32", (TML, Dd)),
            "mask": Tensor("i32", (dim("nseq", nseq), dim("1", 1, "row"), SS)),
            "norm1 w, b": Tensor("f32", (dim("2", 2, "weight and bias"), Dd)),
            "Wqkv^T": Tensor("bf16", (Dd, NN)),
            "b_qkv": Tensor("f32", (NN,)),
            "cos, sin": Tensor(
                "f32", (dim("2", 2, "cos and sin"), dim("L", seq), dim("128", 128, "one head's rotate-half pattern, duplicated for 2 heads per vreg"))
            ),
            "Wo^T": Tensor("bf16", (Dd, Dd)),
            "norm2 w, b": Tensor("f32", (dim("2", 2, "weight and bias"), Dd)),
            "Wi^T": Tensor("bf16", (Dd, Ff)),
            "Wo2^T": Tensor("bf16", (Fp, Dd)),
            "bo, b2, bi": Tensor("f32", (dim("2D+F", 2 * d + f, "the three biases"),)),
            "out x'": Tensor("f32", (TML, Dd)),
        }
        resident = ("norm1 w, b", "Wqkv^T", "b_qkv", "cos, sin", "Wo^T", "norm2 w, b", "Wi^T", "Wo2^T", "bo, b2, bi")
        scratch = {
            "qkv_sc (q | k | v after RoPE, all heads)": Tensor("bf16", (TML, NN)),
            "att_sc (attention output, all heads)": Tensor("bf16", (TML, Dd)),
        }
        wbytes = (d * n + d * d + d * f + fp_ * d) * 2
        x_live = tml * d * 4  # the x block stays live through the attention for the residual add
        ss = seq * seq
        g_label = (
            f"g = gelu(h[:, :Fp]) * h[:, Fp:] -> {Tensor('bf16', (TML, Fp)).shape}"
            if act == "gated_gelu"
            else f"g = relu(h) -> {Tensor('bf16', (TML, Fp)).shape}"
        )
        out[kind] = Kernel(
            kind,
            r // tml,
            blocks,
            resident,
            ("out x'",),
            scratch,
            r * d * 4 + rows * seq * 4 + wbytes,
            r * d * 4,
            [
                ("normalized rows xn", Tensor("bf16", (TML, Dd))),
                ("fp32 matmul result of one QKV chunk, before the bf16 cast", Tensor("f32", (TML, Dd))),
                ("RoPE rotated copy and cos / sin tiles" if with_rope else "bias add", Tensor("bf16" if with_rope else "f32", (TML, Dd))),
                ("mask + window bias, one sequence (shared by all heads)", Tensor("f32", (SS, SS))),
                ("scores s^T = k q^T, one sequence and head (whole S x S: single-pass softmax, no online rescaling)", Tensor("f32", (SS, SS))),
                ("probabilities p^T (fp32, then bf16 for the MXU)", Tensor("f32", (SS, SS))),
                ("o^T = v^T p^T, one head", Tensor("f32", (DH, SS))),
                ("y = att @ Wo + bo + x", Tensor("f32", (TML, Dd))),
                ("normalized y", Tensor("bf16", (TML, Dd))),
                ("MLP hidden h", Tensor("f32", (TML, Ff))),
                ("gelu(a) * b" if act == "gated_gelu" else "relu(h)", Tensor("bf16", (TML, Fp))),
            ],
            substeps=[
                (f"DMA x block {blocks['x'].shape} + mask (prefetched); weights resident", 0),
                (f"LayerNorm -> xn {Tensor('bf16', (TML, Dd)).shape}", x_live + tml * d * 6),
                *[
                    (
                        f"chunk {j}: xn @ Wqkv^T[:, {j}D:{j + 1}D] + b -> bf16[tm, D]"
                        + (" -> RoPE (2 rolls + tiles)" if with_rope and j < 2 else "")
                        + " -> qkv_sc",
                        x_live + tml * d * 2 + tml * d * 6 + (tml * d * 10 if with_rope and j < 2 else 0),
                    )
                    for j in range(3)
                ],
                (f"per sequence (x{nseq}): mask + window -> bias {Tensor('f32', (SS, SS)).shape}", x_live + ss * 4),
                (f"per head (x{heads}): s^T = k q^T {Tensor('f32', (SS, SS)).shape} + bias; p = exp(s^T - max); l = sum p", x_live + ss * 4 * 3),
                (
                    f"per head: o^T = v^T p^T {Tensor('f32', (DH, SS)).shape}; att_sc[rows, head] = (o^T / l)^T bf16",
                    x_live + ss * 4 * 2 + ss * 2 + 64 * seq * 8,
                ),
                (f"y = att_sc @ Wo^T + bo + x  {Tensor('f32', (TML, Dd)).shape}", x_live + tml * d * 4),
                (f"z = LayerNorm(y) -> {Tensor('bf16', (TML, Dd)).shape}", tml * d * 4 + tml * d * 6),
                (f"h = z @ Wi^T + bi  {Tensor('f32', (TML, Ff)).shape}", tml * d * 6 + tml * f * 4),
                (g_label, tml * d * 4 + tml * f * 4 + tml * fp_ * 6),
                (f"out = g @ Wo2^T + b2 + y -> out block {blocks['out x' + chr(39)].shape} -> HBM", tml * d * 8 + tml * fp_ * 2),
            ],
            call=(
                lambda x, mask, nw1, nb1, wqkv_t, bqkv, cos, sin, wo_t, bo, nw2, nb2, wi_t, bi, wo2_t, b2: fused.layer(
                    x,
                    mask,
                    nw1,
                    nb1,
                    wqkv_t,
                    bqkv,
                    fused.Rope(cos, sin, seq) if with_rope else None,
                    wo_t,
                    bo,
                    nw2,
                    nb2,
                    wi_t,
                    bi,
                    wo2_t,
                    b2,
                    seq=seq,
                    heads=heads,
                    window=window,
                    eps=eps,
                    act=act,
                ),
                (x, mask, nw1, nb1, wqkv_t, bqkv, rope.cos, rope.sin, wo_t, bo, nw2, nb2, wi_t, bi, wo2_t, b2),
            ),
            layers=layers,
        )

    enc_w = (
        fp[f"{k0}.attn_norm"],
        zeros,
        fp[f"{k0}.WqkvT"],
        jnp.zeros((3 * d,), jnp.float32),
        fp[f"{k0}.WoT"],
        zeros,
        fp[f"{k0}.mlp_norm"],
        zeros,
        fp[f"{k0}.WiT"],
        f_zeros,
        fp[f"{k0}.Wo2T"],
        zeros,
    )
    head_w = (
        p[f"{h0}.norm1.weight"],
        p[f"{h0}.norm1.bias"],
        fp[f"{h0}.in_projT"],
        p[f"{h0}.self_attn.in_proj_bias"],
        fp[f"{h0}.out_projT"],
        p[f"{h0}.self_attn.out_proj.bias"],
        p[f"{h0}.norm2.weight"],
        p[f"{h0}.norm2.bias"],
        fp[f"{h0}.linear1T"],
        p[f"{h0}.linear1.bias"],
        fp[f"{h0}.linear2T"],
        p[f"{h0}.linear2.bias"],
    )
    layer_kernel("layer_gated_gelu_w64", cfg.window, "gated_gelu", True, enc_w, n_slide)
    layer_kernel("layer_gated_gelu_w0", 0, "gated_gelu", True, enc_w, n_full)
    layer_kernel("layer_relu_w0", 0, "relu", False, head_w, model.config.head_layers)
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
        kind = next((k for k in ALL_KINDS if e["name"].startswith(k) and not e["name"].startswith(k + "_")), "xla")
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
    """The forward as a flow: one box per step with a 0-128 MiB VMEM bar (weights resident |
    double-buffered blocks | scratch | live temporaries) and the HBM tensors on the arrows; then
    every kernel expanded into the sub-steps of one grid step with the analytic live set."""
    cfg = model.config.encoder
    d, r = cfg.hidden_size, rows * seq
    cap = 128 * MB
    bw, bx, x0, row = 560, 430, 16, 34
    seg_color = {"weights": "#334155", "blocks x2": "#64748B", "scratch": "#CBD5E1", "live temporaries": "#F59E0B"}
    kl = kernels

    def bar(y: int, k: Kernel | None, temps: int, note: str = "") -> str:
        if k is None:
            return f'<text x="{bx}" y="{y + 17}" fill="#64748B" font-size="11">XLA op: buffers managed by XLA</text>'
        segs = [("weights", k.weights_bytes), ("blocks x2", k.stream_bytes), ("scratch", k.scratch_bytes), ("live temporaries", temps)]
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
    xs = f"x f32[R={r}, D={d}] {r * d * 4 / 1e6:.0f} MB"
    qkvs = f"qkv bf16[R={r}, N={3 * d}] {r * 3 * d * 2 / 1e6:.0f} MB"
    atts = f"att bf16[R={r}, D={d}] {r * d * 2 / 1e6:.0f} MB"
    n_full = sum(t == "full_attention" for t in cfg.layer_types)
    mlp = "@Wo + x -> LayerNorm -> @Wi -> gelu(a) * b -> @Wo2 + y"
    if "layer_gated_gelu_w0" in kl:
        steps = [
            ("embedding gather (XLA, SparseCore offload)", None, "", xs),
            ("embedding LayerNorm (XLA fusion)", None, "", xs),
            (
                f"encoder layer i, sliding (x{n_enc - n_full}): layer_gated_gelu_w64",
                kl["layer_gated_gelu_w64"],
                f"LayerNorm -> @Wqkv -> RoPE -> attention, window {cfg.window}, all heads -> {mlp}",
                f"{xs} -> next layer",
            ),
            (
                f"encoder layer i, full (x{n_full}): layer_gated_gelu_w0",
                kl["layer_gated_gelu_w0"],
                f"LayerNorm -> @Wqkv -> RoPE -> full attention, all heads -> {mlp}",
                f"{xs} -> next layer",
            ),
            ("final LayerNorm + type_emb add (XLA)", None, "", xs),
            (
                f"head layer j (x{model.config.head_layers}): layer_relu_w0",
                kl["layer_relu_w0"],
                "LayerNorm -> @in_proj + bias -> full attention -> @out_proj + x -> LayerNorm -> @linear1 -> relu -> @linear2 + y",
                xs,
            ),
            ("scorer + act head (XLA, marker rows only)", None, "", "option / act logits"),
        ]
    else:
        steps = [
            ("embedding gather (XLA, SparseCore offload)", None, "", xs),
            ("embedding LayerNorm (XLA fusion)", None, "", xs),
            (f"encoder layer i (x{n_enc}): norm_linear_rope", kl["norm_linear_rope"], "LayerNorm -> @Wqkv -> RoPE on the q and k heads", qkvs),
            (
                "    flash_attention (w64 sliding / w0 full)",
                kl["flash_attention_w64"],
                "online softmax over key blocks; key mask and sliding window built in VMEM",
                f"{atts} (+ {xs} re-read)",
            ),
            ("    out_mlp_gated_gelu", kl["out_mlp_gated_gelu"], "@Wo + x -> LayerNorm -> @Wi -> gelu(a) * b -> @Wo2 + y", f"{xs} -> next layer"),
            ("final LayerNorm + type_emb add (XLA)", None, "", xs),
            (f"head layer j (x{model.config.head_layers}): norm_linear", kl["norm_linear"], "LayerNorm -> @in_proj + bias", qkvs),
            ("    flash_attention_w0", kl["flash_attention_w0"], "full attention", atts),
            ("    out_mlp_relu", kl["out_mlp_relu"], "@out_proj + x -> LayerNorm -> @linear1 -> relu -> @linear2 + y", xs),
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
    for kind, k in kl.items():
        y += 26
        parts.append(
            f'<text x="{x0}" y="{y}" font-size="13" font-weight="600" fill="{COLOR[kind]}">{kind}: one grid step of {k.grid_steps} per call, {k.layers} layers. '
            f"resident = weights {k.weights_bytes / MB:.1f} + blocks x2 {k.stream_bytes / MB:.1f} + scratch {k.scratch_bytes / MB:.1f} MB; measured peak {k.vmem_measured / MB:.1f} MB (red tick)</text>"
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


def svg_timeline(ops: list[dict], lo_us: float, hi_us: float, title: str) -> str:
    cap = 128 * MB
    width, left, top, h1, h2, gap = 1180, 70, 56, 220, 140, 50
    sx = (width - left - 20) / (hi_us - lo_us)
    parts = [f'<svg width="{width}" height="{top + h1 + gap + h2 + 60}" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="12">']
    parts.append(f'<text x="{left}" y="16" font-weight="600" font-size="14">{html.escape(title)}</text>')
    y0, y1 = top, top + h1
    sy = h1 / cap
    for mb in (0, 32, 64, 96, 128):
        y = y1 - mb * MB * sy
        parts.append(
            f'<line x1="{left}" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E2E8F0"/><text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#475569">{mb} MB</text>'
        )
    parts.append(f'<text x="{left}" y="{y0 - 6}" fill="#0F172A" font-weight="600">VMEM in use (measured per kernel)</text>')
    ylim = y1 - 96 * MB * sy
    parts.append(
        f'<line x1="{left}" x2="{width - 20}" y1="{ylim:.1f}" y2="{ylim:.1f}" stroke="#D97706" stroke-dasharray="4 3"/><text x="{width - 22}" y="{ylim - 3:.1f}" text-anchor="end" fill="#D97706">vmem_limit_bytes 96 MB</text>'
    )
    parts.append(f'<text x="{width - 22}" y="{y0 - 3}" text-anchor="end" fill="#475569">capacity 128 MiB</text>')
    for o in ops:
        s, e = max(o["start_us"], lo_us), min(o["start_us"] + o["dur_us"], hi_us)
        if e <= s:
            continue
        x, w = left + (s - lo_us) * sx, max((e - s) * sx, 0.6)
        if o["vmem"]:
            hh = o["vmem"] * sy
            parts.append(
                f'<rect x="{x:.2f}" y="{y1 - hh:.1f}" width="{w:.2f}" height="{hh:.1f}" fill="{COLOR[o["kind"]]}" stroke="white" stroke-width="0.3"><title>{html.escape(o["name"])}: {o["dur_us"]:.1f} us, VMEM {o["vmem"] / MB:.1f} MB, HBM {o["hbm_bytes"] / 1e6:.0f} MB</title></rect>'
            )
        else:
            parts.append(
                f'<rect x="{x:.2f}" y="{y1 - 3}" width="{w:.2f}" height="3" fill="{COLOR["xla"]}"><title>{html.escape(o["name"])} (XLA): {o["dur_us"]:.1f} us, HBM {o["hbm_bytes"] / 1e6:.0f} MB</title></rect>'
            )
    y2, y3 = y1 + gap, y1 + gap + h2
    hbm_max = max(o["hbm_bytes"] for o in ops) or 1
    sy2 = h2 / hbm_max
    for frac in (0, 0.5, 1):
        y = y3 - frac * h2
        parts.append(
            f'<line x1="{left}" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E2E8F0"/><text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#475569">{frac * hbm_max / 1e6:.0f} MB</text>'
        )
    parts.append(
        f'<text x="{left}" y="{y2 - 8}" fill="#0F172A" font-weight="600">HBM bytes per op (kernels: BlockSpec accounting; XLA ops: bytes_accessed)</text>'
    )
    for o in ops:
        s, e = max(o["start_us"], lo_us), min(o["start_us"] + o["dur_us"], hi_us)
        if e <= s or not o["hbm_bytes"]:
            continue
        x, w = left + (s - lo_us) * sx, max((e - s) * sx, 0.6)
        hh = o["hbm_bytes"] * sy2
        parts.append(
            f'<rect x="{x:.2f}" y="{y3 - hh:.1f}" width="{w:.2f}" height="{hh:.1f}" fill="{COLOR[o["kind"]]}" opacity="0.85"><title>{html.escape(o["name"])}: HBM {o["hbm_bytes"] / 1e6:.0f} MB in {o["dur_us"]:.1f} us = {o["hbm_bytes"] / 1e3 / max(o["dur_us"], 1e-3):.0f} GB/s</title></rect>'
        )
    for t in np.linspace(lo_us, hi_us, 9):
        x = left + (t - lo_us) * sx
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y3}" y2="{y3 + 5}" stroke="#94A3B8"/><text x="{x:.1f}" y="{y3 + 18}" text-anchor="middle" fill="#475569">{t / 1000:.2f} ms</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render(model, rows: int, seq: int, kernels: dict[str, Kernel], ops: list[dict], device: str) -> str:
    cfg = model.config.encoder
    d, heads = cfg.hidden_size, cfg.num_attention_heads
    r = rows * seq
    f, fp_ = model.fused["encoder.layers.0.WiT"].shape[1], model.fused["encoder.layers.0.Wo2T"].shape[0]
    total_us = max(o["start_us"] + o["dur_us"] for o in ops)
    for o in ops:
        k = kernels.get(o["kind"])
        o["vmem"] = k.vmem_measured if k else 0
        if k:
            o["hbm_bytes"] = k.hbm_read + k.hbm_write
    one_kernel = "layer_gated_gelu_w0" in kernels
    kernel_us = {kind: sum(o["dur_us"] for o in ops if o["kind"] == kind) for kind in kernels}
    xla_us = sum(o["dur_us"] for o in ops if o["kind"] == "xla")
    hbm_total = sum(o["hbm_bytes"] for o in ops)
    n_prefetch = sum(o.get("prefetch", False) for o in ops)
    xla_fusion = [o for o in ops if o["kind"] == "xla" and not o.get("prefetch")]
    enc_last = ("layer_gated_gelu_w64", "layer_gated_gelu_w0") if one_kernel else ("out_mlp_gated_gelu",)
    ends = sorted(o["start_us"] + o["dur_us"] for o in ops if o["kind"] in enc_last)
    zoom_hi = ends[2] if len(ends) > 2 else total_us
    legend = " ".join(f'<span class="sw" style="background:{COLOR[k]}"></span>{k}' for k in [*kernels, "xla"])
    peak_kind = max(kernels, key=lambda k: kernels[k].vmem_measured)
    peak = kernels[peak_kind]
    vmem_peak = peak.vmem_measured
    nseq = min(rows, max(1, fused.TM // seq))
    tm = nseq * seq if one_kernel else min(fused.TM, r)

    values = {
        "B": rows,
        "S": seq,
        "R": r,
        "D": d,
        "H": heads,
        "Dh": 64,
        "N": 3 * d,
        "F": f,
        "Fp": fp_,
        "tm": tm,
        "nseq": nseq if one_kernel else "n/a (three-kernel path)",
        "tq": "n/a: the whole sequence, S x S scores per head in one pass" if one_kernel else f"{min(256, seq)} (window layers) / {min(512, seq)} (full)",
        "tk": "n/a" if one_kernel else f"{min(128, seq)} (window layers) / {min(512, seq)} (full)",
        "L": seq if one_kernel else max(tm, seq),
    }
    notation = "".join(f"<tr><td><code>{n}</code></td><td>{values[n]}</td><td>{html.escape(m)}</td></tr>" for n, m in DIMS.items())

    def tensor_rows(k: Kernel) -> str:
        out = []
        for name, t in k.blocks.items():
            role = (
                "weight / constant: resident, fetched once, single buffer"
                if name in k.resident_names
                else (
                    "output block: double-buffered, written to HBM per grid step"
                    if name in k.out_names
                    else "input block: double-buffered, read from HBM per grid step"
                )
            )
            out.append(f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(t.long)}</code></td><td>{t.bytes / MB:.2f}</td><td>{role}</td></tr>")
        for name, t in k.scratch.items():
            out.append(
                f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(t.long)}</code></td><td>{t.bytes / MB:.2f}</td><td>scratch: {'written by the projection, read by the attention / out-proj of the same grid step' if one_kernel else 'persists across the key-block axis'}</td></tr>"
            )
        for name, t in k.stays_in_vmem:
            out.append(
                f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(t.long)}</code></td><td>{t.bytes / MB:.2f}</td><td>temporary of the kernel body: never written to HBM</td></tr>"
            )
        return "".join(out)

    kernel_sections = "".join(
        f"<h3><span class='sw' style='background:{COLOR[kind]}'></span>{kind}: {k.layers} layers, {k.grid_steps} grid steps per call, "
        f"VMEM measured <b>{k.vmem_measured / MB:.1f} MB</b> = weights {k.weights_bytes / MB:.1f} + blocks x2 {k.stream_bytes / MB:.1f} + scratch {k.scratch_bytes / MB:.1f} + temporaries {max(k.vmem_measured - k.resident, 0) / MB:.1f}; "
        f"HBM {k.hbm_read / 1e6:.0f} MB read / {k.hbm_write / 1e6:.0f} MB written per call; {kernel_us[kind] / 1000:.2f} ms per forward ({kernel_us[kind] / total_us:.0%})</h3>"
        f"<table><tr><th>tensor</th><th>dtype[dim=value (meaning), ...] per grid step</th><th>MB</th><th>role</th></tr>{tensor_rows(k)}</table>"
        for kind, k in kernels.items()
    )

    def tensor(dtype: str, *dims: tuple[str, int]) -> Tensor:
        return Tensor(dtype, tuple(dim(n, v) for n, v in dims))

    gone = [
        ("normalized rows, both norms", tensor("bf16", ("R", r), ("D", d)), 2),
        ("qkv projection fp32, then re-cast to bf16", tensor("f32", ("R", r), ("N", 3 * d)), 1.5),
        ("attention scores", tensor("f32", ("B", rows), ("H", heads), ("S", seq), ("S", seq)), 1),
        ("attention probabilities", tensor("bf16", ("B", rows), ("H", heads), ("S", seq), ("S", seq)), 1),
        ("attention output (fp32 then; bf16 now, half the bytes)", tensor("f32", ("R", r), ("D", d)), 0.5),
        ("MLP hidden", tensor("f32", ("R", r), ("F", f)), 1),
        ("gated GELU output", tensor("bf16", ("R", r), ("Fp", fp_)), 1),
    ]
    gone_html = "".join(
        f"<tr><td>{n}</td><td><code>{html.escape(t.shape)}</code></td><td>{t.bytes * m / 1e6:.0f}</td><td>{2 * t.bytes * m / 1e6:.0f}</td></tr>"
        for n, t, m in gone
    )
    wbytes = (d * 3 * d + d * d + d * f + fp_ * d) * 2
    three_kernel = [  # the cross-kernel tensors of the three-kernel path (norm_linear -> flash_attention -> out_mlp)
        ("qkv", tensor("bf16", ("R", r), ("N", 3 * d)), "norm_linear -> flash_attention; now the qkv_sc scratch of the layer kernel"),
        ("att", tensor("bf16", ("R", r), ("D", d)), "flash_attention -> out_mlp; now the att_sc scratch of the layer kernel"),
        (
            "x re-read",
            tensor("f32", ("R", r), ("D", d)),
            "norm_linear and out_mlp both read the residual stream; the layer kernel reads it once (read only: half the write + read column)",
        ),
    ]
    three_html = "".join(
        f"<tr><td>{n}</td><td><code>{html.escape(t.shape)}</code></td><td>{t.bytes / 1e6:.0f}</td><td>{(t.bytes if n == 'x re-read' else 2 * t.bytes) / 1e6:.0f}</td><td>{html.escape(why)}</td></tr>"
        for n, t, why in three_kernel
    )
    if one_kernel:
        still = [
            (
                "x in, x' out",
                tensor("f32", ("R", r), ("D", d)),
                "the residual stream: read once at the start of the layer kernel, written once at the end; the only activation on HBM",
            ),
            ("mask", tensor("i32", ("B", rows), ("S", seq)), "key validity per sequence; the window and the mask bias are built in VMEM"),
            (
                "weights Wqkv^T, Wo^T, Wi^T, Wo2^T",
                Tensor("bf16", (dim("4 matrices", wbytes // 2, "elements of Wqkv^T + Wo^T + Wi^T + Wo2^T"),)),
                "fetched once per kernel call (constant block index); XLA prefetches them into VMEM ahead of the call",
            ),
        ]
    else:
        still = [
            (
                "qkv",
                tensor("bf16", ("R", r), ("N", 3 * d)),
                "norm_linear -> flash_attention. Needed: a query block attends to every key block of its sequence, so all positions must be projected before the first query block runs.",
            ),
            (
                "att",
                tensor("bf16", ("R", r), ("D", d)),
                "flash_attention -> out_mlp. A per-row-block boundary only; fusable into out_mlp's prologue (a row block's attention output depends on that block's queries alone).",
            ),
            (
                "x re-read",
                tensor("f32", ("R", r), ("D", d)),
                "norm_linear and out_mlp both read the residual stream; one fused layer kernel would read it once.",
            ),
            (
                "weights Wqkv^T, Wo^T, Wi^T, Wo2^T",
                Tensor("bf16", (dim("4 matrices", wbytes // 2, "elements of Wqkv^T + Wo^T + Wi^T + Wo2^T"),)),
                "fetched once per kernel call (constant block index)",
            ),
        ]
    still_html = "".join(
        f"<tr><td>{n}</td><td><code>{html.escape(t.shape)}</code></td><td>{t.bytes / 1e6:.0f}</td><td>{html.escape(why)}</td></tr>" for n, t, why in still
    )
    enc_kinds = ("layer_gated_gelu_w64",) if one_kernel else ("norm_linear_rope", "flash_attention_w64", "out_mlp_gated_gelu")
    per_layer_now = sum(kernels[k].hbm_read + kernels[k].hbm_write for k in enc_kinds)
    per_layer_three = r * d * 4 * 3 + r * 3 * d * 2 * 2 + r * d * 2 * 2 + wbytes  # x read twice + x' written, qkv and att written + read once, weights
    per_layer_one = r * d * 4 * 2 + rows * seq * 4 + wbytes
    if one_kernel:
        so_what = (
            f"Peak VMEM is {vmem_peak / MB:.0f} MB of 128 MiB ({peak_kind}: the four resident weight matrices {peak.weights_bytes / MB:.0f} MB, the qkv / att scratch of one grid step {peak.scratch_bytes / MB:.0f} MB, "
            f"the double-buffered x / x' blocks {peak.stream_bytes / MB:.0f} MB, and {max(vmem_peak - peak.resident, 0) / MB:.0f} MB of body temporaries: one head's S x S scores and probabilities, the MLP hidden of one row block). "
            f"VMEM does not grow with the batch: a grid step holds nseq = {nseq} whole sequences (tm = {tm} rows) and a bigger batch only adds grid steps; the sequence length sets the S x S scores and, through tm = nseq * S, nothing else up to S = {fused.LAYER_MAX_SEQ}. "
            f"HBM traffic is {hbm_total / 1e9:.1f} GB per call ({hbm_total / 1e3 / total_us:.0f} GB/s average): per encoder layer only the residual stream in and out ({2 * r * d * 4 / 1e6:.0f} MB) and the weights ({wbytes / 1e6:.0f} MB); every intermediate of the layer, including the qkv projection and the attention output that the three-kernel path wrote to HBM, stays in VMEM."
        )
    else:
        so_what = (
            f"Peak VMEM is {vmem_peak / MB:.0f} MB of 128 MiB ({peak_kind}: resident weights {peak.weights_bytes / MB:.0f} MB, plus double-buffered row blocks and the fp32 MLP hidden of one row block). "
            f"VMEM does not grow with the batch: every kernel works on fixed blocks (tm rows, tq x tk positions) and a bigger batch only adds grid steps; the sequence length changes the flash blocks (up to 512 x 512) and the cos / sin tables. "
            f"HBM traffic is {hbm_total / 1e9:.1f} GB per call ({hbm_total / 1e3 / total_us:.0f} GB/s average); every per-token intermediate of a layer stays in VMEM, what still crosses HBM is the three cross-kernel tensors per layer (qkv, att, the residual stream) and the weights."
        )
    closing = (
        f"An encoder layer now moves about {per_layer_now / 1e6:.0f} MB through HBM (x in, x' out, mask, weights) against about {per_layer_three / 1e6:.0f} MB with three kernels (qkv and att written and read back, the residual read twice). "
        f"The single-pass attention keeps one head's S x S scores ({seq * seq * 4 / MB:.1f} MB fp32) in VMEM; above S = {fused.LAYER_MAX_SEQ} the scores and the sequence-sized row block stop fitting next to the weights and the model falls back to norm_linear -> flash_attention -> out_mlp (run this script with --seq {2 * fused.LAYER_MAX_SEQ} for that page)."
        if one_kernel
        else f"At S = {seq} the sequences are longer than LAYER_MAX_SEQ = {fused.LAYER_MAX_SEQ}, so the layer runs as three kernels; the one-kernel layer (grid over whole sequences, S x S scores of a head in VMEM) is used up to S = {fused.LAYER_MAX_SEQ}, where a layer moves about {per_layer_one / 1e6:.0f} MB instead of {per_layer_three / 1e6:.0f} MB."
    )

    css = """body{font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif;color:#0F172A;max-width:1200px;margin:24px auto;padding:0 16px}
h1{font-size:22px}h2{font-size:17px;margin-top:36px}h3{font-size:14px;margin:26px 0 6px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #E2E8F0;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#F1F5F9}code{font:12px ui-monospace,Menlo,monospace;background:#F1F5F9;padding:1px 4px;border-radius:3px}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.legend{font-size:12px;color:#475569;margin:6px 0 14px}.so{border-left:4px solid #1E40AF;background:#EFF6FF;padding:8px 14px;margin:12px 0}small{color:#475569}"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VMEM / HBM trace: laya {rows}x{seq} on {device}</title><style>{css}</style></head><body>
<h1>VMEM occupancy and HBM traffic, one fused forward ({rows} rows x {seq} tokens, {device})</h1>
<p class="so"><b>So what:</b> the forward runs {len([o for o in ops if o["kind"] != "xla"])} Pallas kernels ({"one per layer" if one_kernel else "three per layer"}) in {total_us / 1000:.1f} ms; the kernels hold {(total_us - xla_us) / total_us:.0%} of the time. {so_what}</p>
<div class="legend">{legend} &nbsp;|&nbsp; VMEM = smallest <code>vmem_limit_bytes</code> that compiles (binary search, 0.25 MB); XLA ops (grey, no VMEM shown) are the glue between kernels (embedding gather, norms, scorer).</div>
<h2>Notation</h2>
<table><tr><th>symbol</th><th>value in this run</th><th>meaning</th></tr>{notation}</table>
<p><small>Shapes are written <code>dtype[dim=value (meaning), ...]</code>: the last dimension is the 128-lane axis of the TPU layout and the one before it the sublane axis. Weights are stored transposed (<code>W^T</code>, [in, out]) so the kernels compute <code>x @ W^T</code> without a transpose.</small></p>
<h2>Architecture: the forward as a flow, VMEM per step</h2>
<p><small>Top: the model-level flow; each step's bar is the VMEM the kernel holds while it runs (resident weights, the pipeline's double-buffered activation blocks, scratch, and the body's live temporaries = measured minus the rest); the orange labels on the arrows are the tensors that cross HBM between steps, with dtype and shape. Below: each kernel expanded into the sub-steps of one grid step with the analytic live set at that point. The analytic live set is an upper bound (Mosaic frees dead values and streams matmul results), so the measured peak, the red tick, sits below the largest sub-step.</small></p>
{svg_flow(model, rows, seq, kernels)}
<h2>Per kernel: every block, scratch buffer and on-chip temporary of one grid step</h2>
<p><small>Blocks are the operands the Pallas pipeline moves between HBM and VMEM per grid step. Blocks with a constant index (weights, norm vectors, biases, cos / sin) are fetched once and, from the measured numbers, kept in a single buffer; blocks whose index changes (rows, q / k / v) are double-buffered so the next step's DMA overlaps this step's compute. Temporaries = measured - blocks - scratch; Mosaic does not expose its allocation of them.</small></p>
{kernel_sections}
<p><small>Around the kernels XLA runs {n_prefetch} async copy / slice ops that prefetch kernel operands into VMEM (memory space S(1)) ahead of each custom call; their bytes are the same weight reads the BlockSpec accounting counts and are not added to the HBM total. The remaining {len(xla_fusion)} XLA ops (embedding gather, embedding / final norms, type_emb add, scorer) take {sum(o["dur_us"] for o in xla_fusion) / 1000:.2f} ms and move {sum(o["hbm_bytes"] for o in xla_fusion) / 1e6:.0f} MB.</small></p>
<h2>Timeline</h2>
{svg_timeline(ops, 0, total_us, "Whole forward")}
{svg_timeline(ops, 0, zoom_hi, "Zoom: encoder layers 0-2 (" + ("one layer kernel per layer" if one_kernel else "norm_linear_rope -> flash_attention -> out_mlp per layer") + ")")}
<h2>Intermediates that no longer touch HBM (per encoder layer, vs the XLA path)</h2>
<table><tr><th>intermediate</th><th>shape in the XLA path</th><th>size (MB)</th><th>HBM traffic saved per layer, write + read (MB)</th></tr>{gone_html}</table>
<h2>Cross-kernel tensors of the three-kernel path{" that the layer kernel keeps in VMEM" if one_kernel else ""} (per encoder layer)</h2>
<table><tr><th>tensor</th><th>shape</th><th>size (MB)</th><th>HBM traffic {"saved" if one_kernel else "per layer"} (MB)</th><th>why</th></tr>{three_html}</table>
<h2>What still crosses HBM (per encoder layer)</h2>
<table><tr><th>tensor</th><th>shape</th><th>size (MB)</th><th>why</th></tr>{still_html}</table>
<p>{closing}</p>
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
            f"{k.kind:22s} VMEM {k.vmem_measured / MB:6.2f} MB = weights {k.weights_bytes / MB:5.2f} + blocks x2 {k.stream_bytes / MB:5.2f} + scratch {k.scratch_bytes / MB:4.2f} "
            f"+ temporaries {(k.vmem_measured - k.resident) / MB:5.2f}; HBM r/w {k.hbm_read / 1e6:.0f}/{k.hbm_write / 1e6:.0f} MB per call"
        )
    ops = trace_forward(d.model, synthetic(d.model, args.rows, args.seq))
    page = render(d.model, args.rows, args.seq, kernels, ops, jax.devices()[0].device_kind)
    Path(args.out).write_text(page)
    print(f"{len(ops)} device ops, wrote {args.out}")


if __name__ == "__main__":
    main()
