"""Fused encoder-layer kernels for TPU (Pallas / Mosaic).

The XLA path streams every intermediate of a transformer layer through HBM (~3 GB per layer at
64 x 256 tokens on laya, xprof: every op HBM-bound). These two kernels keep the intermediates in
VMEM and touch HBM once per layer for the residual stream and once for the weights:

- `norm_linear`: fp32 residual -> LayerNorm -> bf16 matmul (+ bias) [-> RoPE on the q / k blocks] -> bf16.
- `out_mlp`: att @ Wo (+ bias) + residual -> LayerNorm -> Wi (+ bias) -> gated GELU | ReLU -> Wo2 (+ bias)
  + residual -> fp32.

Rows are the flattened (batch, position) axis; weights have constant block indices, so the
pipeline fetches them once and keeps them resident in VMEM for the whole grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HEAD_DIM = 64  # rotate-half RoPE below assumes 64-wide heads (two heads per 128-lane vreg)
LANE = 128
VMEM_LIMIT = 96 * 1024 * 1024
TM = 512  # row block
LAYER_MAX_SEQ = 512  # `layer` (one kernel per layer, whole sequences in VMEM) is used up to this sequence length
MASK_FILL = -1e30
F32 = jnp.float32


@dataclass(frozen=True)
class Rope:
    """cos / sin tables for `norm_linear`: [L, 128] fp32, one head's cos duplicated across the
    two 64-lane halves; `seq` = positions per sequence (rows = batch * seq)."""

    cos: jax.Array
    sin: jax.Array
    seq: int


def rope_tables(inv_freq: jax.Array, seq: int) -> Rope:
    freqs = jnp.arange(seq, dtype=F32)[:, None] * inv_freq[None, :]  # [S, 32]
    cos = jnp.tile(jnp.cos(freqs), (1, 4))  # [S, 128]: (first half, second half) x two heads
    sin = jnp.tile(jnp.sin(freqs), (1, 4))
    return Rope(cos, sin, seq)


def _row_block(rows: int) -> int:
    return min(TM, rows)


def _layer_norm(x: jax.Array, w: jax.Array, b: jax.Array, eps: float) -> jax.Array:
    z = x - x.mean(-1, keepdims=True)
    z = z * jax.lax.rsqrt((z * z).mean(-1, keepdims=True) + eps)
    return z * w + b


def _rope(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half RoPE on a [tm, tn] block holding tn / 64 heads side by side, in y's dtype."""
    tn = y.shape[1]
    first_half = jax.lax.broadcasted_iota(jnp.int32, y.shape, 1) % HEAD_DIM < HEAD_DIM // 2
    # rotated[l] = -x[l + 32] in the first half of every head, x[l - 32] in the second half
    rotated = jnp.where(first_half, -pltpu.roll(y, tn - 32, 1), pltpu.roll(y, 32, 1))
    reps = tn // LANE
    return y * jnp.tile(cos, (1, reps)).astype(y.dtype) + rotated * jnp.tile(sin, (1, reps)).astype(y.dtype)


def norm_linear(
    x: jax.Array,
    norm_w: jax.Array,
    norm_b: jax.Array,
    eps: float,
    w_t: jax.Array,
    bias: jax.Array,
    *,
    normalize: bool = True,
    rope: Rope | None = None,
    rope_blocks: int = 0,
    out_dtype: jnp.dtype = jnp.bfloat16,
    tn: int | None = None,
) -> jax.Array:
    """[R, D] fp32 -> LayerNorm (if `normalize`) -> cast -> @ w_t [D, N] + bias [N] -> [R, N].

    The grid runs over row blocks only; w_t stays resident in VMEM and the kernel walks its
    column chunks of `tn` (default D). With `rope`, chunks j < rope_blocks (the q and k heads)
    get rotate-half RoPE at position = row % rope.seq, applied in the output dtype as the jnp
    path does; `tn` must then be a multiple of 64.
    """
    rows, d = x.shape
    n = w_t.shape[1]
    tm = _row_block(rows)
    tn = tn or (d if n % d == 0 else n)
    assert rows % tm == 0 and n % tn == 0 and tn % LANE == 0, (rows, n, tm, tn)
    chunks = n // tn

    in_specs = [
        pl.BlockSpec((tm, d), lambda i: (i, 0)),
        pl.BlockSpec((1, d), lambda i: (0, 0)),
        pl.BlockSpec((1, d), lambda i: (0, 0)),
        pl.BlockSpec((d, n), lambda i: (0, 0)),
        pl.BlockSpec((1, n), lambda i: (0, 0)),
    ]
    args = [x, norm_w.reshape(1, d), norm_b.reshape(1, d), w_t, bias.reshape(1, n)]
    if rope is not None:
        seq = rope.seq
        assert tm % seq == 0 or seq % tm == 0, (tm, seq)
        if tm >= seq:  # the block holds whole sequences: one constant [tm, 128] table
            cos, sin = (jnp.tile(t, (tm // seq, 1)) for t in (rope.cos, rope.sin))
            pos_map = lambda i: (0, 0)
        else:
            cos, sin = rope.cos, rope.sin
            pos_map = lambda i: (i % (seq // tm), 0)
        in_specs += [pl.BlockSpec((tm, LANE), pos_map), pl.BlockSpec((tm, LANE), pos_map)]
        args += [cos, sin]

    def kernel(x_ref, nw_ref, nb_ref, w_ref, b_ref, *rest):
        if rope is not None:
            cos_ref, sin_ref, o_ref = rest
        else:
            o_ref = rest[0]
        xv = x_ref[...].astype(F32)
        xn = (_layer_norm(xv, nw_ref[...], nb_ref[...], eps) if normalize else xv).astype(w_ref.dtype)
        for j in range(chunks):
            cols = slice(j * tn, (j + 1) * tn)
            y = (jnp.dot(xn, w_ref[:, cols], preferred_element_type=F32) + b_ref[:, cols].astype(F32)).astype(o_ref.dtype)
            if rope is not None and j < rope_blocks:
                y = _rope(y, cos_ref[...], sin_ref[...])
            o_ref[:, cols] = y

    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, n), out_dtype),
        grid_spec=pl.GridSpec(grid=(rows // tm,), in_specs=in_specs, out_specs=pl.BlockSpec((tm, n), lambda i: (i, 0))),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
        name="norm_linear_rope" if rope is not None else "norm_linear",
    )(*args)


def _gelu(a: jax.Array) -> jax.Array:  # exact (erf) GELU, as nn.GELU / jax.nn.gelu(approximate=False)
    return 0.5 * a * (1.0 + jax.lax.erf(a * (1.0 / math.sqrt(2.0))))


def out_mlp(
    att: jax.Array,
    x: jax.Array,
    wo_t: jax.Array,
    bo: jax.Array,
    norm_w: jax.Array,
    norm_b: jax.Array,
    eps: float,
    wi_t: jax.Array,
    bi: jax.Array,
    wo2_t: jax.Array,
    b2: jax.Array,
    *,
    act: str,
    tm: int | None = None,
) -> jax.Array:
    """y = att @ wo_t + bo + x;  out = y + act(LN(y) @ wi_t + bi) @ wo2_t + b2   ([R, D] fp32).

    act = "gated_gelu": wi_t is [D, 2 * Fp], gelu(h[:, :Fp]) * h[:, Fp:] (Fp a multiple of 128; pad
    with zero columns / rows, see `pad_gated`); act = "relu": wi_t is [D, F].
    """
    rows, d = x.shape
    f = wi_t.shape[1]
    fp = wo2_t.shape[0]
    assert act in ("gated_gelu", "relu") and (f == 2 * fp if act == "gated_gelu" else f == fp), (act, f, fp)
    tm = tm or _row_block(rows)
    assert rows % tm == 0 and fp % LANE == 0, (rows, tm, fp)

    def kernel(att_ref, x_ref, wo_ref, bo_ref, nw_ref, nb_ref, wi_ref, bi_ref, wo2_ref, b2_ref, o_ref):
        y = jnp.dot(att_ref[...], wo_ref[...], preferred_element_type=F32) + bo_ref[...].astype(F32) + x_ref[...]
        z = _layer_norm(y, nw_ref[...], nb_ref[...], eps).astype(wi_ref.dtype)
        h = jnp.dot(z, wi_ref[...], preferred_element_type=F32) + bi_ref[...].astype(F32)
        g = _gelu(h[:, :fp]) * h[:, fp:] if act == "gated_gelu" else jnp.maximum(h, 0.0)
        o_ref[...] = jnp.dot(g.astype(wo2_ref.dtype), wo2_ref[...], preferred_element_type=F32) + b2_ref[...].astype(F32) + y

    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, d), F32),
        grid_spec=pl.GridSpec(
            grid=(rows // tm,),
            in_specs=[
                pl.BlockSpec((tm, d), lambda i: (i, 0)),
                pl.BlockSpec((tm, d), lambda i: (i, 0)),
                full((d, d)),
                full((1, d)),
                full((1, d)),
                full((1, d)),
                full((d, f)),
                full((1, f)),
                full((fp, d)),
                full((1, d)),
            ],
            out_specs=pl.BlockSpec((tm, d), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
        name=f"out_mlp_{act}",
    )(att, x, wo_t, bo.reshape(1, d), norm_w.reshape(1, d), norm_b.reshape(1, d), wi_t, bi.reshape(1, f), wo2_t, b2.reshape(1, d))


def pad_gated(wi: jax.Array, wo2: jax.Array) -> tuple[jax.Array, jax.Array]:
    """PyTorch-layout gated MLP weights wi [2 * Fi, D], wo2 [D, Fi] -> wi_t [D, 2 * Fp], wo2_t [Fp, D]
    with Fi padded to a multiple of 128 (zero columns: gelu(0) * 0 = 0, so the padding adds nothing)."""
    fi = wo2.shape[1]
    fp = -(-fi // LANE) * LANE
    a, b = wi[:fi], wi[fi:]
    pad = ((0, fp - fi), (0, 0))
    wi_t = jnp.concatenate([jnp.pad(a, pad), jnp.pad(b, pad)], 0).T
    return wi_t, jnp.pad(wo2, ((0, 0), (0, fp - fi))).T


def layer(
    x: jax.Array,
    mask: jax.Array,
    norm1_w: jax.Array,
    norm1_b: jax.Array,
    w_qkv_t: jax.Array,
    b_qkv: jax.Array,
    rope: Rope | None,
    wo_t: jax.Array,
    bo: jax.Array,
    norm2_w: jax.Array,
    norm2_b: jax.Array,
    wi_t: jax.Array,
    bi: jax.Array,
    wo2_t: jax.Array,
    b2: jax.Array,
    *,
    seq: int,
    heads: int,
    window: int,
    eps: float,
    act: str,
    normalize: bool = True,
) -> jax.Array:
    """One transformer layer in one kernel, for seq <= LAYER_MAX_SEQ: x [R, D] fp32 -> x' [R, D] fp32.

    A grid step holds tm / seq whole sequences, so the layer's intermediates never leave VMEM:
    LayerNorm -> @ w_qkv_t (+ b_qkv) -> RoPE on the q / k heads -> per sequence and head a single-pass
    softmax attention with the key mask and sliding window built in-kernel (scores [S, S] fit, no online
    rescaling) -> @ wo_t + bo + x -> LayerNorm -> @ wi_t + bi -> gated GELU | ReLU -> @ wo2_t + b2 + y.
    HBM sees x once in, x' once out and the weights once per kernel.
    """
    rows, d = x.shape
    batch = rows // seq
    n, f, fp = w_qkv_t.shape[1], wi_t.shape[1], wo2_t.shape[0]
    assert n == 3 * d and heads * HEAD_DIM == d and seq <= LAYER_MAX_SEQ, (n, d, heads, seq)
    assert act in ("gated_gelu", "relu") and (f == 2 * fp if act == "gated_gelu" else f == fp), (act, f, fp)
    nseq = min(batch, max(1, TM // seq))
    assert batch % nseq == 0, (batch, nseq)
    tm = nseq * seq
    scale = HEAD_DIM**-0.5
    cos, sin = (rope.cos, rope.sin) if rope is not None else (jnp.zeros((seq, LANE), F32),) * 2

    def kernel(
        x_ref,
        m_ref,
        n1w_ref,
        n1b_ref,
        wqkv_ref,
        bqkv_ref,
        cos_ref,
        sin_ref,
        wo_ref,
        bo_ref,
        n2w_ref,
        n2b_ref,
        wi_ref,
        bi_ref,
        wo2_ref,
        b2_ref,
        o_ref,
        qkv_sc,
        att_sc,
    ):
        xv = x_ref[...]
        xn = (_layer_norm(xv, n1w_ref[...], n1b_ref[...], eps) if normalize else xv).astype(wqkv_ref.dtype)
        for j in range(3):  # q | k | v chunks of the projection, RoPE on q and k
            cols = slice(j * d, (j + 1) * d)
            y = (jnp.dot(xn, wqkv_ref[:, cols], preferred_element_type=F32) + bqkv_ref[:, cols].astype(F32)).astype(qkv_sc.dtype)
            if rope is not None and j < 2:
                y = _rope(y, jnp.tile(cos_ref[...], (nseq, 1)), jnp.tile(sin_ref[...], (nseq, 1)))
            qkv_sc[:, cols] = y

        for s_ in range(nseq):  # attention, one sequence at a time: scores [keys, queries] per head
            rows_ = slice(s_ * seq, (s_ + 1) * seq)
            allowed = jnp.transpose(m_ref[s_] > 0)  # [S, 1]: key validity along the rows
            if window:
                kpos = jax.lax.broadcasted_iota(jnp.int32, (seq, seq), 0)
                qpos = jax.lax.broadcasted_iota(jnp.int32, (seq, seq), 1)
                allowed = allowed & (jnp.abs(qpos - kpos) <= window)
            bias = jnp.broadcast_to(jnp.where(allowed, 0.0, MASK_FILL).astype(F32), (seq, seq))
            for h in range(heads):
                cols = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                q = qkv_sc[rows_, cols]
                k = qkv_sc[rows_, d + h * HEAD_DIM : d + (h + 1) * HEAD_DIM]
                v = qkv_sc[rows_, 2 * d + h * HEAD_DIM : 2 * d + (h + 1) * HEAD_DIM]
                st = jax.lax.dot_general(k, q * scale, (((1,), (1,)), ((), ())), preferred_element_type=F32) + bias
                p = jnp.exp(st - st.max(0, keepdims=True))
                l = p.sum(0, keepdims=True)
                ot = jax.lax.dot_general(v, p.astype(v.dtype), (((0,), (0,)), ((), ())), preferred_element_type=F32)  # [64, S]
                att_sc[rows_, cols] = jnp.transpose(ot / l).astype(att_sc.dtype)

        y = jnp.dot(att_sc[...], wo_ref[...], preferred_element_type=F32) + bo_ref[...].astype(F32) + xv
        z = _layer_norm(y, n2w_ref[...], n2b_ref[...], eps).astype(wi_ref.dtype)
        h = jnp.dot(z, wi_ref[...], preferred_element_type=F32) + bi_ref[...].astype(F32)
        g = _gelu(h[:, :fp]) * h[:, fp:] if act == "gated_gelu" else jnp.maximum(h, 0.0)
        o_ref[...] = jnp.dot(g.astype(wo2_ref.dtype), wo2_ref[...], preferred_element_type=F32) + b2_ref[...].astype(F32) + y

    full = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, d), F32),
        grid_spec=pl.GridSpec(
            grid=(rows // tm,),
            in_specs=[
                pl.BlockSpec((tm, d), lambda i: (i, 0)),
                pl.BlockSpec((nseq, 1, seq), lambda i: (i, 0, 0)),
                full((1, d)),
                full((1, d)),
                full((d, n)),
                full((1, n)),
                full((seq, LANE)),
                full((seq, LANE)),
                full((d, d)),
                full((1, d)),
                full((1, d)),
                full((1, d)),
                full((d, f)),
                full((1, f)),
                full((fp, d)),
                full((1, d)),
            ],
            out_specs=pl.BlockSpec((tm, d), lambda i: (i, 0)),
            scratch_shapes=[pltpu.VMEM((tm, n), w_qkv_t.dtype), pltpu.VMEM((tm, d), w_qkv_t.dtype)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
        name=f"layer_{act}_w{window}",
    )(
        x,
        mask.reshape(batch, 1, seq).astype(jnp.int32),
        norm1_w.reshape(1, d),
        norm1_b.reshape(1, d),
        w_qkv_t,
        b_qkv.reshape(1, n),
        cos,
        sin,
        wo_t,
        bo.reshape(1, d),
        norm2_w.reshape(1, d),
        norm2_b.reshape(1, d),
        wi_t,
        bi.reshape(1, f),
        wo2_t,
        b2.reshape(1, d),
    )
