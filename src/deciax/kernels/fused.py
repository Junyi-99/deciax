"""Fused encoder-layer kernels for TPU (Pallas / Mosaic).

Rows are the flattened (batch, position) axis. Weights use constant block indices, so the
pipeline fetches them once and keeps them resident in VMEM for the whole grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from deciax.kernels.attention.pallas import F32, HEAD_DIM, VMEM_LIMIT, Ref, flash_attention, head_cols, mask_bias, pv_t, scores_t

LANE = 128
TM = 512  # row block
LAYER_MAX_SEQ = 512  # longest sequence `encoder_layer` runs as a single kernel

type Inputs = list[tuple[jax.Array, pl.BlockSpec]]
type Act = Literal["gated_gelu", "relu"]


@dataclass(frozen=True)
class Rope:
    """cos / sin [S, 128] fp32, covering two 64-wide heads."""

    cos: jax.Array
    sin: jax.Array

    @property
    def seq(self) -> int:
        return int(self.cos.shape[0])


def rope_tables(inv_freq: jax.Array, seq: int) -> Rope:
    freqs = jnp.arange(seq, dtype=F32)[:, None] * inv_freq[None, :]  # [S, 32]
    cos, sin = (jnp.tile(f(freqs), (1, 4)) for f in (jnp.cos, jnp.sin))
    return Rope(cos, sin)


@dataclass(frozen=True)
class LayerParams:
    """Pre-norm transformer layer: matmul weights [in, out], 1-D vectors. The MLP is gated GELU
    (wi_t [D, 2 * Fp] from `pad_gated`) or ReLU (wi_t [D, F])."""

    norm1_w: jax.Array
    norm1_b: jax.Array
    w_qkv_t: jax.Array
    b_qkv: jax.Array
    wo_t: jax.Array
    bo: jax.Array
    norm2_w: jax.Array
    norm2_b: jax.Array
    wi_t: jax.Array
    bi: jax.Array
    wo2_t: jax.Array
    b2: jax.Array


def pad_gated(wi: jax.Array, wo2: jax.Array) -> tuple[jax.Array, jax.Array]:
    """PyTorch gated-MLP weights wi [2 * Fi, D], wo2 [D, Fi] -> wi_t [D, 2 * Fp], wo2_t [Fp, D] with
    Fi zero-padded to a multiple of 128 (gelu(0) * 0 = 0)."""
    fi = wo2.shape[1]
    pad = ((0, -fi % LANE), (0, 0))
    wi_t = jnp.concatenate([jnp.pad(wi[:fi], pad), jnp.pad(wi[fi:], pad)], 0).T
    return wi_t, jnp.pad(wo2.T, pad)


# ---------------------------------------------------------------- in-kernel math


def _layer_norm(x: jax.Array, w: jax.Array, b: jax.Array, eps: float) -> jax.Array:
    z = x - x.mean(-1, keepdims=True)
    z = z * jax.lax.rsqrt((z * z).mean(-1, keepdims=True) + eps)
    return z * w + b


def _gelu(a: jax.Array) -> jax.Array:
    """Exact (erf) GELU, as jax.nn.gelu(approximate=False)."""
    return 0.5 * a * (1.0 + jax.lax.erf(a * (1.0 / math.sqrt(2.0))))


def _dense(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.dot(x, w, preferred_element_type=F32) + b.astype(F32)


def _rope(y: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half RoPE on [tm, tn] holding tn / 64 heads side by side, in y's dtype."""
    tn, half = y.shape[1], HEAD_DIM // 2
    first_half = jax.lax.broadcasted_iota(jnp.int32, y.shape, 1) % HEAD_DIM < half
    rotated = jnp.where(first_half, -pltpu.roll(y, tn - half, 1), pltpu.roll(y, half, 1))  # [-x2, x1] per head
    reps = tn // LANE
    return y * jnp.tile(cos, (1, reps)).astype(y.dtype) + rotated * jnp.tile(sin, (1, reps)).astype(y.dtype)


def _norm_project(
    x: jax.Array,
    nw_ref: Ref,
    nb_ref: Ref,
    w_ref: Ref,
    b_ref: Ref,
    o_ref: Ref,
    *,
    eps: float,
    normalize: bool,
    tn: int,
    rope: tuple[jax.Array, jax.Array] | None,
) -> None:
    """o_ref = LayerNorm(x) @ w + b, column chunk by chunk; RoPE on the q and k chunks (j < 2)."""
    xn = (_layer_norm(x, nw_ref[...], nb_ref[...], eps) if normalize else x).astype(w_ref.dtype)
    for j in range(w_ref.shape[1] // tn):
        cols = slice(j * tn, (j + 1) * tn)
        y = _dense(xn, w_ref[:, cols], b_ref[:, cols]).astype(o_ref.dtype)
        if rope is not None and j < 2:
            y = _rope(y, *rope)
        o_ref[:, cols] = y


def _mlp(y: jax.Array, nw_ref: Ref, nb_ref: Ref, wi_ref: Ref, bi_ref: Ref, wo2_ref: Ref, b2_ref: Ref, *, eps: float, act: Act) -> jax.Array:
    """y + act(LayerNorm(y) @ wi + bi) @ wo2 + b2, fp32."""
    h = _dense(_layer_norm(y, nw_ref[...], nb_ref[...], eps).astype(wi_ref.dtype), wi_ref[...], bi_ref[...])
    fp = wo2_ref.shape[0]
    g = _gelu(h[:, :fp]) * h[:, fp:] if act == "gated_gelu" else jnp.maximum(h, 0.0)
    return _dense(g.astype(wo2_ref.dtype), wo2_ref[...], b2_ref[...]) + y


def _check_mlp(act: Act, wi_t: jax.Array, wo2_t: jax.Array) -> None:
    assert wi_t.shape[1] == (2 if act == "gated_gelu" else 1) * wo2_t.shape[0], (act, wi_t.shape, wo2_t.shape)


# ---------------------------------------------------------------- pallas_call plumbing


def _vec(v: jax.Array) -> jax.Array:
    return v.reshape(1, -1)


def _rows(tm: int, width: int) -> pl.BlockSpec:
    return pl.BlockSpec((tm, width), lambda i: (i, 0))


def _resident(*arrays: jax.Array) -> Inputs:
    return [(a, pl.BlockSpec(a.shape, lambda i: (0, 0))) for a in arrays]


def _row_call(
    kernel: Callable[..., None],
    inputs: Inputs,
    out: tuple[int, int],
    out_dtype: Any,
    tm: int,
    name: str,
    scratch: Sequence[Any] = (),
) -> jax.Array:
    """pallas_call over row blocks of `tm`; `inputs` pairs each array with its BlockSpec."""
    arrays, specs = zip(*inputs, strict=True)
    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct(out, out_dtype),  # type: ignore[no-untyped-call]
        grid_spec=pl.GridSpec(grid=(out[0] // tm,), in_specs=list(specs), out_specs=_rows(tm, out[1]), scratch_shapes=scratch),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
        name=name,
    )(*arrays)


# ---------------------------------------------------------------- kernels


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
    tn: int | None = None,
) -> jax.Array:
    """[R, D] fp32 -> LayerNorm (if `normalize`) -> @ w_t [D, N] + bias -> [R, N] bf16.

    The kernel walks w_t in column chunks of `tn` (default D). With `rope`, the first two chunks
    (q and k) get RoPE at position row % rope.seq, in bf16 like the jnp path.
    """
    rows, d = x.shape
    n = w_t.shape[1]
    tm = min(TM, rows)
    tn = tn or (d if n % d == 0 else n)
    assert rows % tm == 0 and n % tn == 0 and tn % LANE == 0, (rows, n, tm, tn)

    inputs = [(x, _rows(tm, d)), *_resident(_vec(norm_w), _vec(norm_b), w_t, _vec(bias))]
    if rope is not None:
        seq = rope.seq
        assert tm % seq == 0 or seq % tm == 0, (tm, seq)
        if tm >= seq:  # whole sequences per block: one constant table
            inputs += _resident(jnp.tile(rope.cos, (tm // seq, 1)), jnp.tile(rope.sin, (tm // seq, 1)))
        else:
            spec = pl.BlockSpec((tm, LANE), lambda i: (i % (seq // tm), 0))
            inputs += [(rope.cos, spec), (rope.sin, spec)]

    def kernel(*refs: Ref) -> None:
        x_ref, nw_ref, nb_ref, w_ref, b_ref, *rope_refs, o_ref = refs
        tables = (rope_refs[0][...], rope_refs[1][...]) if rope_refs else None
        _norm_project(x_ref[...].astype(F32), nw_ref, nb_ref, w_ref, b_ref, o_ref, eps=eps, normalize=normalize, tn=tn, rope=tables)

    return _row_call(kernel, inputs, (rows, n), jnp.bfloat16, tm, "norm_linear_rope" if rope is not None else "norm_linear")


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
    act: Act,
) -> jax.Array:
    """y = att @ wo_t + bo + x, then y + act(LayerNorm(y) @ wi_t + bi) @ wo2_t + b2: [R, D] fp32.

    act="gated_gelu": wi_t [D, 2 * Fp] from `pad_gated`; act="relu": wi_t [D, F].
    """
    rows, d = x.shape
    tm = min(TM, rows)
    _check_mlp(act, wi_t, wo2_t)
    assert rows % tm == 0 and wo2_t.shape[0] % LANE == 0, (rows, tm, wo2_t.shape)

    inputs = [(att, _rows(tm, d)), (x, _rows(tm, d)), *_resident(wo_t, _vec(bo), _vec(norm_w), _vec(norm_b), wi_t, _vec(bi), wo2_t, _vec(b2))]

    def kernel(*refs: Ref) -> None:
        att_ref, x_ref, wo_ref, bo_ref, *mlp_refs, o_ref = refs
        y = _dense(att_ref[...], wo_ref[...], bo_ref[...]) + x_ref[...]
        o_ref[...] = _mlp(y, *mlp_refs, eps=eps, act=act)

    return _row_call(kernel, inputs, (rows, d), F32, tm, f"out_mlp_{act}")


def layer(
    x: jax.Array,
    mask: jax.Array,
    w: LayerParams,
    *,
    seq: int,
    heads: int,
    eps: float,
    act: Act,
    window: int = 0,
    rope: Rope | None = None,
    normalize: bool = True,
) -> jax.Array:
    """One pre-norm layer in a single kernel, seq <= LAYER_MAX_SEQ: x [R, D] fp32 -> [R, D] fp32.

    A grid step holds whole sequences, so attention is a single-pass softmax per sequence and head,
    and HBM sees x once in, x' once out and each weight once.
    """
    rows, d = x.shape
    batch = rows // seq
    n = w.w_qkv_t.shape[1]
    assert n == 3 * d and heads * HEAD_DIM == d and seq <= LAYER_MAX_SEQ, (n, d, heads, seq)
    _check_mlp(act, w.wi_t, w.wo2_t)
    nseq = min(batch, max(1, TM // seq))  # sequences per grid step
    assert batch % nseq == 0, (batch, nseq)
    tm = nseq * seq
    cos, sin = (rope.cos, rope.sin) if rope is not None else (jnp.zeros((seq, LANE), F32),) * 2

    inputs = [
        (x, _rows(tm, d)),
        (mask.reshape(batch, 1, seq).astype(jnp.int32), pl.BlockSpec((nseq, 1, seq), lambda i: (i, 0, 0))),
        *_resident(
            _vec(w.norm1_w),
            _vec(w.norm1_b),
            w.w_qkv_t,
            _vec(w.b_qkv),
            cos,
            sin,
            w.wo_t,
            _vec(w.bo),
            _vec(w.norm2_w),
            _vec(w.norm2_b),
            w.wi_t,
            _vec(w.bi),
            w.wo2_t,
            _vec(w.b2),
        ),
    ]

    def kernel(*refs: Ref) -> None:
        x_ref, m_ref, n1w_ref, n1b_ref, wqkv_ref, bqkv_ref, cos_ref, sin_ref, wo_ref, bo_ref, *mlp_refs, o_ref, qkv_sc, att_sc = refs
        xv = x_ref[...]
        tables = (jnp.tile(cos_ref[...], (nseq, 1)), jnp.tile(sin_ref[...], (nseq, 1))) if rope is not None else None
        _norm_project(xv, n1w_ref, n1b_ref, wqkv_ref, bqkv_ref, qkv_sc, eps=eps, normalize=normalize, tn=d, rope=tables)

        for s in range(nseq):  # scores transposed, [keys, queries]
            r = slice(s * seq, (s + 1) * seq)
            bias = mask_bias(m_ref[s], (seq, seq), window)
            for h in range(heads):
                q, k, v = (qkv_sc[r, head_cols(h, j * d)] for j in range(3))
                st = scores_t(q, k) + bias
                p = jnp.exp(st - st.max(0, keepdims=True))
                att_sc[r, head_cols(h)] = jnp.transpose(pv_t(v, p) / p.sum(0, keepdims=True)).astype(att_sc.dtype)

        y = _dense(att_sc[...], wo_ref[...], bo_ref[...]) + xv
        o_ref[...] = _mlp(y, *mlp_refs, eps=eps, act=act)

    scratch = [pltpu.VMEM((tm, n), w.w_qkv_t.dtype), pltpu.VMEM((tm, d), w.w_qkv_t.dtype)]  # qkv, attention output
    return _row_call(kernel, inputs, (rows, d), F32, tm, f"layer_{act}_w{window}", scratch)


def encoder_layer(
    x: jax.Array,
    mask: jax.Array,
    w: LayerParams,
    *,
    seq: int,
    heads: int,
    eps: float,
    act: Act,
    window: int = 0,
    rope: Rope | None = None,
    normalize: bool = True,
) -> jax.Array:
    """Pre-norm layer on [B * S, D] fp32 rows: `layer` up to LAYER_MAX_SEQ, else
    norm_linear -> flash_attention -> out_mlp."""
    if seq <= LAYER_MAX_SEQ:
        return layer(x, mask, w, seq=seq, heads=heads, eps=eps, act=act, window=window, rope=rope, normalize=normalize)
    d = x.shape[1]
    qkv = norm_linear(x, w.norm1_w, w.norm1_b, eps, w.w_qkv_t, w.b_qkv, normalize=normalize, rope=rope)
    att = flash_attention(qkv.reshape(-1, seq, 3 * d), mask, heads=heads, window=window)
    return out_mlp(att.reshape(-1, d), x, w.wo_t, w.bo, w.norm2_w, w.norm2_b, eps, w.wi_t, w.bi, w.wo2_t, w.b2, act=act)
