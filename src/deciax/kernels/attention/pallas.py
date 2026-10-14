"""Flash attention for TPU (Pallas / Mosaic) on the packed q | k | v projection.

The key mask and sliding window are applied in-kernel, so no [B, H, S, S] bias or score tensor
reaches HBM. The helpers below are shared with the single-pass attention in `fused.layer`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from deciax.kernels.attention.reference import MASK_FILL

HEAD_DIM = 64
BLOCK = 256
F32 = jnp.float32
VMEM_LIMIT = 96 * 1024 * 1024

type Ref = Any  # a Pallas kernel ref


def head_cols(h: int, base: int = 0) -> slice:
    """Lanes of head h in a packed [..., H * 64] row starting at `base`."""
    return slice(base + h * HEAD_DIM, base + (h + 1) * HEAD_DIM)


def mask_bias(key_mask: jax.Array, shape: tuple[int, int], window: int, k0: jax.Array | int = 0, q0: jax.Array | int = 0) -> jax.Array:
    """[1, tk] key mask (1 = real token) -> [tk, tq] additive bias for transposed scores whose
    keys / queries start at positions k0 / q0."""
    allowed = jnp.transpose(key_mask > 0)
    if window:
        kpos = k0 + jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        qpos = q0 + jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        allowed = allowed & (jnp.abs(qpos - kpos) <= window)
    return jnp.broadcast_to(jnp.where(allowed, 0.0, MASK_FILL).astype(F32), shape)


def scores_t(q: jax.Array, k: jax.Array, scale: float) -> jax.Array:
    """k @ (q * scale)^T: [tk, tq] fp32."""
    return jax.lax.dot_general(k, q * scale, (((1,), (1,)), ((), ())), preferred_element_type=F32)


def pv_t(v: jax.Array, p: jax.Array) -> jax.Array:
    """v^T @ p for p [tk, tq]: [64, tq] fp32."""
    return jax.lax.dot_general(v, p.astype(v.dtype), (((0,), (0,)), ((), ())), preferred_element_type=F32)


def flash_attention(
    qkv: jax.Array,
    mask: jax.Array,
    *,
    heads: int,
    window: int = 0,
    tq: int | None = None,
    tk: int | None = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Packed [B, S, 3 * H * 64] q | k | v, mask [B, S] (1 = real token) -> [B, S, H * 64].

    Grid (batch, query block, key step); each step runs all heads and keeps the online-softmax
    state in VMEM scratch. Sliding windows visit only the key blocks that intersect the window.
    Scores are transposed (k @ q^T, [tk, tq]): PV then streams 64 rows per head through the MXU
    instead of tq, and the softmax reductions run along sublanes. Default blocks, tuned on v6e:
    512 x 512 full, 256 x 128 sliding.
    """
    batch, seq, three_d = qkv.shape
    d = heads * HEAD_DIM
    assert three_d == 3 * d, (qkv.shape, heads)
    tq = min(tq or (BLOCK if window else 2 * BLOCK), seq)
    tk = min(tk or (BLOCK // 2 if window else 2 * BLOCK), seq)
    assert seq % tq == 0 and seq % tk == 0, (seq, tq, tk)
    nq, nk = seq // tq, seq // tk
    steps = min(nk, (tq - 1 + 2 * window) // tk + 2) if window else nk
    scale = HEAD_DIM**-0.5

    def first_block(i: jax.Array) -> jax.Array:
        """First key block within the window of query block i."""
        return jnp.maximum((i * tq - window) // tk, 0) if window else i * 0

    def kv_block(i: jax.Array, kk: jax.Array) -> jax.Array:
        # clamped: steps past the last block repeat its index, so no DMA (and the kernel skips them)
        return jnp.minimum(first_block(i) + kk, nk - 1)

    def kernel(q_ref: Ref, k_ref: Ref, v_ref: Ref, m_ref: Ref, o_ref: Ref, m_sc: Ref, l_sc: Ref, acc_sc: Ref) -> None:
        i, kk = pl.program_id(1), pl.program_id(2)
        kb = first_block(i) + kk

        @pl.when(kk == 0)
        def _() -> None:
            m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, F32)
            l_sc[...] = jnp.zeros(l_sc.shape, F32)
            acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

        in_range = kb < nk
        if window:
            in_range &= kb <= ((i + 1) * tq - 1 + window) // tk

        @pl.when(in_range)
        def _() -> None:
            bias = mask_bias(m_ref[...], (tk, tq), window, k0=kb * tk, q0=i * tq)
            for h in range(heads):
                cols = head_cols(h)
                st = scores_t(q_ref[:, cols], k_ref[:, cols], scale) + bias
                m_prev = m_sc[h]  # [1, tq]
                m_new = jnp.maximum(m_prev, st.max(0, keepdims=True))
                alpha = jnp.exp(m_prev - m_new)
                p = jnp.exp(st - m_new)
                l_sc[h] = alpha * l_sc[h] + p.sum(0, keepdims=True)
                acc_sc[h] = alpha * acc_sc[h] + pv_t(v_ref[:, cols], p)
                m_sc[h] = m_new

        @pl.when(kk == steps - 1)
        def _() -> None:
            for h in range(heads):
                o_ref[:, head_cols(h)] = jnp.transpose(acc_sc[h] / l_sc[h]).astype(o_ref.dtype)

    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, seq, d), out_dtype),  # type: ignore[no-untyped-call]
        grid_spec=pl.GridSpec(
            grid=(batch, nq, steps),
            in_specs=[
                pl.BlockSpec((None, tq, d), lambda b, i, kk: (b, i, 0)),
                pl.BlockSpec((None, tk, d), lambda b, i, kk: (b, kv_block(i, kk), 1)),
                pl.BlockSpec((None, tk, d), lambda b, i, kk: (b, kv_block(i, kk), 2)),
                pl.BlockSpec((None, 1, tk), lambda b, i, kk: (b, 0, kv_block(i, kk))),
            ],
            out_specs=pl.BlockSpec((None, tq, d), lambda b, i, kk: (b, i, 0)),
            # running max, running sum, accumulator
            scratch_shapes=[pltpu.VMEM((heads, 1, tq), F32), pltpu.VMEM((heads, 1, tq), F32), pltpu.VMEM((heads, HEAD_DIM, tq), F32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"), vmem_limit_bytes=VMEM_LIMIT),
        name=f"flash_attention_w{window}",
    )(qkv, qkv, qkv, mask.reshape(batch, 1, seq).astype(jnp.int32))


def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, window: int = 0) -> jax.Array:
    """[B, S, H, Dh] q, k, v -> [B, S, H, Dh] fp32 (the `ops.attention` interface)."""
    batch, seq, heads, dh = q.shape
    assert dh == HEAD_DIM, dh
    qkv = jnp.concatenate([t.reshape(batch, seq, heads * dh) for t in (q, k, v)], -1)
    out = flash_attention(qkv, mask, heads=heads, window=window, out_dtype=F32)
    return out.reshape(batch, seq, heads, dh)
