"""Flash attention for TPU (Pallas / Mosaic) with the key mask and the sliding window applied
in-kernel: no [B, H, S, S] bias or score tensor ever reaches HBM.

Input is the packed projection [B, S, 3 * H * 64] (q | k | v, 64-wide heads) straight out of
`fused.norm_linear`; a grid step handles every head of one query block and loops over key
blocks with the online softmax in VMEM scratch (transposed: scores are [keys, queries]). Sliding-window layers only
visit the key blocks that intersect the window (the others are clamped to a repeated block
index, so no DMA, and skipped).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HEAD_DIM = 64
LANE = 128
BLOCK = 256
MASK_FILL = -1e30
F32 = jnp.float32
VMEM_LIMIT = 96 * 1024 * 1024


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
    """[B, S, 3 * H * 64] packed q | k | v, mask [B, S] (1 = real token) -> [B, S, H * 64].

    One grid step = one query block of one sequence, all heads (static loop in-kernel), one key
    block; the online-softmax state lives in VMEM scratch across the key-block axis.

    Transposed formulation: s^T = k @ q^T [tk, tq] and o^T = v^T @ p^T [64, tq]. The MXU streams
    LHS rows, so PV costs 64 cycles per head instead of the 256 of p @ v (N = 64 fills a quarter of
    the array), and the softmax max / sum run along sublanes (plain VPU adds) instead of across
    lanes. Block defaults from benchmarks/attn_micro.py on v6e: 512 x 512 for full attention,
    256 x 128 for sliding windows (fewer wasted keys around the window).
    """
    batch, seq, three_d = qkv.shape
    d = heads * HEAD_DIM
    assert three_d == 3 * d, (qkv.shape, heads)
    tq = tq or (BLOCK if window else 2 * BLOCK)
    tk = tk or (BLOCK // 2 if window else 2 * BLOCK)
    tq, tk = min(tq, seq), min(tk, seq)
    assert seq % tq == 0 and seq % tk == 0, (seq, tq, tk)
    nq, nk = seq // tq, seq // tk
    scale = HEAD_DIM**-0.5

    if window:
        steps = min(nk, (tq - 1 + 2 * window) // tk + 2)
        lo = lambda i: jnp.maximum((i * tq - window) // tk, 0)  # first key block that can intersect query block i
    else:
        steps = nk
        lo = lambda i: i * 0

    def kv_map(i, kk):
        return jnp.minimum(lo(i) + kk, nk - 1)

    def kernel(q_ref, k_ref, v_ref, m_ref, o_ref, m_sc, l_sc, acc_sc):
        i, kk = pl.program_id(1), pl.program_id(2)

        @pl.when(kk == 0)
        def _():
            m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, F32)
            l_sc[...] = jnp.zeros(l_sc.shape, F32)
            acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

        kb = lo(i) + kk
        if window:
            hi = ((i + 1) * tq - 1 + window) // tk
            step_ok = (kb <= hi) & (kb < nk)
        else:
            step_ok = kb < nk

        @pl.when(step_ok)
        def _():
            allowed = jnp.transpose(m_ref[...] > 0)  # [tk, 1]
            if window:
                kpos = kb * tk + jax.lax.broadcasted_iota(jnp.int32, (tk, tq), 0)
                qpos = i * tq + jax.lax.broadcasted_iota(jnp.int32, (tk, tq), 1)
                allowed = allowed & (jnp.abs(qpos - kpos) <= window)
            bias = jnp.broadcast_to(jnp.where(allowed, 0.0, MASK_FILL).astype(F32), (tk, tq))
            for h in range(heads):
                cols = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                q, k, v = q_ref[:, cols], k_ref[:, cols], v_ref[:, cols]
                st = jax.lax.dot_general(k, q * scale, (((1,), (1,)), ((), ())), preferred_element_type=F32) + bias  # [tk, tq]
                m_prev = m_sc[h]  # [1, tq]
                m_new = jnp.maximum(m_prev, st.max(0, keepdims=True))
                alpha = jnp.exp(m_prev - m_new)
                p = jnp.exp(st - m_new)
                l_sc[h] = alpha * l_sc[h] + p.sum(0, keepdims=True)
                acc_sc[h] = alpha * acc_sc[h] + jax.lax.dot_general(v, p.astype(v.dtype), (((0,), (0,)), ((), ())), preferred_element_type=F32)  # [64, tq]
                m_sc[h] = m_new

        @pl.when(kk == steps - 1)
        def _():
            for h in range(heads):
                o_ref[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = jnp.transpose(acc_sc[h] / l_sc[h]).astype(o_ref.dtype)

    scratch = [pltpu.VMEM((heads, 1, tq), F32), pltpu.VMEM((heads, 1, tq), F32), pltpu.VMEM((heads, HEAD_DIM, tq), F32)]
    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, seq, d), out_dtype),
        grid_spec=pl.GridSpec(
            grid=(batch, nq, steps),
            in_specs=[
                pl.BlockSpec((None, tq, d), lambda b, i, kk: (b, i, 0)),
                pl.BlockSpec((None, tk, d), lambda b, i, kk: (b, kv_map(i, kk), 1)),
                pl.BlockSpec((None, tk, d), lambda b, i, kk: (b, kv_map(i, kk), 2)),
                pl.BlockSpec((None, 1, tk), lambda b, i, kk: (b, 0, kv_map(i, kk))),
            ],
            out_specs=pl.BlockSpec((None, tq, d), lambda b, i, kk: (b, i, 0)),
            scratch_shapes=scratch,
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
