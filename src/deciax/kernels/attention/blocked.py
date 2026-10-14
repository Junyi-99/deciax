"""Exact pure-jnp attention for Qwen3.5 / Clef: GQA, cross and causal masks, queries scanned in blocks to bound score memory."""

import jax
import jax.numpy as jnp

from deciax.kernels.attention.reference import MASK_FILL


def attention(q: jax.Array, k: jax.Array, v: jax.Array, key_mask: jax.Array, *, causal: bool = False, block: int = 128) -> jax.Array:
    """[B, Q, H, D], [B, K, Hkv, D] -> [B, Q, H, D]."""
    b, length, heads, dim = q.shape
    k = jnp.repeat(k, heads // k.shape[2], axis=2)
    v = jnp.repeat(v, heads // v.shape[2], axis=2)

    block = min(block, length)
    pad = (-length) % block
    chunks = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0))).reshape(b, -1, block, heads, dim).swapaxes(0, 1)

    def step(_: None, item: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
        idx, query = item
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, k, preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST) * dim**-0.5

        valid = key_mask[:, None, None, :]
        if causal:
            valid = valid & (jnp.arange(k.shape[1])[None, :] <= (idx * block + jnp.arange(block))[:, None])[None, None]

        weights = jax.nn.softmax(jnp.where(valid, scores, MASK_FILL), axis=-1).astype(q.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v, preferred_element_type=jnp.float32, precision=jax.lax.Precision.HIGHEST)
        return None, out.astype(q.dtype)

    _, out = jax.lax.scan(step, None, (jnp.arange(chunks.shape[0]), chunks))
    return jnp.asarray(out).swapaxes(0, 1).reshape(b, -1, heads, dim)[:, :length]
