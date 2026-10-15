"""Backend-agnostic primitives. Matmuls run in the weight dtype with fp32 accumulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

MATMUL_PRECISION = jax.lax.Precision.HIGHEST


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array | None = None) -> jax.Array:
    """`x @ weight.T (+ bias)` with a PyTorch-layout [out, in] weight; fp32 result."""
    y = jnp.matmul(
        x.astype(weight.dtype),
        weight.T,
        precision=MATMUL_PRECISION,
        preferred_element_type=jnp.float32,
    )
    return y if bias is None else y + bias


def layer_norm(x: jax.Array, weight: jax.Array, bias: jax.Array | None, eps: float) -> jax.Array:
    """LayerNorm computed in fp32."""
    x = x.astype(jnp.float32)
    z = x - x.mean(-1, keepdims=True)
    z = z * jax.lax.rsqrt((z * z).mean(-1, keepdims=True) + eps)
    z = z * weight
    return z if bias is None else z + bias


def rope_inv_freq(head_dim: int, theta: float) -> jax.Array:
    return 1.0 / theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)


def apply_rope(x: jax.Array, inv_freq: jax.Array) -> jax.Array:
    """Rotate-half RoPE on [B, S, H, Dh] using positions 0..S-1."""
    seq_len, half = x.shape[1], x.shape[-1] // 2

    freqs = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.concatenate([jnp.cos(freqs)] * 2, -1).astype(x.dtype)[None, :, None, :]
    sin = jnp.concatenate([jnp.sin(freqs)] * 2, -1).astype(x.dtype)[None, :, None, :]

    rotated = jnp.concatenate([-x[..., half:], x[..., :half]], -1)
    return (x * cos + rotated * sin).astype(x.dtype)
