"""Pure-jnp attention: the numerical reference for every kernel."""

import jax
import jax.numpy as jnp

MASK_FILL = -1e30


def attention_mask(mask: jax.Array, window: int = 0) -> jax.Array:
    """[B, S] valid-token mask -> [B, 1, S, S] boolean key mask."""
    allowed = mask.astype(bool)[:, None, None, :]
    if window:
        pos = jnp.arange(mask.shape[1])
        allowed = allowed & (jnp.abs(pos[:, None] - pos[None, :]) <= window)[None, None]
    return allowed


def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, window: int = 0) -> jax.Array:
    """[B, S, H, Dh] -> [B, S, H, Dh] (fp32)."""
    scale = q.shape[-1] ** -0.5
    logits = (
        jnp.einsum(
            "bshd,bthd->bhst",
            q,
            k,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    probs = jax.nn.softmax(jnp.where(attention_mask(mask, window), logits, MASK_FILL), -1)

    return jnp.einsum(
        "bhst,bthd->bshd",
        probs.astype(v.dtype),
        v,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
