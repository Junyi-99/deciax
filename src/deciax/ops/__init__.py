"""The only place where the backend forks."""

import jax

from deciax.ops.backend import resolve


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    window: int = 0,
    impl: str = "auto",
) -> jax.Array:
    """Bidirectional attention over [B, S, H, Dh]; `mask` [B, S] marks real tokens,
    `window` > 0 limits each query to keys within that distance."""
    if resolve(impl) == "pallas":
        from deciax.kernels.attention import pallas

        return pallas.attention(q, k, v, mask, window)
    from deciax.kernels.attention import reference

    return reference.attention(q, k, v, mask, window)


__all__ = ["attention"]
