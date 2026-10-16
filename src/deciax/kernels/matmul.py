"""VMEM-local matrix multiplication with M/N/K dot tiles no larger than 512."""

from math import gcd
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def blocked_dot(lhs: Any, rhs: Any, out: Any) -> None:
    """Multiply VMEM refs, retaining each output tile's FP32 K accumulator.

    Ref slices load only the active tile into registers. Results are written
    to VMEM, never to an intermediate HBM buffer.
    """
    m, k = lhs.shape
    rk, n = rhs.shape
    if rk != k or out.shape != (m, n) or any(d <= 0 or d % 256 for d in (m, n, k)):
        raise ValueError("blocked_dot requires compatible 256-aligned dimensions")
    bm, bn, bk = (gcd(d, 512) for d in (m, n, k))

    def output_tile(index: jax.Array, _: None) -> None:
        row, col = pl.ds(index // (n // bn) * bm, bm), pl.ds(index % (n // bn) * bn, bn)

        def reduce_tile(index: jax.Array, acc: jax.Array) -> jax.Array:
            inner = pl.ds(index * bk, bk)
            return acc + jnp.dot(lhs[row, inner], rhs[inner, col], preferred_element_type=jnp.float32)

        acc = jax.lax.fori_loop(0, k // bk, reduce_tile, jnp.zeros((bm, bn), jnp.float32))
        out[row, col] = acc.astype(out.dtype)

    jax.lax.fori_loop(0, (m // bm) * (n // bn), output_tile, None)
