"""Numerical coverage for multi-tile M/N/K reductions."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from deciax.kernels.matmul import blocked_dot


def test_blocked_dot_multi_axis() -> None:
    rng = np.random.default_rng(9)
    lhs = jnp.asarray(rng.normal(0, 0.1, (768, 1536)), jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(0, 0.1, (1536, 1280)), jnp.bfloat16)

    def kernel(a: Any, b: Any, out: Any) -> None:
        blocked_dot(a, b, out)

    actual = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((768, 1280), jnp.float32), interpret=True)(lhs, rhs)  # type: ignore[no-untyped-call]
    expected = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
