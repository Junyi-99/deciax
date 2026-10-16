"""Resident-weight RMSNorm + packed input projections on TPU."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from deciax.kernels.matmul import blocked_dot


def norm_project(x: jax.Array, norm: jax.Array, mask: jax.Array, weight: jax.Array, *, eps: float, tile_rows: int = 512, interpret: bool = False) -> jax.Array:
    """[R,D] -> RMSNorm -> mask -> single packed [D,N] BF16 projection.

    Weight indices do not change across row tiles, so the pipeline loads the
    packed weights once. All GEMM dimensions are multiples of 256. Masks are
    applied after normalization, matching Gated DeltaNet's padding semantics.
    """
    rows, d = x.shape
    n = weight.shape[1]
    if tile_rows <= 0 or tile_rows % 256 or d % 256 or n % 256:
        raise ValueError("projection tiles and dimensions must be positive multiples of 256")
    m = min(tile_rows, (rows + 255) // 256 * 256)
    if any(a.dtype != jnp.bfloat16 for a in (x, norm, weight)):
        raise ValueError("fused Clef projection requires BF16")
    x = jnp.pad(x, ((0, -rows % m), (0, 0)))
    mask = jnp.pad(mask.reshape(rows, 1), ((0, -rows % m), (0, 0))).astype(jnp.int32)

    def kernel(xr: Any, nr: Any, mr: Any, wr: Any, yr: Any, hr: Any) -> None:
        h = xr[...].astype(jnp.float32)
        h *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
        h = (h * (nr[...].astype(jnp.float32) + 1)).astype(jnp.bfloat16)
        h *= mr[...].astype(jnp.bfloat16)
        hr[...] = h
        blocked_dot(hr, wr, yr)

    return pl.pallas_call(  # type: ignore[no-any-return]
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], n), jnp.bfloat16),  # type: ignore[no-untyped-call]
        grid=(x.shape[0] // m,),
        in_specs=[
            pl.BlockSpec((m, d), lambda i: (i, 0)),
            pl.BlockSpec((1, d), lambda i: (0, 0)),
            pl.BlockSpec((m, 1), lambda i: (i, 0)),
            pl.BlockSpec((d, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((m, n), lambda i: (i, 0)),
        scratch_shapes=[pltpu.VMEM((m, d), jnp.bfloat16)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=120 * 1024**2),
        interpret=interpret,
        name=f"clef_norm_project_m{m}_n{n}",
    )(x, norm.reshape(1, d), mask, weight)[:rows]
