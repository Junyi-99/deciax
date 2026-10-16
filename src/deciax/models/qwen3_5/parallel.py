"""Explicit row-wise collectives for tensor-parallel Clef projections."""

import jax
import jax.numpy as jnp


def sum_rows(partial: jax.Array, *, axis_size: int) -> jax.Array:
    """Reduce FP32 [rows, hidden] partials, then replicate the reduced rows.

    Row sharding preserves complete hidden vectors. Padding handles batches
    whose flattened row count is not divisible by the tensor-parallel size.
    Both collectives follow the complete local projection; this is not a
    compute/communication pipeline.
    """
    if axis_size == 1:
        return partial
    rows = partial.shape[0]
    padded = jnp.pad(partial, ((0, -rows % axis_size), (0, 0)))
    reduced = jax.lax.psum_scatter(padded, "tp", scatter_dimension=0, tiled=True)  # type: ignore[no-untyped-call]
    return jax.lax.all_gather(reduced, "tp", axis=0, tiled=True)[:rows]  # type: ignore[no-any-return]
