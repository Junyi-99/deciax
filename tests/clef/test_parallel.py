"""RS/AG parity and uneven-row padding on one, two, or four local devices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from deciax.models.qwen3_5.parallel import sum_rows


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 3, 257, 512])
def test_sum_rows(size: int, rows: int) -> None:
    if jax.local_device_count() < size:
        pytest.skip("requires additional local devices; CPU supports XLA_FLAGS=--xla_force_host_platform_device_count=4")
    mesh = Mesh(mesh_utils.create_device_mesh((size,), devices=jax.local_devices()[:size]), ("tp",))
    # Different signed contributions on every shard reveal wrong shard order,
    # duplicated rows, and padding that leaks into the replicated result.
    rng = np.random.default_rng(8)
    partials = rng.normal(size=(size, rows, 256)).astype(np.float32)
    values = jax.device_put(partials, NamedSharding(mesh, P("tp", None, None)))  # type: ignore[no-untyped-call]

    def local(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return sum_rows(x[0], axis_size=size), jax.lax.psum(x[0], "tp")  # type: ignore[no-untyped-call]

    fn = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P("tp", None, None), out_specs=(P(), P()), check_vma=False))  # type: ignore[no-untyped-call]
    actual, reference = fn(values)
    assert actual.shape == (rows, 256)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual, partials.sum(axis=0), atol=1e-6, rtol=1e-6)
    for shard in actual.addressable_shards:
        np.testing.assert_allclose(shard.data, partials.sum(axis=0), atol=1e-6, rtol=1e-6)
