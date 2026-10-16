"""Fused MLP rounding, padded row/F tails, and tensor-parallel reduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from deciax.kernels.swiglu import MlpTiling, fused_mlp_partial, pack
from deciax.models.qwen3_5 import pallas
from deciax.models.qwen3_5.text import rms
from deciax.nn import linear


@pytest.mark.parametrize("rows,tile", [(17, 256), (257, 512), (777, 256)])
def test_fused_mlp_interpret(rows: int, tile: int) -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    norm = jnp.asarray(rng.normal(0, 0.1, 256), jnp.bfloat16)
    g, u = [jnp.asarray(rng.normal(0, 0.04, (300, 256)), jnp.bfloat16) for _ in range(2)]
    d = jnp.asarray(rng.normal(0, 0.04, (256, 300)), jnp.bfloat16)
    gu, down = pack(g, u, d, tile)
    h = rms(x, norm, 1e-6)
    expected = linear(jax.nn.silu(linear(h, g).astype(x.dtype)) * linear(h, u).astype(x.dtype), d)
    actual = fused_mlp_partial(x, norm, gu, down, eps=1e-6, tiling=MlpTiling(rows=512, intermediate=tile), interpret=True)
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("interpret", [pytest.param(True, id="interpret"), pytest.param(False, marks=pytest.mark.tpu, id="tpu")])
@pytest.mark.parametrize("rows,hidden,tile", [(257, 768, 256), (257, 768, 512), (17, 5120, 256)])
def test_fused_mlp_multiblock_tails(interpret: bool, rows: int, hidden: int, tile: int) -> None:
    """Exercise five F blocks, multiple output tiles, and padded rows/F.

    All intermediate blocks have nonzero weights, including the final F tail,
    so dropping a reduction block or prematurely writing an accumulator fails.
    D=5120 also covers five down DMA groups with two output subtiles per group.
    """
    rng = np.random.default_rng(19)
    intermediate = 4 * tile + 1
    x = jnp.asarray(rng.normal(size=(rows, hidden)), jnp.bfloat16)
    norm = jnp.asarray(rng.normal(0, 0.1, hidden), jnp.bfloat16)
    # Keep projection variance comparable across widths so the same absolute
    # error threshold measures BF16 rounding at a consistent output scale.
    gate_up_scale = 0.04 * np.sqrt(768 / hidden)
    down_scale = 0.04 * np.sqrt(1025 / intermediate)
    gate, up = [jnp.asarray(rng.normal(0, gate_up_scale, (intermediate, hidden)), jnp.bfloat16) for _ in range(2)]
    down = jnp.asarray(rng.normal(0, down_scale, (hidden, intermediate)), jnp.bfloat16)
    packed_gate_up, packed_down = pack(gate, up, down, tile)
    assert packed_gate_up.shape[0] == 5

    h = rms(x, norm, 1e-6)
    activated = jax.nn.silu(linear(h, gate).astype(x.dtype)) * linear(h, up).astype(x.dtype)
    expected = np.asarray(linear(activated, down), np.float32)
    actual = np.asarray(
        fused_mlp_partial(x, norm, packed_gate_up, packed_down, eps=1e-6, tiling=MlpTiling(rows=512, intermediate=tile), interpret=interpret),
        np.float32,
    )
    assert actual.shape == (rows, hidden)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005


@pytest.mark.tpu
@pytest.mark.parametrize("staged", [False, True])
def test_tensor_parallel_mlp(staged: bool) -> None:
    if jax.device_count() != 4:
        pytest.skip("requires four TPU chips")
    mesh = Mesh(np.asarray(jax.devices()), ("tp",))
    rng = np.random.default_rng(5)

    def array(shape: tuple[int, ...], spec: P, scale: float = 0.05) -> jax.Array:
        return jax.device_put(jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16), NamedSharding(mesh, spec))  # type: ignore[no-any-return]

    x = array((1, 257, 256), P(), 1)  # type: ignore[no-untyped-call]
    weights = {
        "mlp.gate_proj.weight": array((768, 256), P("tp", None)),  # type: ignore[no-untyped-call]
        "mlp.up_proj.weight": array((768, 256), P("tp", None)),  # type: ignore[no-untyped-call]
        "mlp.down_proj.weight": array((256, 768), P(None, "tp")),  # type: ignore[no-untyped-call]
        "post_attention_layernorm.weight": array((256,), P()),  # type: ignore[no-untyped-call]
    }

    @jax.jit
    def reference(x: jax.Array, w: dict[str, jax.Array]) -> jax.Array:
        h = rms(x, w["post_attention_layernorm.weight"], 1e-6)
        h = jax.nn.silu(linear(h, w["mlp.gate_proj.weight"]).astype(x.dtype)) * linear(h, w["mlp.up_proj.weight"]).astype(x.dtype)
        return (x + linear(h, w["mlp.down_proj.weight"]).astype(x.dtype)).astype(x.dtype)

    expected = np.asarray(reference(x, weights), np.float32)
    tiling = MlpTiling(intermediate=256, vmem_limit_bytes=32 * 1024**2, staged=staged)
    packed = pallas.pack_layer(weights, mesh, tiling)
    actual = np.asarray(jax.jit(lambda x, w: pallas.mlp(x, w, eps=1e-6, mesh=mesh, tiling=tiling))(x, packed), np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.032, rtol=0.02)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005
