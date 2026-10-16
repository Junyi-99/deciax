import jax.numpy as jnp
import numpy as np

from deciax.kernels.projection import norm_project
from deciax.models.qwen3_5.text import rms


def test_masked_projection_interpret() -> None:
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(277, 256)), jnp.bfloat16)
    norm = jnp.asarray(rng.normal(0, 0.1, 256), jnp.bfloat16)
    w = jnp.asarray(rng.normal(0, 0.05, (256, 512)), jnp.bfloat16)
    mask = jnp.arange(277) < 199
    expected = (rms(x, norm, 1e-6) * mask[:, None]) @ w
    actual = norm_project(x, norm, mask, w, eps=1e-6, interpret=True)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=0.032, rtol=0.02)
    np.testing.assert_array_equal(actual[199:], 0)


def test_projection_requires_aligned_tiles() -> None:
    import pytest

    with pytest.raises(ValueError, match="multiples of 256"):
        norm_project(
            jnp.zeros((8, 256), jnp.bfloat16), jnp.zeros(256, jnp.bfloat16), jnp.ones(8, bool), jnp.zeros((256, 512), jnp.bfloat16), eps=1e-6, tile_rows=128
        )
