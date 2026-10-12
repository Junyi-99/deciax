import jax.numpy as jnp
import numpy as np

from deciax import nn


def test_linear_uses_torch_weight_layout() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.arange(12, dtype=np.float32).reshape(4, 3)
    b = np.ones(4, np.float32)

    np.testing.assert_allclose(
        nn.linear(jnp.asarray(x), jnp.asarray(w), jnp.asarray(b)), x @ w.T + b
    )


def test_layer_norm() -> None:
    x = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    w, b = np.full(8, 2.0, np.float32), np.full(8, 0.5, np.float32)

    want = (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + 1e-5
    ) * w + b
    got = nn.layer_norm(jnp.asarray(x), jnp.asarray(w), jnp.asarray(b), 1e-5)

    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)


def test_rope_preserves_norm_and_leaves_position_zero() -> None:
    x = jnp.asarray(
        np.random.default_rng(0).normal(size=(1, 5, 2, 8)).astype(np.float32)
    )

    y = nn.apply_rope(x, nn.rope_inv_freq(8, 10000.0))

    np.testing.assert_allclose(y[:, 0], x[:, 0], rtol=1e-6)
    np.testing.assert_allclose(
        jnp.linalg.norm(y, axis=-1), jnp.linalg.norm(x, axis=-1), rtol=1e-5
    )
