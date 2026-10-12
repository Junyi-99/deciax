import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deciax import ops
from deciax.kernels.attention import reference


def _inputs(seed: int = 0, b: int = 2, s: int = 16, h: int = 2, d: int = 8) -> tuple[jax.Array, ...]:
    kq, kk, kv = jax.random.split(jax.random.key(seed), 3)
    q, k, v = (jax.random.normal(key, (b, s, h, d), jnp.float32) for key in (kq, kk, kv))
    mask = jnp.ones((b, s), jnp.int32).at[1, s - 5 :].set(0)
    return q, k, v, mask


def _naive(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    b, s, h, d = q.shape
    out = np.zeros_like(q)

    for bi in range(b):
        for hi in range(h):
            for i in range(s):
                keys = [j for j in range(s) if mask[bi, j] and (not window or abs(i - j) <= window)]
                if not keys:  # padded query with no real key in its window; never read
                    continue

                scores = np.array([q[bi, i, hi] @ k[bi, j, hi] for j in keys]) / np.sqrt(d)
                w = np.exp(scores - scores.max())
                out[bi, i, hi] = (w / w.sum()) @ v[bi, keys, hi]

    return out


@pytest.mark.parametrize("window", [0, 3])
def test_reference_matches_naive(window: int) -> None:
    q, k, v, mask = _inputs()

    got = reference.attention(q, k, v, mask, window)
    want = _naive(np.asarray(q), np.asarray(k), np.asarray(v), np.asarray(mask), window)

    valid = np.asarray(mask, bool)
    np.testing.assert_allclose(np.asarray(got)[valid], want[valid], rtol=1e-5, atol=1e-5)


def test_ops_dispatches_to_reference_off_tpu() -> None:
    q, k, v, mask = _inputs()
    np.testing.assert_array_equal(
        np.asarray(ops.attention(q, k, v, mask, impl="jax")),
        np.asarray(reference.attention(q, k, v, mask)),
    )


@pytest.mark.tpu
@pytest.mark.parametrize("window", [0, 64])
def test_pallas_matches_reference(window: int) -> None:
    q, k, v, mask = _inputs(b=2, s=256, h=4, d=64)

    got = ops.attention(q, k, v, mask, window, impl="pallas")
    want = reference.attention(q, k, v, mask, window)

    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-2, atol=2e-2)
