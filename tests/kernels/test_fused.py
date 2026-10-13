"""The fused TPU kernels against their pure-jnp definitions (`nn`, `kernels.attention.reference`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deciax import nn
from deciax.kernels import fused
from deciax.kernels.attention import reference

pytestmark = pytest.mark.tpu

D, HEADS, EPS = 256, 4, 1e-5


def _normal(key, shape, scale=1.0, dtype=jnp.float32):
    return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _close(got, want, rtol=2e-2, atol=2e-2):
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=rtol, atol=atol)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(("rows", "seq"), [(1024, 256), (1024, 1024)])  # tm >= seq and tm < seq
def test_norm_linear_rope(normalize: bool, rows: int, seq: int) -> None:
    ks = jax.random.split(jax.random.key(0), 5)
    x = _normal(ks[0], (rows, D), 2.0)
    nw, nb = 1.0 + _normal(ks[1], (D,), 0.1), _normal(ks[2], (D,), 0.1)
    w = _normal(ks[3], (3 * D, D), D**-0.5, jnp.bfloat16)  # PyTorch layout [out, in]
    bias = _normal(ks[4], (3 * D,), 0.1)
    inv_freq = nn.rope_inv_freq(64, 10000.0)

    got = fused.norm_linear(x, nw, nb, EPS, w.T, bias, normalize=normalize, rope=fused.rope_tables(inv_freq, seq), rope_blocks=2, tn=D)

    xn = nn.layer_norm(x, nw, nb, EPS) if normalize else x
    y = nn.linear(xn.astype(jnp.bfloat16), w, bias).astype(jnp.bfloat16)  # RoPE runs on the bf16 projection, as in modernbert.encode
    q, k, v = (t.reshape(rows // seq, seq, HEADS, 64) for t in jnp.split(y, 3, -1))
    want = jnp.concatenate([nn.apply_rope(q, inv_freq).reshape(rows, D), nn.apply_rope(k, inv_freq).reshape(rows, D), v.reshape(rows, D)], -1)
    _close(got, want)


@pytest.mark.parametrize("act", ["gated_gelu", "relu"])
def test_out_mlp(act: str) -> None:
    rows, fi = 512, 320 if act == "gated_gelu" else 512  # 320: not a lane multiple, exercises the padding
    ks = jax.random.split(jax.random.key(1), 9)
    att = _normal(ks[0], (rows, D), 1.0, jnp.bfloat16)
    x = _normal(ks[1], (rows, D), 2.0)
    wo = _normal(ks[2], (D, D), D**-0.5, jnp.bfloat16)
    bo, nw, nb, b2 = (_normal(k, (D,), 0.1) for k in ks[3:7])
    wi = _normal(ks[7], ((2 if act == "gated_gelu" else 1) * fi, D), D**-0.5, jnp.bfloat16)
    wo2 = _normal(ks[8], (D, fi), fi**-0.5, jnp.bfloat16)
    bi = jnp.zeros((wi.shape[0],), jnp.float32)

    if act == "gated_gelu":
        wi_t, wo2_t = fused.pad_gated(wi, wo2)
        bi_k = jnp.zeros((wi_t.shape[1],), jnp.float32)
    else:
        wi_t, wo2_t, bi_k = wi.T, wo2.T, bi
    got = fused.out_mlp(att, x, wo.T, bo, nw, nb, EPS, wi_t, bi_k, wo2_t, b2, act=act)

    y = nn.linear(att, wo, bo) + x
    h = nn.linear(nn.layer_norm(y, nw, nb, EPS).astype(jnp.bfloat16), wi, bi)
    if act == "gated_gelu":
        a, b = jnp.split(h, 2, -1)
        g = jax.nn.gelu(a, approximate=False) * b
    else:
        g = jax.nn.relu(h)
    want = y + nn.linear(g.astype(jnp.bfloat16), wo2, b2)
    _close(got, want, rtol=1e-2, atol=5e-2)


@pytest.mark.parametrize(("seq", "window"), [(256, 0), (512, 64), (1024, 64), (1024, 0)])
def test_flash_attention(seq: int, window: int) -> None:
    from deciax.kernels.attention import pallas

    b = 2
    ks = jax.random.split(jax.random.key(2), 3)
    q, k, v = (_normal(key, (b, seq, HEADS, 64), 1.0, jnp.bfloat16) for key in ks)
    mask = jnp.ones((b, seq), jnp.int32).at[1, seq - 37 :].set(0)

    got = pallas.attention(q, k, v, mask, window)
    want = reference.attention(q, k, v, mask, window)
    valid = np.asarray(mask, bool)  # padded queries are never read
    _close(np.asarray(got)[valid], np.asarray(want)[valid])
