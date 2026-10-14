"""The fused TPU kernels against their pure-jnp definitions (`nn`, `kernels.attention.reference`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deciax import nn
from deciax.kernels import fused
from deciax.kernels.attention import pallas, reference
from deciax.kernels.fused import Act

pytestmark = pytest.mark.tpu

D, HEADS, EPS = 256, 4, 1e-5


def _normal(key: jax.Array, shape: tuple[int, ...], scale: float = 1.0, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _close(got: jax.Array | np.ndarray, want: jax.Array | np.ndarray, rtol: float = 2e-2, atol: float = 2e-2) -> None:
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=rtol, atol=atol)


def _mlp_weights(k1: jax.Array, k2: jax.Array, act: Act) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """PyTorch-layout wi, wo2 and the kernel layout wi_t, wo2_t. Gated Fi = 320 is not a lane
    multiple, so it exercises `pad_gated`."""
    fi = 320 if act == "gated_gelu" else 512
    wi = _normal(k1, ((2 if act == "gated_gelu" else 1) * fi, D), D**-0.5, jnp.bfloat16)
    wo2 = _normal(k2, (D, fi), fi**-0.5, jnp.bfloat16)
    wi_t, wo2_t = fused.pad_gated(wi, wo2) if act == "gated_gelu" else (wi.T, wo2.T)
    return wi, wo2, wi_t, wo2_t


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
def test_out_mlp(act: Act) -> None:
    rows = 512
    ks = jax.random.split(jax.random.key(1), 9)
    att = _normal(ks[0], (rows, D), 1.0, jnp.bfloat16)
    x = _normal(ks[1], (rows, D), 2.0)
    wo = _normal(ks[2], (D, D), D**-0.5, jnp.bfloat16)
    bo, nw, nb, b2 = (_normal(k, (D,), 0.1) for k in ks[3:7])
    wi, wo2, wi_t, wo2_t = _mlp_weights(ks[7], ks[8], act)
    bi = jnp.zeros((wi_t.shape[1],), jnp.float32)

    got = fused.out_mlp(att, x, wo.T, bo, nw, nb, EPS, wi_t, bi, wo2_t, b2, act=act)

    y = nn.linear(att, wo, bo) + x
    h = nn.linear(nn.layer_norm(y, nw, nb, EPS).astype(jnp.bfloat16), wi)
    if act == "gated_gelu":
        a, b = jnp.split(h, 2, -1)
        g = jax.nn.gelu(a, approximate=False) * b
    else:
        g = jax.nn.relu(h)
    want = y + nn.linear(g.astype(jnp.bfloat16), wo2, b2)
    _close(got, want, rtol=1e-2, atol=5e-2)


@pytest.mark.parametrize(("seq", "window"), [(256, 0), (512, 64), (1024, 64), (1024, 0)])
def test_flash_attention(seq: int, window: int) -> None:
    b = 2
    ks = jax.random.split(jax.random.key(2), 3)
    q, k, v = (_normal(key, (b, seq, HEADS, 64), 1.0, jnp.bfloat16) for key in ks)
    mask = jnp.ones((b, seq), jnp.int32).at[1, seq - 37 :].set(0)

    got = pallas.attention(q, k, v, mask, window)
    want = reference.attention(q, k, v, mask, window)
    valid = np.asarray(mask, bool)  # padded queries are never read
    _close(np.asarray(got)[valid], np.asarray(want)[valid])


@pytest.mark.parametrize(("seq", "window", "act"), [(128, 0, "gated_gelu"), (256, 64, "gated_gelu"), (512, 0, "relu"), (64, 64, "relu")])
def test_layer_matches_three_kernels(seq: int, window: int, act: Act) -> None:
    """One-kernel layer vs norm_linear -> flash_attention -> out_mlp on the same weights."""
    b = 4
    rows = b * seq
    ks = jax.random.split(jax.random.key(3), 12)
    x = _normal(ks[0], (rows, D), 2.0)
    mask = jnp.ones((b, seq), jnp.int32).at[1, seq - 9 :].set(0)
    n1w, n1b, n2w, n2b = 1.0 + _normal(ks[1], (D,), 0.1), _normal(ks[2], (D,), 0.1), 1.0 + _normal(ks[3], (D,), 0.1), _normal(ks[4], (D,), 0.1)
    w_qkv = _normal(ks[5], (3 * D, D), D**-0.5, jnp.bfloat16)
    b_qkv = _normal(ks[6], (3 * D,), 0.1)
    wo = _normal(ks[7], (D, D), D**-0.5, jnp.bfloat16)
    bo, b2 = _normal(ks[8], (D,), 0.1), _normal(ks[9], (D,), 0.1)
    _, _, wi_t, wo2_t = _mlp_weights(ks[10], ks[11], act)
    bi = jnp.zeros((wi_t.shape[1],), jnp.float32)
    rope = fused.rope_tables(nn.rope_inv_freq(64, 10000.0), seq)
    w = fused.LayerParams(n1w, n1b, w_qkv.T, b_qkv, wo.T, bo, n2w, n2b, wi_t, bi, wo2_t, b2)

    got = fused.layer(x, mask, w, seq=seq, heads=HEADS, eps=EPS, act=act, window=window, rope=rope)

    qkv = fused.norm_linear(x, n1w, n1b, EPS, w_qkv.T, b_qkv, rope=rope, rope_blocks=2, tn=D)
    att = pallas.flash_attention(qkv.reshape(b, seq, 3 * D), mask, heads=HEADS, window=window).reshape(rows, D)
    want = fused.out_mlp(att, x, wo.T, bo, n2w, n2b, EPS, wi_t, bi, wo2_t, b2, act=act)
    valid = np.asarray(mask, bool).reshape(-1)
    _close(np.asarray(got)[valid], np.asarray(want)[valid], rtol=1e-2, atol=5e-2)
