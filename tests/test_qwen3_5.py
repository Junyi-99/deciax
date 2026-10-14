"""Numerical checks against the independent recurrence and pinned HF implementation."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deciax.kernels.attention.blocked import attention
from deciax.kernels.gdn.reference import chunked, recurrent
from deciax.models.clef.weights import layer_weights
from deciax.models.qwen3_5.config import TextConfig
from deciax.models.qwen3_5.text import layer, rms


@pytest.mark.parametrize("length", [1, 63, 64, 65, 130])
def test_chunked_matches_recurrence(length: int) -> None:
    rng = np.random.default_rng(12)
    q, k, v = [jnp.asarray(rng.normal(size=(2, length, 3, 8)), jnp.float32) for _ in range(3)]
    g = -jnp.asarray(rng.uniform(0, 3, (2, length, 3)), jnp.float32)
    beta = jnp.asarray(rng.uniform(0, 1, (2, length, 3)), jnp.float32)

    np.testing.assert_allclose(chunked(q, k, v, g, beta), recurrent(q, k, v, g, beta), atol=2e-6, rtol=2e-5)


def test_blocked_attention_matches_dense() -> None:
    rng = np.random.default_rng(3)
    s, d = 139, 8  # over one 128-query block
    q = jnp.asarray(rng.normal(size=(2, s, 4, d)), jnp.float32)
    k, v = [jnp.asarray(rng.normal(size=(2, s, 2, d)), jnp.float32) for _ in range(2)]
    mask = jnp.arange(s)[None] < jnp.array([s, 97])[:, None]

    keys, values = [jnp.repeat(x, 2, axis=2) for x in (k, v)]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, keys) * d**-0.5
    causal = jnp.arange(s)[:, None] >= jnp.arange(s)[None, :]
    logits = jnp.where(mask[:, None, None, :] & causal, logits, -1e30)
    expected = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), values)

    np.testing.assert_allclose(attention(q, k, v, mask, causal=True), expected, atol=1e-6, rtol=1e-5)


def test_text_backbone_matches_transformers() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(21)
    c = Qwen3_5TextConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    c._attn_implementation = "eager"
    model: Any = Qwen3_5TextModel(c)
    model.eval()

    config = TextConfig.from_dict(c.to_dict())
    ids = torch.randint(0, 100, (2, 130))
    mask = torch.ones_like(ids)
    mask[1, 99:] = 0

    with torch.no_grad():
        expected = model(ids, attention_mask=mask, use_cache=False).last_hidden_state.numpy()
        h = jnp.asarray(model.embed_tokens(ids).numpy())

    weights = model.state_dict()
    for i, kind in enumerate(config.layer_types):
        prefix = f"layers.{i}."
        params = layer_weights({k[len(prefix) :]: v.numpy() for k, v in weights.items() if k.startswith(prefix)}, config)
        h = layer(jax.tree.map(jnp.asarray, params), h, jnp.asarray(mask.numpy(), bool), config=config, linear_attention=kind == "linear_attention")

    actual = np.asarray(rms(h, jnp.asarray(weights["norm.weight"].numpy()), config.rms_norm_eps))
    valid = mask.numpy().astype(bool)
    np.testing.assert_allclose(actual[valid], expected[valid], atol=3e-6, rtol=3e-5)


def test_aligned_keys_with_weak_decay_are_stable() -> None:
    # This makes powers of the strictly triangular matrix huge even though
    # the delta recurrence is bounded. It regresses the real Clef failure.
    q = k = jnp.ones((1, 130, 2, 128), jnp.bfloat16)
    v = jnp.ones_like(q)
    g = jnp.full((1, 130, 2), -1e-5, jnp.float32)
    beta = jnp.full(g.shape, 0.996, jnp.bfloat16)

    actual = jax.jit(chunked)(q, k, v, g, beta).astype(jnp.float32)
    expected = recurrent(q, k, v, g, beta).astype(jnp.float32)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
