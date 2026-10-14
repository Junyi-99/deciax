# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modified for functional JAX inference. Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.

"""Dense Qwen3.5 text forward, following Transformers 5.10.2 (Apache-2.0).

No persistent KV/conv/recurrent cache. Full attention remains causal. Parameters
use PyTorch [out,in] layout; each layer executable is reused across model depth.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from deciax.kernels.attention.blocked import attention
from deciax.kernels.gdn.reference import chunked
from deciax.models.qwen3_5.config import TextConfig
from deciax.nn import apply_rope, linear, rope_inv_freq


def dense(x: jax.Array, p: Mapping[str, jax.Array], name: str) -> jax.Array:
    return linear(x, p[name + ".weight"], p.get(name + ".bias")).astype(x.dtype)


def rms(x: jax.Array, weight: jax.Array, eps: float, offset: bool = True) -> jax.Array:
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * (weight.astype(jnp.float32) + int(offset))).astype(x.dtype)


def causal_conv(x: jax.Array, weight: jax.Array) -> jax.Array:
    width = weight.shape[-1]
    padded = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
    result = jnp.zeros(x.shape, jnp.float32)

    for i in range(width):
        result += padded[:, i : i + x.shape[1]].astype(jnp.float32) * weight[:, i].astype(jnp.float32)

    return jax.nn.silu(result.astype(x.dtype))


def delta_net(p: Mapping[str, jax.Array], x: jax.Array, mask: jax.Array, c: TextConfig) -> jax.Array:
    x = x * mask[..., None].astype(x.dtype)
    b, s, _ = x.shape
    nh, nv, dk, dv = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim

    q, k, v = [causal_conv(dense(x, p, n), p[n + ".conv"]) for n in ("q", "k", "v")]
    q, k, v = q.reshape(b, s, nh, dk), k.reshape(b, s, nh, dk), v.reshape(b, s, nv, dv)
    q, k = jnp.repeat(q, nv // nh, axis=2), jnp.repeat(k, nv // nh, axis=2)

    a, beta = dense(x, p, "in_proj_a"), jax.nn.sigmoid(dense(x, p, "in_proj_b"))
    g = -jnp.exp(p["A_log"].astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + p["dt_bias"].astype(jnp.float32))
    out = chunked(q, k, v, g, beta)

    z = dense(x, p, "in_proj_z").reshape(out.shape)
    # RMSNormGated rounds normalization before multiplying norm weight, then
    # applies the SiLU gate in float32 (Transformers reference ordering).
    norm = out.astype(jnp.float32) * jax.lax.rsqrt(jnp.mean(out.astype(jnp.float32) ** 2, -1, keepdims=True) + c.rms_norm_eps)
    norm = (norm.astype(x.dtype) * p["norm.weight"]).astype(x.dtype)
    out = (norm.astype(jnp.float32) * jax.nn.silu(z.astype(jnp.float32))).astype(x.dtype).reshape(b, s, nv * dv)

    return dense(out, p, "out_proj")


def full_attention(p: Mapping[str, jax.Array], x: jax.Array, mask: jax.Array, c: TextConfig) -> jax.Array:
    b, s, _ = x.shape
    nh, nk, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim

    raw = dense(x, p, "q_proj").reshape(b, s, nh, 2 * d)
    q, gate = raw[..., :d], raw[..., d:]
    k = dense(x, p, "k_proj").reshape(b, s, nk, d)
    v = dense(x, p, "v_proj").reshape(b, s, nk, d)
    q, k = rms(q, p["q_norm.weight"], c.rms_norm_eps), rms(k, p["k_norm.weight"], c.rms_norm_eps)

    rot = int(d * c.partial_rotary_factor)
    inv = rope_inv_freq(rot, c.rope_theta)
    q = jnp.concatenate((apply_rope(q[..., :rot], inv), q[..., rot:]), axis=-1)
    k = jnp.concatenate((apply_rope(k[..., :rot], inv), k[..., rot:]), axis=-1)

    out = attention(q, k, v, mask, causal=True)
    return dense((out * jax.nn.sigmoid(gate)).reshape(b, s, nh * d), p, "o_proj")


def layer(p: Mapping[str, jax.Array], x: jax.Array, mask: jax.Array, *, config: TextConfig, linear_attention: bool) -> jax.Array:
    h = rms(x, p["input_layernorm.weight"], config.rms_norm_eps)
    mixer = {k.removeprefix("mixer."): v for k, v in p.items() if k.startswith("mixer.")}
    attn = delta_net(mixer, h, mask, config) if linear_attention else full_attention(mixer, h, mask, config)
    x = (x + attn).astype(x.dtype)

    h = rms(x, p["post_attention_layernorm.weight"], config.rms_norm_eps)
    h = jax.nn.silu(dense(h, p, "mlp.gate_proj")) * dense(h, p, "mlp.up_proj")
    return (x + dense(h, p, "mlp.down_proj")).astype(x.dtype)
