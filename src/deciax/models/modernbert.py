"""ModernBERT / mmBERT encoder in JAX.

Global and sliding-window attention layers alternate per `layer_types`; RoPE inverse
frequencies come from the per-layer-type thetas. Port of the encoder half of laya-tpu
`scripts/jax_model.py` (validated against PyTorch there).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from deciax import nn, ops
from deciax.kernels import fused

DEFAULT_THETA = {"full_attention": 160000.0, "sliding_attention": 10000.0}


@dataclass(frozen=True)
class ModernBertConfig:
    hidden_size: int
    num_attention_heads: int
    layer_types: tuple[str, ...]
    norm_eps: float
    rope_theta: tuple[tuple[str, float], ...]
    window: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> ModernBertConfig:
        """From an HF `encoder/config.json` (transformers 4.x or 5.x layout)."""
        n = config["num_hidden_layers"]
        layer_types = config.get("layer_types") or [
            "full_attention" if i % config.get("global_attn_every_n_layers", 3) == 0 else "sliding_attention" for i in range(n)
        ]

        theta = dict(DEFAULT_THETA)
        legacy = {
            "full_attention": "global_rope_theta",
            "sliding_attention": "local_rope_theta",
        }
        rope = config.get("rope_parameters")
        for layer_type, key in legacy.items():
            if isinstance(rope, dict) and isinstance(rope.get(layer_type), dict):
                theta[layer_type] = float(rope[layer_type]["rope_theta"])
            elif config.get(key) is not None:
                theta[layer_type] = float(config[key])

        return cls(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            layer_types=tuple(layer_types),
            norm_eps=config.get("norm_eps", 1e-5),
            rope_theta=tuple(sorted(theta.items())),
            window=config.get("sliding_window") or config.get("local_attention", 128) // 2,
        )


def _norm(p: dict[str, jax.Array], key: str, x: jax.Array, eps: float) -> jax.Array:
    return nn.layer_norm(x, p[f"{key}.weight"], p.get(f"{key}.bias"), eps)


def _linear(p: dict[str, jax.Array], key: str, x: jax.Array) -> jax.Array:
    return nn.linear(x, p[f"{key}.weight"], p.get(f"{key}.bias"))


def encode(
    p: dict[str, jax.Array],
    cfg: ModernBertConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    dtype: jnp.dtype,
    impl: str = "auto",
) -> jax.Array:
    """[B, S] token ids -> [B, S, D] final-norm hidden states (fp32)."""
    batch, seq = input_ids.shape
    heads, dh = cfg.num_attention_heads, cfg.head_dim
    inv_freq = {t: nn.rope_inv_freq(dh, theta) for t, theta in cfg.rope_theta}

    x = p["encoder.embeddings.tok_embeddings.weight"][input_ids].astype(jnp.float32)
    x = _norm(p, "encoder.embeddings.norm", x, cfg.norm_eps)

    for i, layer_type in enumerate(cfg.layer_types):
        k = f"encoder.layers.{i}"
        y = _norm(p, f"{k}.attn_norm", x, cfg.norm_eps) if f"{k}.attn_norm.weight" in p else x
        qkv = _linear(p, f"{k}.attn.Wqkv", y).astype(dtype).reshape(batch, seq, 3, heads, dh)

        q = nn.apply_rope(qkv[:, :, 0], inv_freq[layer_type])
        kk = nn.apply_rope(qkv[:, :, 1], inv_freq[layer_type])
        window = cfg.window if layer_type == "sliding_attention" else 0
        att = ops.attention(q, kk, qkv[:, :, 2], attention_mask, window, impl)
        x = x + _linear(p, f"{k}.attn.Wo", att.reshape(batch, seq, -1))

        h = _linear(p, f"{k}.mlp.Wi", _norm(p, f"{k}.mlp_norm", x, cfg.norm_eps)).astype(dtype)
        a, b = jnp.split(h, 2, -1)
        x = x + _linear(p, f"{k}.mlp.Wo", jax.nn.gelu(a, approximate=False) * b)

    return _norm(p, "encoder.final_norm", x, cfg.norm_eps)


# ---------------------------------------------------------------- fused TPU path (impl="pallas")


def prepare_fused(p: dict[str, jax.Array], cfg: ModernBertConfig) -> dict[str, jax.Array]:
    """Per-layer weights in the layout `kernels.fused` wants: transposed [in, out] matmul weights,
    the gated MLP padded to 128-lane multiples, explicit zero biases (ModernBERT has none)."""
    d = cfg.hidden_size
    zeros, ones = jnp.zeros((d,), jnp.float32), jnp.ones((d,), jnp.float32)
    out = {"zeros": zeros, "ones": ones}
    for i in range(len(cfg.layer_types)):
        k = f"encoder.layers.{i}"
        out[f"{k}.attn_norm"] = p.get(f"{k}.attn_norm.weight", ones)
        out[f"{k}.WqkvT"] = p[f"{k}.attn.Wqkv.weight"].T
        out[f"{k}.WoT"] = p[f"{k}.attn.Wo.weight"].T
        out[f"{k}.mlp_norm"] = p[f"{k}.mlp_norm.weight"]
        out[f"{k}.WiT"], out[f"{k}.Wo2T"] = fused.pad_gated(p[f"{k}.mlp.Wi.weight"], p[f"{k}.mlp.Wo.weight"])
    return out


def encode_fused(
    p: dict[str, jax.Array],
    fp: dict[str, jax.Array],
    cfg: ModernBertConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """`encode` with the three fused kernels per layer; the residual stream stays fp32 [B * S, D]."""
    batch, seq = input_ids.shape
    d = cfg.hidden_size
    rope = {t: fused.rope_tables(nn.rope_inv_freq(cfg.head_dim, theta), seq) for t, theta in cfg.rope_theta}
    zeros = fp["zeros"]
    f_zeros = jnp.zeros((fp["encoder.layers.0.WiT"].shape[1],), jnp.float32)

    x = p["encoder.embeddings.tok_embeddings.weight"][input_ids].astype(jnp.float32)
    x = _norm(p, "encoder.embeddings.norm", x, cfg.norm_eps).reshape(batch * seq, d)

    for i, layer_type in enumerate(cfg.layer_types):
        k = f"encoder.layers.{i}"
        with jax.named_scope(f"layer{i}_{layer_type.split('_')[0]}"):
            x = _layer_fused(p, fp, cfg, k, layer_type, x, attention_mask, rope[layer_type], batch, seq, zeros, f_zeros)

    return _norm(p, "encoder.final_norm", x, cfg.norm_eps).reshape(batch, seq, d)


def _layer_fused(
    p: dict[str, jax.Array],
    fp: dict[str, jax.Array],
    cfg: ModernBertConfig,
    k: str,
    layer_type: str,
    x: jax.Array,
    attention_mask: jax.Array,
    rope: fused.Rope,
    batch: int,
    seq: int,
    zeros: jax.Array,
    f_zeros: jax.Array,
) -> jax.Array:
    """One encoder layer, [B * S, D] fp32 in and out: norm_linear (+ RoPE) -> flash_attention -> out_mlp."""
    from deciax.kernels.attention import pallas

    d, heads = cfg.hidden_size, cfg.num_attention_heads
    qkv = fused.norm_linear(
        x,
        fp[f"{k}.attn_norm"],
        zeros,
        cfg.norm_eps,
        fp[f"{k}.WqkvT"],
        jnp.zeros((3 * d,), jnp.float32),
        normalize=f"{k}.attn_norm.weight" in p,
        rope=rope,
        rope_blocks=2,
        tn=d,
    )
    window = cfg.window if layer_type == "sliding_attention" else 0
    att = pallas.flash_attention(qkv.reshape(batch, seq, 3 * d), attention_mask, heads=heads, window=window)
    return fused.out_mlp(
        att.reshape(batch * seq, d),
        x,
        fp[f"{k}.WoT"],
        zeros,
        fp[f"{k}.mlp_norm"],
        zeros,
        cfg.norm_eps,
        fp[f"{k}.WiT"],
        f_zeros,
        fp[f"{k}.Wo2T"],
        zeros,
        act="gated_gelu",
    )
