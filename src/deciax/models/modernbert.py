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
    """Matmul weights transposed to [in, out], the gated MLP padded by `fused.pad_gated`."""
    out = {}
    for i in range(len(cfg.layer_types)):
        k = f"encoder.layers.{i}"
        out[f"{k}.WqkvT"] = p[f"{k}.attn.Wqkv.weight"].T
        out[f"{k}.WoT"] = p[f"{k}.attn.Wo.weight"].T
        out[f"{k}.WiT"], out[f"{k}.Wo2T"] = fused.pad_gated(p[f"{k}.mlp.Wi.weight"], p[f"{k}.mlp.Wo.weight"])
    return out


def encode_fused(
    p: dict[str, jax.Array],
    fp: dict[str, jax.Array],
    cfg: ModernBertConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """`encode` on the fused kernels; the residual stream stays fp32 [B * S, D]."""
    batch, seq = input_ids.shape
    d = cfg.hidden_size
    rope = {t: fused.rope_tables(nn.rope_inv_freq(cfg.head_dim, theta), seq) for t, theta in cfg.rope_theta}

    x = p["encoder.embeddings.tok_embeddings.weight"][input_ids].astype(jnp.float32)
    x = _norm(p, "encoder.embeddings.norm", x, cfg.norm_eps).reshape(batch * seq, d)

    for i, layer_type in enumerate(cfg.layer_types):
        k = f"encoder.layers.{i}"
        with jax.named_scope(f"layer{i}_{layer_type.split('_')[0]}"):
            x = fused.encoder_layer(
                x,
                attention_mask,
                _fused_params(p, fp, k, d),
                seq=seq,
                heads=cfg.num_attention_heads,
                eps=cfg.norm_eps,
                act="gated_gelu",
                window=cfg.window if layer_type == "sliding_attention" else 0,
                rope=rope[layer_type],
                normalize=f"{k}.attn_norm.weight" in p,
            )

    return _norm(p, "encoder.final_norm", x, cfg.norm_eps).reshape(batch, seq, d)


def _fused_params(p: dict[str, jax.Array], fp: dict[str, jax.Array], k: str, d: int) -> fused.LayerParams:
    """ModernBERT has no biases (fp32 zeros here) and layer 0 has no attn_norm (`normalize=False`)."""
    zeros = jnp.zeros((d,), jnp.float32)
    return fused.LayerParams(
        norm1_w=p.get(f"{k}.attn_norm.weight", jnp.ones((d,), jnp.float32)),
        norm1_b=zeros,
        w_qkv_t=fp[f"{k}.WqkvT"],
        b_qkv=jnp.zeros((3 * d,), jnp.float32),
        wo_t=fp[f"{k}.WoT"],
        bo=zeros,
        norm2_w=p[f"{k}.mlp_norm.weight"],
        norm2_b=zeros,
        wi_t=fp[f"{k}.WiT"],
        bi=jnp.zeros((fp[f"{k}.WiT"].shape[1],), jnp.float32),
        wo2_t=fp[f"{k}.Wo2T"],
        b2=zeros,
    )
