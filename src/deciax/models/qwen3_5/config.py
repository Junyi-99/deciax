"""The supported dense, text-only Qwen3.5 configuration."""

from dataclasses import dataclass
from typing import Any, Literal

type LayerType = Literal["linear_attention", "full_attention"]


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    layer_types: tuple[LayerType, ...]
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25

    def __post_init__(self) -> None:
        sizes = (
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
        )
        if any(size <= 0 for size in sizes):
            raise ValueError("Qwen3.5 dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("Qwen3.5 query/value heads must divide into key head groups")
        if not self.layer_types or set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("unsupported Qwen3.5 layer types")
        rotary = int(self.head_dim * self.partial_rotary_factor)
        if rotary <= 0 or rotary % 2 or rotary > self.head_dim:
            raise ValueError("invalid partial rotary dimension")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TextConfig":
        c = raw.get("text_config", raw)
        if c.get("attention_bias", False) or c.get("hidden_act", "silu") != "silu" or not c.get("attn_output_gate", True):
            raise ValueError("unsupported Qwen3.5 projection/activation configuration")
        if c.get("num_experts", 0):
            raise ValueError("only dense Qwen3.5 models are supported")
        if len(c["layer_types"]) != c["num_hidden_layers"]:
            raise ValueError("layer_types does not match num_hidden_layers")
        rope = c["rope_parameters"]
        if rope.get("rope_type", "default") != "default":
            raise ValueError("only default RoPE is supported")
        keys = [k for k in cls.__dataclass_fields__ if k not in {"layer_types", "rope_theta", "partial_rotary_factor"}]
        return cls(
            **{k: c[k] for k in keys},
            layer_types=tuple(c["layer_types"]),
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25)),
        )
