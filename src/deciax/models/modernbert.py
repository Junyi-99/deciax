"""ModernBERT / mmBERT encoder in JAX.

Global and sliding-window attention layers alternate per `layer_types`; RoPE inverse
frequencies are computed from the config thetas. Port of the encoder half of
laya-tpu `scripts/jax_model.py`.
"""

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class ModernBertConfig:
    hidden_size: int
    num_attention_heads: int
    layer_types: tuple[str, ...]
    norm_eps: float
    global_rope_theta: float
    local_rope_theta: float
    sliding_window: int

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> ModernBertConfig:
        """From an HF `encoder/config.json` (transformers 4.x or 5.x layout)."""
        raise NotImplementedError


def encode(
    params: dict[str, jax.Array],
    cfg: ModernBertConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """[B, S] token ids -> [B, S, D] final-norm hidden states (fp32)."""
    raise NotImplementedError
