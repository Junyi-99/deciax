"""Laya decision heads in JAX.

Note the details that differ from the encoder:
- `type_emb[qtype]` is added to every position after the encoder's final norm;
- 2 x `nn.TransformerEncoderLayer`: pre-norm (eps 1e-5, with bias), standard MHA with
  64-dim heads and no positional encoding, **ReLU** FFN;
- scorer: LayerNorm -> Linear -> exact GELU -> Linear(d, 1), read at each option's
  [MASK] marker; padded slots filled with -1e4;
- act head: [CLS] state ++ (top1, top1 - top2, normalised entropy, K / 255) ->
  Linear -> exact GELU -> Linear(n_act).

Port of the head half of laya-tpu `scripts/jax_model.py`.
"""

import jax


def decision_heads(
    params: dict[str, jax.Array],
    hidden: jax.Array,
    attention_mask: jax.Array,
    marker_pos: jax.Array,
    marker_mask: jax.Array,
    qtype: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """[B, S, D] encoder states -> (option logits [B, K], act logits [B, n_act])."""
    raise NotImplementedError
