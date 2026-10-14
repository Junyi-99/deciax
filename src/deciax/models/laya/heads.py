"""Laya decision heads in JAX.

- `type_emb[qtype]` is added to every position after the encoder's final norm;
- 2 x `nn.TransformerEncoderLayer`: pre-norm (eps 1e-5, with bias), standard MHA with
  64-dim heads and no positional encoding, ReLU FFN;
- scorer: LayerNorm -> Linear -> exact GELU -> Linear(d, 1), read at each option's
  [MASK] marker; padded slots filled with -1e4;
- act head: [CLS] state ++ (top1, top1 - top2, normalised entropy, K / 255) ->
  Linear -> exact GELU -> Linear(n_act).

Port of the head half of laya-tpu `scripts/jax_model.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from deciax import nn, ops
from deciax.kernels import fused

HEAD_DIM = 64
HEAD_NORM_EPS = 1e-5
OPTION_MASK_FILL = -1e4
ENTROPY_EPS = 1e-9
OPTION_COUNT_SCALE = 255.0


def _norm(p: dict[str, jax.Array], key: str, x: jax.Array) -> jax.Array:
    return nn.layer_norm(x, p[f"{key}.weight"], p[f"{key}.bias"], HEAD_NORM_EPS)


def _linear(p: dict[str, jax.Array], key: str, x: jax.Array) -> jax.Array:
    return nn.linear(x, p[f"{key}.weight"], p[f"{key}.bias"])


def decision_heads(
    p: dict[str, jax.Array],
    x: jax.Array,
    attention_mask: jax.Array,
    marker_pos: jax.Array,
    marker_mask: jax.Array,
    qtype: jax.Array,
    head_layers: int,
    dtype: jnp.dtype,
    impl: str = "auto",
) -> tuple[jax.Array, jax.Array]:
    """[B, S, D] encoder states -> (option logits [B, K], act logits [B, n_act])."""
    batch, seq, d = x.shape
    x = x + p["type_emb.weight"][qtype][:, None, :]

    for i in range(head_layers):
        k = f"head.layers.{i}"
        y = nn.linear(
            _norm(p, f"{k}.norm1", x),
            p[f"{k}.self_attn.in_proj_weight"],
            p[f"{k}.self_attn.in_proj_bias"],
        )
        q, kk, v = (z.reshape(batch, seq, d // HEAD_DIM, HEAD_DIM) for z in jnp.split(y.astype(dtype), 3, -1))

        att = ops.attention(q, kk, v, attention_mask, 0, impl)
        x = x + _linear(p, f"{k}.self_attn.out_proj", att.reshape(batch, seq, d))

        x = x + _linear(
            p,
            f"{k}.linear2",
            jax.nn.relu(_linear(p, f"{k}.linear1", _norm(p, f"{k}.norm2", x))),
        )

    return _scorer(p, x, marker_pos, marker_mask)


def _scorer(p: dict[str, jax.Array], x: jax.Array, marker_pos: jax.Array, marker_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """[B, S, D] head output -> (option logits, act logits)."""
    m = jnp.take_along_axis(x, jnp.maximum(marker_pos, 0)[..., None], axis=1)
    hidden = jax.nn.gelu(_linear(p, "scorer.1", _norm(p, "scorer.0", m)), approximate=False)
    logits = jnp.where(marker_mask, _linear(p, "scorer.3", hidden)[..., 0], OPTION_MASK_FILL)

    probs = jax.nn.softmax(logits, -1)
    count = jnp.maximum(marker_mask.sum(-1), 2)
    ent = -(probs * jnp.log(jnp.maximum(probs, ENTROPY_EPS))).sum(-1) / jnp.log(count)

    if probs.shape[-1] == 1:
        top = jnp.concatenate([jnp.zeros_like(probs), probs], axis=-1)
    else:
        top = jnp.sort(probs, axis=-1)[:, -2:]
    feats = jnp.stack([top[:, -1], top[:, -1] - top[:, -2], ent, count / OPTION_COUNT_SCALE], -1)
    hidden = jax.nn.gelu(
        _linear(p, "act_head.0", jnp.concatenate([x[:, 0], feats], -1)),
        approximate=False,
    )

    return logits, _linear(p, "act_head.2", hidden)


# ---------------------------------------------------------------- fused TPU path (impl="pallas")


def prepare_fused(p: dict[str, jax.Array], head_layers: int) -> dict[str, jax.Array]:
    """Matmul weights transposed to [in, out]."""
    out = {}
    for i in range(head_layers):
        k = f"head.layers.{i}"
        out[f"{k}.in_projT"] = p[f"{k}.self_attn.in_proj_weight"].T
        out[f"{k}.out_projT"] = p[f"{k}.self_attn.out_proj.weight"].T
        out[f"{k}.linear1T"] = p[f"{k}.linear1.weight"].T
        out[f"{k}.linear2T"] = p[f"{k}.linear2.weight"].T
    return out


def decision_heads_fused(
    p: dict[str, jax.Array],
    fp: dict[str, jax.Array],
    x: jax.Array,
    attention_mask: jax.Array,
    marker_pos: jax.Array,
    marker_mask: jax.Array,
    qtype: jax.Array,
    head_layers: int,
) -> tuple[jax.Array, jax.Array]:
    """`decision_heads` on the fused kernels."""
    batch, seq, d = x.shape
    x = (x + p["type_emb.weight"][qtype][:, None, :]).reshape(batch * seq, d)

    for i in range(head_layers):
        k = f"head.layers.{i}"
        w = fused.LayerParams(
            norm1_w=p[f"{k}.norm1.weight"],
            norm1_b=p[f"{k}.norm1.bias"],
            w_qkv_t=fp[f"{k}.in_projT"],
            b_qkv=p[f"{k}.self_attn.in_proj_bias"],
            wo_t=fp[f"{k}.out_projT"],
            bo=p[f"{k}.self_attn.out_proj.bias"],
            norm2_w=p[f"{k}.norm2.weight"],
            norm2_b=p[f"{k}.norm2.bias"],
            wi_t=fp[f"{k}.linear1T"],
            bi=p[f"{k}.linear1.bias"],
            wo2_t=fp[f"{k}.linear2T"],
            b2=p[f"{k}.linear2.bias"],
        )
        with jax.named_scope(f"head{i}"):
            x = fused.encoder_layer(x, attention_mask, w, seq=seq, heads=d // HEAD_DIM, eps=HEAD_NORM_EPS, act="relu")

    with jax.named_scope("scorer"):
        return _scorer(p, x.reshape(batch, seq, d), marker_pos, marker_mask)
