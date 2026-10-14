"""Clef joint schema head, ported from Cloudflare's Apache-2.0 reference.

Options are flattened with explicit field ownership; padding never participates
in evidence attention, field attention, or per-field routing softmax.
"""

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from deciax.kernels.attention.blocked import attention
from deciax.models.qwen3_5.text import dense
from deciax.nn import layer_norm


def norm(x: jax.Array, p: Mapping[str, jax.Array], name: str) -> jax.Array:
    return layer_norm(x, p[name + ".weight"], p[name + ".bias"], 1e-5).astype(x.dtype)


def means(x: jax.Array, spans: jax.Array) -> jax.Array:
    # Accumulate in fp32, as torch.mean does for BF16 inputs.
    positions = jnp.arange(x.shape[1])
    mask = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
    total = jnp.einsum("bos,bsd->bod", mask.astype(jnp.float32), x.astype(jnp.float32), precision="highest")
    return (total / jnp.maximum(mask.sum(-1, keepdims=True), 1)).astype(x.dtype)


def gather(x: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.asarray(jax.vmap(lambda row, ids: row[ids])(x, indices))


def normalize(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    length = jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2, -1, keepdims=True)).astype(x.dtype)
    return x / jnp.maximum(length, eps)


def mha(query: jax.Array, memory: jax.Array, mask: jax.Array, p: Mapping[str, jax.Array], name: str, heads: int) -> jax.Array:
    w = p[name + ".in_proj_weight"]
    bias = p[name + ".in_proj_bias"]
    width = query.shape[-1]
    values = []

    for i, x in enumerate((query, memory, memory)):
        values.append(
            dense(x, {"p.weight": w[i * width : (i + 1) * width], "p.bias": bias[i * width : (i + 1) * width]}, "p").reshape(
                *x.shape[:2], heads, width // heads
            )
        )

    out = attention(values[0], values[1], values[2], mask).reshape(query.shape)
    return dense(out, p, name + ".out_proj")


def forward(
    p: Mapping[str, jax.Array], hidden: jax.Array, lexical: jax.Array, a: Mapping[str, jax.Array], *, heads: int, routing_layers: int, layers: int
) -> jax.Array:
    h = norm(hidden, p, "hidden_norm")
    memory = dense(h, p, "memory_projection")
    last = jnp.maximum(a["mask"].sum(-1) - 1, 0)
    global_vector = h[jnp.arange(h.shape[0]), last]

    questions = means(h, a["question_spans"])
    contexts = means(h, a["option_spans"])
    owners = a["option_field"]

    options = dense(contexts, p, "option_context_projection") + dense(lexical, p, "option_lexical_projection")
    options = options + gather(dense(questions, p, "option_question_projection"), owners)

    for i in range(routing_layers):
        n = f"evidence_layers.{i}"
        q, m = norm(options, p, n + ".query_norm"), norm(memory, p, n + ".memory_norm")
        options = options + mha(q, m, a["mask"], p, n + ".attention", heads)
        ff = dense(norm(options, p, n + ".feedforward_norm"), p, n + ".feedforward.0")
        options = options + dense(jax.nn.gelu(ff, approximate=False), p, n + ".feedforward.3")

    base = dense(questions, p, "question_projection")
    logits = jnp.einsum("bow,bqw->bqo", options, base, preferred_element_type=jnp.float32, precision="highest").astype(h.dtype)
    logits = (logits / math.sqrt(base.shape[-1])).astype(h.dtype)
    owns = (owners[:, None, :] == jnp.arange(base.shape[1])[None, :, None]) & a["option_mask"][:, None, :]
    routing = jax.nn.softmax(jnp.where(owns, logits.astype(jnp.float32), -1e30), -1).astype(h.dtype)

    # Reference multiplies in BF16 before the sum.
    summary = jnp.sum((routing[..., None] * options[:, None]).astype(h.dtype), axis=2, dtype=jnp.float32).astype(h.dtype)
    fields = base + norm(summary, p, "option_summary_norm")
    fields = fields + dense(global_vector, p, "global_projection")[:, None]
    fields = fields + p["type_embedding.weight"][a["types"]]

    for i in range(layers):
        n = f"layers.{i}"
        q = norm(fields, p, n + ".norm1")
        fields = fields + mha(q, q, a["question_mask"], p, n + ".self_attn", heads)
        fields = fields + mha(norm(fields, p, n + ".norm2"), memory, a["mask"], p, n + ".multihead_attn", heads)
        ff = dense(norm(fields, p, n + ".norm3"), p, n + ".linear1")
        fields = fields + dense(jax.nn.gelu(ff, approximate=False), p, n + ".linear2")

    fields = gather(norm(fields, p, "field_norm"), owners)
    anchor = normalize(gather(questions, owners) + global_vector[:, None])
    prior = jnp.sum((normalize(lexical) * anchor).astype(h.dtype), -1, dtype=jnp.float32).astype(h.dtype)
    prior *= jnp.exp(jnp.minimum(p["prior_logit_scale"], math.log(100.0)))

    options = norm(options, p, "option_norm")
    cosine = jnp.sum((normalize(fields, 1e-8) * normalize(options, 1e-8)).astype(h.dtype), -1, dtype=jnp.float32).astype(h.dtype)
    features = jnp.concatenate((fields, options, fields * options, jnp.abs(fields - options)), -1)
    residual = dense(jax.nn.gelu(dense(features, p, "residual_scorer.0"), approximate=False), p, "residual_scorer.3")[..., 0]
    joint = jnp.exp(jnp.minimum(p["joint_logit_scale"], math.log(100.0))) * cosine + residual

    return jnp.where(a["option_mask"], prior + jax.nn.sigmoid(p["residual_gate"]) * joint, -1e9).astype(jnp.float32)
