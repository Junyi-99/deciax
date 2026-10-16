"""Clef TPU layer with fused, tensor-parallel projections and MLP.

Attention/Gated DeltaNet cores retain the JAX implementation. Projection and
MLP weights are packed once; TP partial outputs reduce before BF16 rounding.
"""

from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from deciax.kernels.mlp_collective import mlp_reduce_scatter_all_gather
from deciax.kernels.projection import norm_project
from deciax.kernels.swiglu import MlpTiling, fused_mlp_partial, pack
from deciax.models.clef.weights import spec
from deciax.models.qwen3_5 import text
from deciax.models.qwen3_5.config import TextConfig
from deciax.models.qwen3_5.parallel import sum_rows
from deciax.nn import linear

LINEAR_PROJECTIONS = ("q", "k", "v", "in_proj_z", "in_proj_a", "in_proj_b")
FULL_PROJECTIONS = ("q_proj", "k_proj", "v_proj")


def pack_projections(params: dict[str, jax.Array], mesh: Mesh) -> dict[str, jax.Array]:
    names = LINEAR_PROJECTIONS if "mixer.q.weight" in params else FULL_PROJECTIONS
    keys = tuple("mixer." + n + ".weight" for n in names)

    def local(*weights: jax.Array) -> jax.Array:
        # Pad a/b independently to 128 columns, preserving TP head ownership.
        parts = [jnp.pad(w, ((0, -w.shape[0] % 128), (0, 0))).T for w in weights]
        return jnp.concatenate(parts, axis=1)[None]

    packed = jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=tuple(P("tp", None) for _ in keys), out_specs=P("tp", None, None), check_vma=False)  # type: ignore[no-untyped-call]
    )(*(params[n] for n in keys))
    packed.block_until_ready()
    return {**{k: v for k, v in params.items() if k not in keys}, "mixer.input_packed": packed}


def pack_layer(params: dict[str, jax.Array], mesh: Mesh, tiling: MlpTiling) -> dict[str, jax.Array]:
    """Replace the three MLP weights with their packed, padded local shards."""
    pack_shards = jax.jit(
        jax.shard_map(
            lambda g, u, d: pack(g, u, d, tiling.intermediate),
            mesh=mesh,
            in_specs=(P("tp", None), P("tp", None), P(None, "tp")),  # type: ignore[no-untyped-call]
            out_specs=(P("tp", None, None), P("tp", None, None)),  # type: ignore[no-untyped-call]
            check_vma=False,
        )
    )
    names = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
    gate_up, down = pack_shards(*(params[n] for n in names))
    gate_up.block_until_ready()
    down.block_until_ready()
    return {**{k: v for k, v in params.items() if k not in names}, "mlp.gate_up_packed": gate_up, "mlp.down_packed": down}


def mlp(x: jax.Array, p: Mapping[str, jax.Array], *, eps: float, mesh: Mesh, tiling: MlpTiling) -> jax.Array:
    def local(x: jax.Array, n: jax.Array, gu: jax.Array, down: jax.Array) -> jax.Array:
        if tiling.communication_overlap:
            return mlp_reduce_scatter_all_gather(x.reshape(-1, x.shape[-1]), n, gu, down, eps=eps, rows=tiling.rows).reshape(x.shape)
        partial = fused_mlp_partial(x.reshape(-1, x.shape[-1]), n, gu, down, eps=eps, tiling=tiling)
        # Preserve dense() semantics: sum all TP shards in FP32, then cast,
        # then add the BF16 residual. Never round each shard independently.
        projected = sum_rows(partial, axis_size=mesh.size).astype(x.dtype).reshape(x.shape)
        return (x + projected).astype(x.dtype)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P("tp", None, None), P("tp", None, None)),  # type: ignore[no-untyped-call]
        out_specs=P(),  # type: ignore[no-untyped-call]
        check_vma=False,
    )(x, p["post_attention_layernorm.weight"], p["mlp.gate_up_packed"], p["mlp.down_packed"])


def mixer(
    x: jax.Array, mask: jax.Array, p: Mapping[str, jax.Array], *, config: TextConfig, linear_attention: bool, mesh: Mesh, projection_rows: int
) -> jax.Array:
    local_config = replace(
        config,
        num_attention_heads=config.num_attention_heads // mesh.size,
        num_key_value_heads=config.num_key_value_heads // mesh.size,
        linear_num_key_heads=config.linear_num_key_heads // mesh.size,
        linear_num_value_heads=config.linear_num_value_heads // mesh.size,
    )
    params = {k.removeprefix("mixer."): v for k, v in p.items() if k.startswith("mixer.") and k != "mixer.input_packed"}
    specs = {k: spec(k, v) for k, v in params.items()}

    def local(x: jax.Array, mask: jax.Array, norm: jax.Array, packed: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        c = local_config
        widths: tuple[int, ...]
        names: tuple[str, ...]
        projected = norm_project(
            x.reshape(-1, x.shape[-1]), norm, mask if linear_attention else jnp.ones_like(mask), packed[0], eps=c.rms_norm_eps, tile_rows=projection_rows
        ).reshape(*x.shape[:2], -1)
        if linear_attention:
            qk = c.linear_num_key_heads * c.linear_key_head_dim
            zv = c.linear_num_value_heads * c.linear_value_head_dim
            widths = (qk, qk, zv, zv, c.linear_num_value_heads, c.linear_num_value_heads)
            names = LINEAR_PROJECTIONS
        else:
            kv = c.num_key_value_heads * c.head_dim
            widths = (c.num_attention_heads * 2 * c.head_dim, kv, kv)
            names = FULL_PROJECTIONS
        offset = 0
        projections = {}
        for name, width in zip(names, widths, strict=True):
            projections[name] = projected[..., offset : offset + width]
            offset += (width + 127) // 128 * 128
        if linear_attention:
            out = text.delta_net_core(params, projections, c)
            weight = params["out_proj.weight"]
        else:
            out = text.full_attention_core(params, projections, mask, c)
            weight = params["o_proj.weight"]
        return jax.lax.psum(linear(out, weight), "tp").astype(x.dtype)  # type: ignore[no-untyped-call,no-any-return]

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("tp", None, None), specs),  # type: ignore[no-untyped-call]
        out_specs=P(),  # type: ignore[no-untyped-call]
        check_vma=False,
    )(x, mask, p["input_layernorm.weight"], p["mixer.input_packed"], params)


def layer(
    p: Mapping[str, jax.Array],
    x: jax.Array,
    mask: jax.Array,
    *,
    config: TextConfig,
    linear_attention: bool,
    mesh: Mesh,
    tiling: MlpTiling,
    projection_rows: int = 512,
) -> jax.Array:
    if "mixer.input_packed" in p:
        attn = mixer(x, mask, p, config=config, linear_attention=linear_attention, mesh=mesh, projection_rows=projection_rows)
    else:
        h = text.rms(x, p["input_layernorm.weight"], config.rms_norm_eps)
        params = {k.removeprefix("mixer."): v for k, v in p.items() if k.startswith("mixer.")}
        attn = text.delta_net(params, h, mask, config) if linear_attention else text.full_attention(params, h, mask, config)
    x = (x + attn).astype(x.dtype)
    return mlp(x, p, eps=config.rms_norm_eps, mesh=mesh, tiling=tiling)
