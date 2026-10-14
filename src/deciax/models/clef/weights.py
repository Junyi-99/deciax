"""Read HF tensors on the host and place each directly into its TPU shard."""

import json
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import jax
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.typing import DTypeLike, NDArray
from safetensors import safe_open

from deciax.models.qwen3_5.config import TextConfig

# JAX PartitionSpec lacks annotations, hence the ignores.
REPLICATED = P()  # type: ignore[no-untyped-call]
ROWS = P("tp", None)  # type: ignore[no-untyped-call]
COLUMNS = P(None, "tp")  # type: ignore[no-untyped-call]
SHARDED = P("tp")  # type: ignore[no-untyped-call]


def tensors(path: Path) -> Iterator[tuple[str, NDArray[Any]]]:
    """Memory-map safetensors, including BF16 without a PyTorch dependency."""
    # Have safetensors validate offsets/shapes before mapping BF16 with ml_dtypes.
    with safe_open(path, framework="numpy"):
        pass

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))

    types = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "F16": np.float16}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        yield key, np.memmap(path, mode="r", dtype=types[entry["dtype"]], offset=8 + size + entry["data_offsets"][0], shape=tuple(entry["shape"]))


def layer_weights(raw: Mapping[str, NDArray[Any]], c: TextConfig) -> dict[str, NDArray[Any]]:
    out = {}
    for name, value in raw.items():
        name = name.replace("linear_attn.", "mixer.").replace("self_attn.", "mixer.")
        if name in ("mixer.in_proj_qkv.weight", "mixer.conv1d.weight"):
            nk = c.linear_num_key_heads * c.linear_key_head_dim
            suffix = ".weight" if "in_proj" in name else ".conv"
            if suffix == ".conv":
                value = value[:, 0]
            for part, array in zip(("q", "k", "v"), np.split(value, [nk, 2 * nk]), strict=True):
                out["mixer." + part + suffix] = array
        else:
            out[name] = value

    return out


def spec(name: str, array: NDArray[Any]) -> P:
    if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
        return COLUMNS
    if array.ndim == 2 or name.endswith(".conv"):
        return ROWS
    if name.endswith(("A_log", "dt_bias")):
        return SHARDED
    return REPLICATED


def load(path: Path, config: TextConfig, dtype: DTypeLike, mesh: Mesh) -> tuple[dict[str, jax.Array], list[dict[str, jax.Array]], dict[str, jax.Array]]:
    def put(array: NDArray[Any], partition: P) -> jax.Array:
        host = np.asarray(array, dtype=dtype)
        sharding = NamedSharding(mesh, partition)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])

    layers: list[dict[str, jax.Array]] = [{} for _ in config.layer_types]
    top: dict[str, jax.Array] = {}
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    for filename in sorted(set(index.values())):
        for key, value in tensors(path / filename):
            if key.startswith("model.visual."):
                continue
            if key == "lm_head.weight":
                top["lm_head"] = put(value, ROWS)
            elif key == "model.language_model.embed_tokens.weight":
                top["embedding"] = put(value, ROWS)
            elif key == "model.language_model.norm.weight":
                top["norm"] = put(value, REPLICATED)
            elif key.startswith("model.language_model.layers."):
                _, _, _, number, name = key.split(".", 4)
                for n, v in layer_weights({name: value}, config).items():
                    layers[int(number)][n] = put(v, spec(n, v))
            else:
                raise ValueError(f"unexpected backbone tensor: {key}")

    if set(top) != {"lm_head", "embedding", "norm"} or any(not p for p in layers):
        raise ValueError("incomplete Clef text checkpoint")

    head = {key: put(v, REPLICATED) for key, v in tensors(path / "joint_head.safetensors")}
    return top, layers, head
