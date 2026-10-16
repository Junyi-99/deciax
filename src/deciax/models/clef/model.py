"""Text-only Clef checkpoint and tensor-parallel forward pass."""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from numpy.typing import DTypeLike, NDArray
from tokenizers import Tokenizer

from deciax.kernels.swiglu import DEFAULT_TILING, MlpTiling
from deciax.models import ModelName
from deciax.models.base import RequestEngine
from deciax.models.clef import head, weights
from deciax.models.clef.input import MAX_TOKENS
from deciax.models.qwen3_5 import text
from deciax.models.qwen3_5.config import TextConfig
from deciax.ops.backend import ImplChoice
from deciax.protocol.systemone import MAX_BODY_BYTES, MAX_TOTAL_OPTIONS, RequestLimits


class ClefModel:
    max_token_budget = MAX_TOKENS
    request_limits = RequestLimits(state_chars=MAX_BODY_BYTES, choice_options=MAX_TOTAL_OPTIONS, score_levels=MAX_TOTAL_OPTIONS)

    def __init__(
        self,
        path: Path,
        dtype: DTypeLike,
        name: ModelName = "clef",
        *,
        impl: ImplChoice = "auto",
        mlp_tiling: MlpTiling = DEFAULT_TILING,
        fuse_projections: bool = True,
        projection_rows: int = 512,
    ) -> None:
        self.name = name
        if impl not in ("auto", "jax", "pallas"):
            raise ValueError(f"unknown Clef implementation: {impl!r}")
        self.impl = "jax" if impl == "auto" else impl
        if self.impl == "pallas" and (jax.default_backend() != "tpu" or np.dtype(dtype) != jnp.bfloat16):
            raise ValueError("Clef Pallas requires a TPU and BF16 weights")
        self.config = TextConfig.from_dict(json.loads((path / "config.json").read_text()))
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        if pad_id is None:
            raise ValueError("missing pad token")
        self.pad_token_id: int = pad_id

        devices = jax.local_devices()
        if len(devices) not in (1, 2, 4):
            raise ValueError("Clef supports one host with 1, 2, or 4 local devices")
        mesh = Mesh(np.asarray(devices), ("tp",))
        self.replicated = NamedSharding(mesh, weights.REPLICATED)

        self.top, self.layers, self.head = weights.load(path, self.config, dtype, mesh)

        self._embed = jax.jit(lambda w, ids: w[ids], out_shardings=self.replicated)
        layer_fn = text.layer
        if self.impl == "pallas":
            from deciax.models.qwen3_5 import pallas

            for i in range(len(self.layers)):
                self.layers[i] = pallas.pack_layer(self.layers[i], mesh, mlp_tiling)
                if fuse_projections:
                    self.layers[i] = pallas.pack_projections(self.layers[i], mesh)
            layer_fn = functools.partial(pallas.layer, mesh=mesh, tiling=mlp_tiling, projection_rows=projection_rows)
        self._layer = jax.jit(layer_fn, static_argnames=("config", "linear_attention"), out_shardings=self.replicated)
        cfg = json.loads((path / "joint_head_config.json").read_text())
        self._head = jax.jit(
            functools.partial(head.forward, heads=cfg["heads"], routing_layers=cfg["routing_layers"], layers=cfg["layers"]), out_shardings=self.replicated
        )
        self._norm = jax.jit(lambda h, w: text.rms(h, w, self.config.rms_norm_eps))
        self._lexical = jax.jit(lambda w, ids, spans: head.means(w[ids], spans), out_shardings=self.replicated)

    def build_engine(self) -> RequestEngine:
        from deciax.runtime.clef import ClefEngine

        return ClefEngine(self)

    def forward(self, arrays: Mapping[str, NDArray[Any]]) -> jax.Array:
        a = jax.tree.map(lambda v: jax.device_put(v, self.replicated), arrays)

        h = self._embed(self.top["embedding"], a["input_ids"])
        for kind, params in zip(self.config.layer_types, self.layers, strict=True):
            h = self._layer(params, h, a["mask"], config=self.config, linear_attention=kind == "linear_attention")

        h = self._norm(h, self.top["norm"])
        lexical = self._lexical(self.top["lm_head"], a["lexical_ids"], a["lexical_spans"])

        return self._head(self.head, h, lexical, a)  # type: ignore[no-any-return]
