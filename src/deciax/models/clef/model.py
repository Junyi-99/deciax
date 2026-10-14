"""Text-only Clef checkpoint with request-level execution and tensor parallelism."""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from numpy.typing import DTypeLike, NDArray
from tokenizers import Tokenizer

from deciax.models import ModelName
from deciax.models.clef import head, weights
from deciax.models.clef.input import MAX_TOKENS, collate, decode, encode
from deciax.models.qwen3_5 import text
from deciax.models.qwen3_5.config import TextConfig
from deciax.protocol.systemone import MAX_BODY_BYTES, MAX_TOTAL_OPTIONS, RequestLimits
from deciax.protocol.types import State


class ClefModel:
    max_token_budget = MAX_TOKENS
    request_limits = RequestLimits(state_chars=MAX_BODY_BYTES, choice_options=MAX_TOTAL_OPTIONS, score_levels=MAX_TOTAL_OPTIONS)

    def __init__(self, path: Path, dtype: DTypeLike, name: ModelName = "clef") -> None:
        self.name = name
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
        self._layer = jax.jit(text.layer, static_argnames=("config", "linear_attention"), out_shardings=self.replicated)
        cfg = json.loads((path / "joint_head_config.json").read_text())
        self._head = jax.jit(
            functools.partial(head.forward, heads=cfg["heads"], routing_layers=cfg["routing_layers"], layers=cfg["layers"]), out_shardings=self.replicated
        )
        self._norm = jax.jit(lambda h, w: text.rms(h, w, self.config.rms_norm_eps))
        self._lexical = jax.jit(lambda w, ids, spans: head.means(w[ids], spans), out_shardings=self.replicated)

        self._lock = threading.Lock()

    def build_engine(self) -> ClefModel:
        return self

    def forward(self, arrays: Mapping[str, NDArray[Any]]) -> jax.Array:
        a = jax.tree.map(lambda v: jax.device_put(v, self.replicated), arrays)

        h = self._embed(self.top["embedding"], a["input_ids"])
        for kind, params in zip(self.config.layer_types, self.layers, strict=True):
            h = self._layer(params, h, a["mask"], config=self.config, linear_attention=kind == "linear_attention")

        h = self._norm(h, self.top["norm"])
        lexical = self._lexical(self.top["lm_head"], a["lexical_ids"], a["lexical_spans"])

        return self._head(self.head, h, lexical, a)  # type: ignore[no-any-return]

    def decide_batch(
        self, states: Sequence[State], questions: Mapping[Any, Mapping[str, Any]], *, max_len: int | None = None, head_max_len: int | None = None
    ) -> list[dict[str, Any]]:
        if head_max_len is not None:
            raise ValueError("Clef has one joint token budget; head_max_len is unsupported")

        records = [encode(self.tokenizer, s, questions, self.max_token_budget if max_len is None else max_len) for s in states]
        results: list[dict[str, Any]] = []

        # Bound activation memory independently of the HTTP batch length.
        for start in range(0, len(records), 2):
            part = records[start : start + 2]
            with self._lock:
                logits = np.asarray(self.forward(collate(part, self.pad_token_id)), np.float32)
            results.extend(decode(r, logits[i], questions, self.name) for i, r in enumerate(part))

        return results
