"""Clef execution: encode complete schemas and run bounded request batches."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from deciax.models.clef.input import collate, decode, encode
from deciax.protocol.types import State

if TYPE_CHECKING:
    from deciax.models.clef.model import ClefModel


class ClefEngine:
    def __init__(self, model: ClefModel) -> None:
        self.model = model
        self._lock = threading.Lock()

    def decide_batch(
        self, states: Sequence[State], questions: Mapping[Any, Mapping[str, Any]], *, max_len: int | None = None, head_max_len: int | None = None
    ) -> list[dict[str, Any]]:
        if head_max_len is not None:
            raise ValueError("Clef has one joint token budget; head_max_len is unsupported")

        model = self.model
        records = [encode(model.tokenizer, s, questions, model.max_token_budget if max_len is None else max_len) for s in states]
        results: list[dict[str, Any]] = []

        # Bound activation memory independently of the HTTP batch length.
        for start in range(0, len(records), 2):
            part = records[start : start + 2]
            with self._lock:
                logits = np.asarray(model.forward(collate(part, model.pad_token_id)), np.float32)
            results.extend(decode(r, logits[i], questions, model.name) for i, r in enumerate(part))

        return results
