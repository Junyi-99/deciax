"""Runs encoded rows through a model in bucketed chunks."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from deciax.models.base import DecisionModel, EncodedRow
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import State, parse_questions
from deciax.runtime.buckets import ROW_BUCKETS, collate


class Engine:
    def __init__(self, model: DecisionModel) -> None:
        self.model = model
        self._lock = threading.Lock()

    def run(self, rows: Sequence[EncodedRow]) -> DecisionOutput:
        chunk = max(ROW_BUCKETS)
        k = max(len(r.marker_pos) for r in rows)
        logits, acts, masks = [], [], []

        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            arrays = collate(part, self.model.pad_token_id)

            with self._lock:
                out = self.model.forward({key: jax.device_put(v) for key, v in arrays.items()})
                lg, act = (np.asarray(v, np.float32) for v in out)

            n = len(part)
            logits.append(lg[:n, :k])
            masks.append(arrays["marker_mask"][:n, :k])
            acts.append(act[:n])

        return DecisionOutput(
            option_logits=np.concatenate(logits),
            option_mask=np.concatenate(masks),
            act_logits=np.concatenate(acts),
            input_tokens=np.array([len(r.input_ids) for r in rows], np.int64),
        )

    def decide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[Any, Mapping[str, Any]],
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        parsed = parse_questions(questions)
        if not parsed:
            empty = {"input_tokens": 0, "output_tokens": 0}
            return [{"model": "laya-rl-agent", "answers": {}, "usage": dict(empty)} for _ in states]

        per_state = [self.model.encode(s, parsed, max_len, head_max_len) for s in states]
        out = self.run([r for rows in per_state for r in rows])

        results, n = [], len(parsed)
        for i, rows in enumerate(per_state):
            part = slice(i * n, (i + 1) * n)
            sub = DecisionOutput(
                out.option_logits[part],
                out.option_mask[part],
                out.act_logits[part],
                out.input_tokens[part],
            )
            results.append(self.model.decode(rows, sub, parsed))

        return results
