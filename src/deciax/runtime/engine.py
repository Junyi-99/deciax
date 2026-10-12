"""Runs encoded rows through a model in bucketed chunks."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import jax
import numpy as np

from deciax.models.base import DecisionModel, EncodedRow
from deciax.protocol.output import DecisionOutput
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
