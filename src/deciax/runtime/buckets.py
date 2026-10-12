"""Shape buckets: pad (rows, seq_len, options) up to a fixed set so jit compiles once
per bucket rather than once per request shape. Padding is right-side and masked, so
it does not change the real rows' outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from deciax.models.base import EncodedRow

ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
SEQ_BUCKETS = (64, 128, 256, 512, 1024)
OPTION_BUCKETS = (2, 4, 8, 16, 32, 64, 128, 256)


def bucket(n: int, buckets: Sequence[int]) -> int:
    """Smallest bucket >= n."""
    for b in sorted(buckets):
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {max(buckets)}")


def collate(rows: Sequence[EncodedRow], pad_id: int) -> dict[str, npt.NDArray[np.generic]]:
    """Pad rows into the five model inputs at bucketed shapes."""
    n = bucket(len(rows), ROW_BUCKETS)
    s = bucket(max(len(r.input_ids) for r in rows), SEQ_BUCKETS)
    k = bucket(max(len(r.marker_pos) for r in rows), OPTION_BUCKETS)

    input_ids = np.full((n, s), pad_id, np.int32)
    attention_mask = np.zeros((n, s), np.int32)
    marker_pos = np.zeros((n, k), np.int32)
    marker_mask = np.zeros((n, k), bool)
    qtype = np.zeros((n,), np.int32)

    for i, r in enumerate(rows):
        input_ids[i, : len(r.input_ids)] = r.input_ids
        attention_mask[i, : len(r.input_ids)] = 1
        marker_pos[i, : len(r.marker_pos)] = r.marker_pos
        marker_mask[i, : len(r.marker_pos)] = True
        qtype[i] = r.qtype

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }
