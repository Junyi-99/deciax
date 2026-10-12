"""What a decision model returns for a batch of question rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DecisionOutput:
    """Host-side model outputs, one row per (state, question) sequence.

    `option_logits` is [rows, K] (padded slots filled with -1e4), `option_mask` marks
    the real options, `act_logits` is [rows, n_act], `input_tokens` the real tokens
    per row.
    """

    option_logits: npt.NDArray[np.float32]
    option_mask: npt.NDArray[np.bool_]
    act_logits: npt.NDArray[np.float32]
    input_tokens: npt.NDArray[np.int64]
