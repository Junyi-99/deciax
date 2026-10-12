"""What a decision model returns for a batch of question rows."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DecisionOutput:
    """Host-side model outputs, one row per (state, question) sequence.

    `option_logits` is [rows, K] with padded slots already filled; `option_mask` marks
    the real options. `act_logits` is [rows, n_act]. `truncation` carries per-row
    input statistics (e.g. state tokens dropped); `extras` is model-specific.
    """

    option_logits: npt.NDArray[np.float32]
    option_mask: npt.NDArray[np.bool_]
    act_logits: npt.NDArray[np.float32]
    truncation: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
