"""The interface every decision-model family implements."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import Question, State


class EncodedBatch(Protocol):
    """Host arrays for one forward pass plus the bookkeeping needed to decode it."""

    arrays: Mapping[str, npt.NDArray[Any]]


class DecisionModel(Protocol):
    def encode(
        self, states: Sequence[State], questions: Mapping[str, Question]
    ) -> EncodedBatch: ...

    def forward(self, batch: EncodedBatch) -> DecisionOutput: ...

    def calibrate(
        self, logits: npt.NDArray[np.float32], qtype: str, k: int, lang: str | None
    ) -> npt.NDArray[np.float64]: ...
