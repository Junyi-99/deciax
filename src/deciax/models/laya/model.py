"""Laya = ModernBERT encoder + decision heads, implementing `DecisionModel`."""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from deciax.models.base import EncodedBatch
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import Question, State


class LayaModel:
    def encode(
        self, states: Sequence[State], questions: Mapping[str, Question]
    ) -> EncodedBatch:
        raise NotImplementedError

    def forward(self, batch: EncodedBatch) -> DecisionOutput:
        raise NotImplementedError

    def calibrate(
        self, logits: npt.NDArray[np.float32], qtype: str, k: int, lang: str | None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError
