"""Calibrated option probabilities -> typed answers.

choice: argmax label; score: expected level sum(i * p_i); noul: p[1].
Port of the answer half of laya `Agent._decode_answers` (laya 0.4.4).
"""

import numpy as np
import numpy.typing as npt

from deciax.protocol.types import Answer, Question


def to_answer(
    question: Question,
    probs: npt.NDArray[np.float64],
    act_probs: npt.NDArray[np.float64],
    answer_confidence: float,
) -> Answer:
    raise NotImplementedError
