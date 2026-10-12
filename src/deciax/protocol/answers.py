"""Calibrated slot probabilities -> typed answers.

Port of the answer half of laya `Agent._decode_answers` plus `unpermute_probs`,
`answer_confidence` and `confidence_from_probs` (laya 0.4.4, Apache-2.0).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from deciax.protocol.types import Answer, Question, render_criterion

type Probs = npt.NDArray[np.floating]


def unpermute_probs(p: Probs, option_order: tuple[int, ...] | None) -> Probs:
    """A slot-ordered row back in the caller's option order."""
    if option_order is None or len(option_order) != len(p):
        return p
    canonical = np.empty_like(p)
    canonical[np.asarray(option_order, dtype=int)] = p
    return canonical


def answer_confidence(p: Probs, k: int) -> float:
    """max(p): the quantity temperature scaling fits."""
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def confidence_from_probs(p: Probs, k: int) -> float:
    """1 - H(p) / log(k). Not calibrated; see `answer_confidence`."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def softmax(z: Probs) -> Probs:
    e = np.exp(z - z.max())
    return e / e.sum()  # type: ignore[no-any-return]


def to_answer(question: Question, slot_probs: Probs, act_probability: float) -> Answer:
    """`slot_probs`: the calibrated distribution over this question's K slots."""
    k = len(slot_probs)
    p = unpermute_probs(slot_probs, question.option_order)
    action = {"act_probability": round(float(act_probability), 4)}
    ans_conf = round(answer_confidence(p, k), 4)

    if question.type == "choice":
        keys = list(question.criteria.keys())
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p, strict=True)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": ans_conf,
            "action": {"act_probability": action["act_probability"]},
        }

    if question.type == "score":
        return {
            "type": "score",
            "score": round(float((np.arange(k) * p).sum()), 4),
            "legend": {str(i): render_criterion(c) for i, c in enumerate(question.criteria)},
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": round(confidence_from_probs(p, k), 4),
            "answer_confidence": ans_conf,
            "action": {"act_probability": action["act_probability"]},
        }

    p_true = float(p[1])
    return {
        "type": "noul",
        "noul": round(p_true, 4),
        "confidence": round(max(p_true, 1.0 - p_true), 4),
        "answer_confidence": ans_conf,
        "action": {"act_probability": action["act_probability"]},
    }
