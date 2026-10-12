"""Question and answer types, wire-compatible with Laya's `system_one`.

Validation and normalisation port laya `Agent._check_question` / `Agent._to_internal`
(laya 0.4.4, Apache-2.0).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

type State = str | dict[str, Any] | list[Any]
type QuestionType = Literal["choice", "score", "noul"]

QTYPES: dict[str, int] = {"choice": 0, "score": 1, "noul": 2}
DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


@dataclass(frozen=True)
class Question:
    """A validated, normalised question.

    `criteria` is a dict label -> description (None for none) for choice, a list of
    level descriptions for score, and a dict with optional "false"/"true" descriptions
    (or None) for noul.
    """

    type: QuestionType
    instructions: str
    criteria: Any
    labels: dict[str, str] | None = None
    option_order: tuple[int, ...] | None = None

    @property
    def option_count(self) -> int:
        return 2 if self.type == "noul" else len(self.criteria)


class Action(TypedDict):
    act_probability: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: Any
    probabilities: dict[Any, float]
    confidence: float
    answer_confidence: float
    action: Action


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float
    confidence: float
    answer_confidence: float
    action: Action


type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


def render_criterion(value: Any) -> str:
    """Strings pass through; anything structured becomes single-line JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def resolve_noul_labels(labels: Any = None) -> tuple[str, str]:
    """(false_label, true_label), validated."""
    if labels is None:
        labels = DEFAULT_NOUL_LABELS

    message = (
        "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
    )

    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError(message)

    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError(message)

    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError(message)

    return false_label, true_label


def _option_count(raw: Mapping[str, Any]) -> int:
    if raw.get("type") == "noul":
        return 2
    crit = raw.get("criteria")
    return len(crit) if isinstance(crit, (dict, list, tuple)) else 0


def _check_choice_labels(qid: Any, crit: dict[Any, Any] | list[Any]) -> None:
    labels = crit if isinstance(crit, list) else list(crit.keys())
    for i, label in enumerate(labels):
        if label is None:
            raise ValueError(
                f"question {qid!r}: choice label {i} is null; a label is rendered as option "
                "text and used as the answer key, so it must be a string, number or bool"
            )
        if not isinstance(label, (str, int, float, bool)):
            raise ValueError(
                f"question {qid!r}: choice label {i} is a {type(label).__name__}; a label is "
                f"rendered as option text and used as the answer key, so it must be a scalar "
                f"(a string, number or bool), got {label!r}"
            )

    if isinstance(crit, list):
        seen: dict[Any, int] = {}
        for i, label in enumerate(crit):
            if label in seen:
                raise ValueError(
                    f"question {qid!r}: choice label {i} ({label!r}) repeats label "
                    f"{seen[label]}; the labels are the answer keys, so every option needs "
                    "its own (1, 1.0 and True are one key)"
                )
            seen[label] = i


def _check(qid: Any, raw: Any) -> None:
    if qid is None:
        raise ValueError("question id must not be None")
    if not isinstance(qid, (str, int)) or (isinstance(qid, str) and not qid.strip()):
        raise ValueError(f"question id must be a non-empty string, got {qid!r}")

    if not isinstance(raw, dict):
        raise ValueError(
            f"question {qid!r}: definition must be a dict, got {type(raw).__name__}"
        )
    t = raw.get("type")
    if not isinstance(t, str) or t not in QTYPES:
        raise ValueError(
            f"question {qid!r}: unknown type {t!r}; use one of {sorted(QTYPES)}"
        )

    if "instructions" not in raw:
        raise ValueError(
            f"question {qid!r}: no 'instructions'; add the text the model should answer"
        )
    ins = raw["instructions"]
    if ins is None:
        raise ValueError(f"question {qid!r}: 'instructions' must not be None")
    if (isinstance(ins, str) and not ins.strip()) or (
        isinstance(ins, (list, dict)) and not ins
    ):
        raise ValueError(f"question {qid!r}: 'instructions' must not be empty")
    if not isinstance(ins, (str, dict, list, int, float)):
        raise ValueError(
            f"question {qid!r}: 'instructions' must be a string, dict, or list, "
            f"got {type(ins).__name__}"
        )

    crit = raw.get("criteria")
    if t == "choice":
        if not isinstance(crit, (dict, list)):
            raise ValueError(
                f"question {qid!r}: a choice question takes 'criteria' as a dict of "
                "label -> description, or a list of labels"
            )
        if not crit:
            raise ValueError(
                f"question {qid!r}: a choice question needs at least one criterion"
            )
        _check_choice_labels(qid, crit)
    elif t == "score":
        if not isinstance(crit, list):
            raise ValueError(
                f"question {qid!r}: a score question takes 'criteria' as a list of level "
                "descriptions, index 0 first"
            )
        if not crit:
            raise ValueError(
                f"question {qid!r}: a score question needs at least one level"
            )
        if None in crit:
            raise ValueError(
                f"question {qid!r}: score level {crit.index(None)} is null"
            )
    elif crit is not None and not isinstance(crit, dict):
        raise ValueError(
            f"question {qid!r}: a noul question takes 'criteria' as a dict with optional "
            "'true'/'false' descriptions, or omits it"
        )
    elif isinstance(crit, dict):
        keys = {str(k).lower() for k in crit}
        if not keys <= {"true", "false"}:
            raise ValueError(
                f"question {qid!r}: a noul question takes 'criteria' keyed only "
                f"'true'/'false', got {sorted(keys)}; use 'labels' to reword the answer"
            )

    if "option_order" in raw:
        order = raw["option_order"]
        n = _option_count(raw)
        ints = (
            [i for i in order if isinstance(i, int) and not isinstance(i, bool)]
            if isinstance(order, (list, tuple))
            else []
        )
        if (
            not isinstance(order, (list, tuple))
            or len(order) != n
            or sorted(ints) != list(range(n))
        ):
            raise ValueError(
                f"question {qid!r}: 'option_order' must be a permutation of range({n}), got {order!r}"
            )

    if "labels" in raw:
        if t != "noul":
            raise ValueError(
                f"question {qid!r}: 'labels' is only supported for noul questions"
            )
        try:
            resolve_noul_labels(raw["labels"])
        except ValueError as e:
            raise ValueError(f"question {qid!r}: {e}") from e


def parse_question(qid: Any, raw: Any) -> Question:
    """Validate and normalise one wire-format question."""
    _check(qid, raw)

    t = raw["type"]
    crit = raw.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}

    ins = raw["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)

    order = raw.get("option_order")
    return Question(
        type=t,
        instructions=ins,
        criteria=crit,
        labels=raw.get("labels"),
        option_order=None if order is None else tuple(int(i) for i in order),
    )


def parse_questions(raw: Mapping[Any, Any]) -> dict[Any, Question]:
    return {qid: parse_question(qid, q) for qid, q in raw.items()}
