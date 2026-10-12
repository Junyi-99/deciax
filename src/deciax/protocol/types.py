"""Question and answer types, mirroring Laya's `system_one` request/response schema."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

type State = str | Mapping[str, Any] | Sequence[Any]
type QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Choice:
    """Pick one label; `criteria` maps label -> description (None or "" for none)."""

    instructions: str
    criteria: Mapping[str, Any]


@dataclass(frozen=True)
class Score:
    """Rate on an ordinal scale; `criteria` lists the level descriptions in order."""

    instructions: str
    criteria: Sequence[Any]


@dataclass(frozen=True)
class Noul:
    """Yes/no probability; options are always [false, true]."""

    instructions: str
    criteria: Mapping[str, Any] | None = None
    labels: Mapping[str, str] | None = None


type Question = Choice | Score | Noul


@dataclass(frozen=True)
class Action:
    act_probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action
    type: Literal["choice"] = field(default="choice", init=False)


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action
    type: Literal["score"] = field(default="score", init=False)


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    confidence: float
    answer_confidence: float
    action: Action
    type: Literal["noul"] = field(default="noul", init=False)


type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


def parse_question(qid: str, raw: Mapping[str, Any]) -> Question:
    """Validate and normalise one wire-format question.

    Port of laya `Agent._check_question` + `Agent._to_internal` (laya 0.4.4).
    """
    raise NotImplementedError
