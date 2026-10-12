"""/v1/systemone request <-> response, wire-compatible with laya's HTTP server."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deciax.protocol.types import Answer, Question, State


@dataclass(frozen=True)
class SystemOneRequest:
    model: str
    state: State
    questions: dict[str, Question]


def parse_request(body: Mapping[str, Any]) -> SystemOneRequest:
    raise NotImplementedError


def render_response(answers: Mapping[str, Answer], input_tokens: int) -> dict[str, Any]:
    raise NotImplementedError
