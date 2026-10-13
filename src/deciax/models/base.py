"""Request-engine interface and the row-based interface used by Laya."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax

from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import Question, State


@dataclass(frozen=True)
class EncodedRow:
    """One (state, question) sequence ready for collation."""

    input_ids: list[int]
    marker_pos: list[int]
    qtype: int
    state_stats: dict[str, Any] = field(default_factory=dict)
    option_stats: dict[str, Any] = field(default_factory=dict)


class DecisionModel(Protocol):
    pad_token_id: int

    def encode(
        self,
        state: State,
        questions: Mapping[Any, Question],
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[EncodedRow]: ...

    def forward(self, arrays: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]: ...

    def decode(
        self,
        rows: list[EncodedRow],
        output: DecisionOutput,
        questions: Mapping[Any, Question],
    ) -> dict[str, Any]: ...


class RequestEngine(Protocol):
    def decide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[Any, Mapping[str, Any]],
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[dict[str, Any]]: ...


class RequestModel(Protocol):
    def build_engine(self) -> RequestEngine: ...
