"""Sequence construction.

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

Port of laya `render_options`, `render_criterion`, `build_head`, `build_sequence`,
`state_room` (laya/common.py, 0.4.4). Only the sequential option layout for now;
the return type leaves room for the parallel layout's position/option ids.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from deciax.models.laya.tokenizer import LayaTokenizer
from deciax.protocol.types import Question, State


@dataclass(frozen=True)
class EncodedSequence:
    input_ids: list[int]
    marker_pos: list[int]
    qtype: int
    truncation: dict[str, Any] = field(default_factory=dict)


def render_options(question: Question) -> list[str]:
    raise NotImplementedError


def build_sequence(
    tok: LayaTokenizer,
    state: State,
    question: Question,
    max_len: int,
    head_max_len: int,
    option_order: Sequence[int] | None = None,
    state_ids: Sequence[int] | None = None,
) -> EncodedSequence:
    raise NotImplementedError
