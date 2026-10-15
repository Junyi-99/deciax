"""The `/v1/systemone` wire protocol (TypeSafe Jev), as Laya's HTTP server speaks it.

Request limits, the extra `x_jev_confidence` field and the strict-Jev projection port
laya/serve.py and laya/confidence.py (laya 0.4.4, Apache-2.0). Model-agnostic.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BATCH_STATES = 64
MAX_CHOICE_OPTIONS = 100
MAX_SCORE_LEVELS = 32
MAX_TOTAL_OPTIONS = 512


@dataclass(frozen=True)
class RequestLimits:
    state_chars: int = MAX_STATE_CHARS
    choice_options: int = MAX_CHOICE_OPTIONS
    score_levels: int = MAX_SCORE_LEVELS


DEFAULT_REQUEST_LIMITS = RequestLimits()


LONE_SURROGATE_DETAIL = "request body contains an unpaired surrogate escape; those cannot be encoded as UTF-8"
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


class RequestError(Exception):
    """A client error: HTTP `status` with a `detail` that is safe to return."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _state_length(state: Any) -> int:
    return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))


def check_request_limits(state: Any, questions: Any, limits: RequestLimits = DEFAULT_REQUEST_LIMITS) -> None:
    """Refuse absent or oversized requests before tokenization (400 / 413 / 422)."""
    if state is None:
        raise RequestError(400, "'state' is required")
    if not isinstance(questions, dict):
        raise RequestError(400, "'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(413, f"too many questions ({len(questions)} > {MAX_QUESTIONS})")

    total = 0
    for qid, q in questions.items():
        if not isinstance(q, dict):
            continue
        crit, qtype = q.get("criteria"), q.get("type")
        if qtype == "choice" and isinstance(crit, (dict, list)):
            total += len(crit)
            if len(crit) > limits.choice_options:
                raise RequestError(
                    413,
                    f"too many choice options for {qid!r} ({len(crit)} > {limits.choice_options})",
                )
        elif qtype == "score" and isinstance(crit, list):
            total += len(crit)
            if len(crit) > limits.score_levels:
                raise RequestError(
                    413,
                    f"too many score levels for {qid!r} ({len(crit)} > {limits.score_levels})",
                )
            if None in crit:
                raise RequestError(
                    422,
                    f"score question {qid!r} has a null level at index {crit.index(None)}; give every level a description",
                )

    if total > MAX_TOTAL_OPTIONS:
        raise RequestError(
            413,
            f"too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})",
        )

    length = _state_length(state)
    if length > limits.state_chars:
        raise RequestError(413, f"state too large ({length} > {limits.state_chars} chars)")


def check_batch_limits(states: Any, questions: Any, limits: RequestLimits = DEFAULT_REQUEST_LIMITS) -> None:
    if not isinstance(states, list) or not states:
        raise RequestError(400, "'states' must be a non-empty list")
    if len(states) > MAX_BATCH_STATES:
        raise RequestError(413, f"too many states in batch ({len(states)} > {MAX_BATCH_STATES})")

    for state in states:
        check_request_limits(state, questions, limits)


def has_lone_surrogate(value: Any) -> bool:
    """True if any string in a parsed body holds an unpaired surrogate (iterative walk)."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)

    return False


def jev_confidence(answer: Mapping[str, Any]) -> float | None:
    """The `confidence` Jev reports for this answer's probabilities, or None.

    choice: (max(p) - 1/n) / (1 - 1/n);
    score: max(0, 1 - sum(p_i * |i - mode|) / D), D the same sum for a uniform distribution.
    """
    kind, probabilities = answer.get("type"), answer.get("probabilities")
    if kind not in ("choice", "score") or not isinstance(probabilities, dict) or not probabilities:
        return None
    if any(isinstance(v, bool) for v in probabilities.values()):
        return None

    try:
        if kind == "score":
            p = [float(probabilities[key]) for key in sorted(probabilities, key=int)]
        else:
            p = [float(v) for v in probabilities.values()]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in p):
        return None

    n = len(p)
    if n == 1:
        return 1.0

    total = sum(p)
    p = [v / total for v in p] if total else [1.0 / n] * n

    if kind == "choice":
        return (max(p) - 1.0 / n) / (1.0 - 1.0 / n)

    mode = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - mode) for i, v in enumerate(p))
    center = (n - 1) / 2
    uniform = sum(abs(i - center) for i in range(n)) / n
    return max(0.0, 1.0 - spread / uniform)


def add_jev_confidence(result: Mapping[str, Any]) -> dict[str, Any]:
    """`result` with `x_jev_confidence` on every choice and score answer (copied)."""
    answers = {}
    for qid, answer in result["answers"].items():
        conf = jev_confidence(answer)
        answers[qid] = answer if conf is None else {**answer, "x_jev_confidence": round(conf, 4)}

    return {**result, "answers": answers}


def project_jev_strict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Only the fields of the strict Jev contract."""
    keep = {
        "choice": ("type", "choice", "confidence", "probabilities"),
        "score": ("type", "score", "confidence", "probabilities", "legend"),
        "noul": ("type", "noul"),
    }

    answers = {}
    for qid, answer in result["answers"].items():
        fields = keep.get(answer.get("type"))
        answers[qid] = answer if fields is None else {k: answer[k] for k in fields}

    usage = result.get("usage") or {}
    return {
        "model": result["model"],
        "answers": answers,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }
