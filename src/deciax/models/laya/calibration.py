"""Laya temperature calibration (laya/common.py + Agent.__init__, laya 0.4.4).

Temperatures are bucketed by question type and option count, fall back to the
per-type temperature, and are clamped to [TEMP_MIN, TEMP_MAX] at load.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from deciax.protocol.types import QTYPES, QuestionType

TEMP_MIN = 0.5
TEMP_MAX = 5.0


def temp_bucket(qtype: QuestionType, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


def clamp_temperature(t: object) -> float:
    """`t` confined to [TEMP_MIN, TEMP_MAX]; 1.0 for anything that is not a finite number."""
    if isinstance(t, bool):
        return 1.0

    try:
        value = float(t)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0

    if not math.isfinite(value):
        return 1.0

    return min(TEMP_MAX, max(TEMP_MIN, value))


class Calibration:
    def __init__(self, temperature: Sequence[object], temperature_by_options: Mapping[str, object]) -> None:
        if len(temperature) != len(QTYPES):
            raise ValueError(f"temperature must have {len(QTYPES)} entries, got {temperature!r}")

        self.temperature = [clamp_temperature(t) for t in temperature]
        self.temperature_by_options = {k: clamp_temperature(v) for k, v in temperature_by_options.items()}

    def temperature_for(self, qtype: QuestionType, k: int) -> float:
        return self.temperature_by_options.get(temp_bucket(qtype, k), self.temperature[QTYPES[qtype]])
