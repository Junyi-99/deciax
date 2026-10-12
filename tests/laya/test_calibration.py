import math

import pytest

from deciax.models.laya.calibration import Calibration, clamp_temperature, temp_bucket


@pytest.mark.parametrize(
    ("k", "bucket"),
    [
        (1, "choice:2"),
        (2, "choice:2"),
        (3, "choice:3-5"),
        (5, "choice:3-5"),
        (6, "choice:6-10"),
        (10, "choice:6-10"),
        (11, "choice:11+"),
    ],
)
def test_temp_bucket(k: int, bucket: str) -> None:
    assert temp_bucket("choice", k) == bucket


@pytest.mark.parametrize(
    ("raw", "clamped"),
    [
        (0.1006, 0.5),
        (7.0, 5.0),
        (1.3, 1.3),
        (True, 1.0),
        ("x", 1.0),
        (None, 1.0),
        (math.nan, 1.0),
        (math.inf, 1.0),
    ],
)
def test_clamp_temperature(raw: object, clamped: float) -> None:
    assert clamp_temperature(raw) == clamped


def test_bucket_then_type_fallback() -> None:
    cal = Calibration([1.2, 0.9, 2.0], {"choice:11+": 0.1006, "noul:2": 1.5})
    assert cal.temperature_for("choice", 12) == 0.5
    assert cal.temperature_for("choice", 4) == 1.2
    assert cal.temperature_for("noul", 2) == 1.5
    assert cal.temperature_for("score", 3) == 0.9


def test_temperature_needs_three_entries() -> None:
    with pytest.raises(ValueError):
        Calibration([1.0, 1.0], {})
