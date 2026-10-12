import pytest

from deciax.protocol.systemone import (
    RequestError,
    add_jev_confidence,
    check_request_limits,
    has_lone_surrogate,
    jev_confidence,
    project_jev_strict,
)


@pytest.mark.parametrize(
    ("answer", "want"),
    [
        ({"type": "choice", "probabilities": {"a": 1.0, "b": 0.0}}, 1.0),
        ({"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}, 0.0),
        ({"type": "choice", "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}, 0.55),
        ({"type": "choice", "probabilities": {"only": 1.0}}, 1.0),
        ({"type": "score", "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}, 1.0),
        ({"type": "score", "probabilities": {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3}}, 0.0),
        ({"type": "noul", "noul": 0.9}, None),
        ({"type": "choice", "probabilities": {"a": True, "b": 0.0}}, None),
    ],
)
def test_jev_confidence(answer: dict[str, object], want: float | None) -> None:
    got = jev_confidence(answer)
    assert got == want if want is None else got == pytest.approx(want)


RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "c": {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.7, "b": 0.3},
            "confidence": 0.1,
            "answer_confidence": 0.7,
            "action": {"act_probability": 1.0},
        },
        "n": {
            "type": "noul",
            "noul": 0.8,
            "confidence": 0.8,
            "answer_confidence": 0.8,
            "action": {"act_probability": 1.0},
        },
    },
    "usage": {"input_tokens": 10, "output_tokens": 0, "truncated": False},
}


def test_add_jev_confidence_copies() -> None:
    out = add_jev_confidence(RESULT)
    assert out["answers"]["c"]["x_jev_confidence"] == 0.4
    assert "x_jev_confidence" not in out["answers"]["n"]
    assert "x_jev_confidence" not in RESULT["answers"]["c"]  # type: ignore[index]


def test_project_jev_strict() -> None:
    out = project_jev_strict(RESULT)
    assert out["answers"]["c"] == {
        "type": "choice",
        "choice": "a",
        "confidence": 0.1,
        "probabilities": {"a": 0.7, "b": 0.3},
    }
    assert out["answers"]["n"] == {"type": "noul", "noul": 0.8}
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 0}


@pytest.mark.parametrize(
    ("state", "questions", "status"),
    [
        (None, {}, 400),
        ("s", [], 400),
        ("s", {f"q{i}": {} for i in range(65)}, 413),
        ("s", {"q": {"type": "choice", "criteria": list(range(101))}}, 413),
        ("s", {"q": {"type": "score", "criteria": ["a", None]}}, 422),
        ("x" * 50001, {}, 413),
    ],
)
def test_request_limits(state: object, questions: object, status: int) -> None:
    with pytest.raises(RequestError) as e:
        check_request_limits(state, questions)
    assert e.value.status == status


def test_lone_surrogate() -> None:
    assert has_lone_surrogate({"a": ["ok", "\ud800"]})
    assert not has_lone_surrogate({"a": ["ok", "🙂"]})
