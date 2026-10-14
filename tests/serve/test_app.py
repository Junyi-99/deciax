import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from deciax.models import ModelName
from deciax.models.clef.model import ClefModel
from deciax.protocol.systemone import MAX_CHOICE_OPTIONS, MAX_STATE_CHARS
from deciax.serve.app import Service

QUESTIONS = {"n": {"type": "noul", "instructions": "Is it?"}}


class FakeDecider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()
        self.release.set()

    def _result(self) -> dict[str, Any]:
        return {
            "model": "laya-rl-agent",
            "answers": {
                "n": {
                    "type": "noul",
                    "noul": 0.9,
                    "confidence": 0.9,
                    "answer_confidence": 0.9,
                    "action": {"act_probability": 1.0},
                },
                "c": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.1,
                    "answer_confidence": 0.7,
                    "action": {"act_probability": 1.0},
                },
            },
            "usage": {"input_tokens": 7, "output_tokens": 0, "served_by": self.name},
        }

    def decide(self, state: Any, questions: Any, **kw: Any) -> dict[str, Any]:
        self.calls.append({"state": state, **kw})
        self.release.wait(5)

        if state == "bad":
            raise ValueError("question 'x': bad")
        if state == "boom":
            raise RuntimeError("secret /path/to/weights")

        return self._result()

    def decide_batch(self, states: Any, questions: Any, **kw: Any) -> list[dict[str, Any]]:
        return [self.decide(s, questions, **kw) for s in states]


def make(**kw: Any) -> tuple[Service, dict[ModelName, FakeDecider]]:
    deciders: dict[ModelName, FakeDecider] = {
        "laya": FakeDecider("laya"),
        "laya-multilingual": FakeDecider("laya-multilingual"),
    }
    return Service(deciders, "laya-multilingual", **kw), deciders


def post(service: Service, body: Any, path: str = "/v1/systemone", **headers: str) -> Any:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return service.handle("POST", path, headers, raw)


def test_systemone_adds_jev_confidence_and_timing() -> None:
    service, _ = make()
    r = post(service, {"state": "s", "questions": QUESTIONS})
    assert r.status == 200
    assert r.payload["answers"]["c"]["x_jev_confidence"] == 0.4
    assert "X-Inference-Time-Ms" in r.headers and r.headers["Server-Timing"].startswith("inference;dur=")


def test_jev_strict() -> None:
    service, _ = make(jev_strict=True)
    r = post(service, {"state": "s", "questions": QUESTIONS})
    assert r.payload["answers"]["n"] == {"type": "noul", "noul": 0.9}
    assert r.payload["usage"] == {"input_tokens": 7, "output_tokens": 0}


@pytest.mark.parametrize(
    ("model", "served_by"),
    [
        (None, "laya-multilingual"),
        ("jev-1", "laya-multilingual"),
        ("convaiinnovations/laya", "laya-multilingual"),
        ("english", "laya"),
        ("EN", "laya"),
        ("convaiinnovations/laya-multilingual", "laya-multilingual"),
        ("ml", "laya-multilingual"),
    ],
)
def test_model_resolution(model: str | None, served_by: ModelName) -> None:
    service, _ = make()
    r = post(service, {"state": "s", "questions": QUESTIONS, "model": model})
    assert r.status == 200 and r.payload["usage"]["served_by"] == served_by


@pytest.mark.parametrize("model", ["typed-decisions", "./ckpt", "someone/other-model"])
def test_model_refused(model: str) -> None:
    service, _ = make()
    assert post(service, {"state": "s", "questions": QUESTIONS, "model": model}).status == 422


def test_model_not_loaded() -> None:
    service = Service({"laya": FakeDecider("laya")}, "laya")
    assert post(service, {"state": "s", "questions": QUESTIONS, "model": "multilingual"}).status == 422


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (b"{not json", 400),
        ({"state": "s"}, 400),
        ({"state": "s", "questions": QUESTIONS, "lang": "de"}, 422),
        ({"state": "s", "questions": QUESTIONS, "hooks": []}, 422),
        ({"state": "s", "questions": QUESTIONS, "max_len": "512"}, 422),
        ({"state": "s", "questions": QUESTIONS, "max_len": True}, 422),
        ({"state": "s", "questions": QUESTIONS, "max_len": 0}, 422),
        ({"state": "s", "questions": QUESTIONS, "max_len": 4096}, 422),
        ({"state": "\ud800", "questions": QUESTIONS}, 400),
        ({"state": "bad", "questions": QUESTIONS}, 422),
    ],
)
def test_client_errors(body: Any, status: int) -> None:
    service, _ = make()
    r = post(service, body)
    assert r.status == status and "detail" in r.payload


def test_internal_error_does_not_leak() -> None:
    service, _ = make()
    r = post(service, {"state": "boom", "questions": QUESTIONS})
    assert r.status == 500 and r.payload == {"detail": "inference failed"}


def test_budgets_are_forwarded() -> None:
    service, deciders = make()
    post(
        service,
        {"state": "s", "questions": QUESTIONS, "max_len": 256, "head_max_len": 64},
    )
    assert deciders["laya-multilingual"].calls[-1] == {
        "state": "s",
        "max_len": 256,
        "head_max_len": 64,
    }


def test_batch() -> None:
    service, _ = make()
    r = post(service, {"states": ["a", "b"], "questions": QUESTIONS}, "/v1/systemone/batch")

    assert r.status == 200
    assert len(r.payload["results"]) == 2
    assert r.payload["total_usage"] == {"input_tokens": 14, "output_tokens": 0}
    assert "x_jev_confidence" in r.payload["results"][0]["answers"]["c"]

    bad = post(
        service,
        {"states": ["a"], "questions": QUESTIONS, "batch_size": 2},
        "/v1/systemone/batch",
    )
    assert bad.status == 422

    assert post(service, {"states": [], "questions": QUESTIONS}, "/v1/systemone/batch").status == 400


def test_auth_and_health() -> None:
    service, _ = make(api_key="sekrit")
    assert post(service, {"state": "s", "questions": QUESTIONS}).status == 401

    ok = post(service, {"state": "s", "questions": QUESTIONS}, authorization="Bearer sekrit")
    assert ok.status == 200

    assert service.handle("GET", "/health", {}, b"").payload == {"status": "ok"}
    full = service.handle("GET", "/health", {"authorization": "Bearer sekrit"}, b"").payload
    assert full["loaded"] == ["laya", "laya-multilingual"] and full["default_model"] == "laya-multilingual"


def test_routing_errors() -> None:
    service, _ = make()
    assert service.handle("GET", "/nope", {}, b"").status == 404
    assert service.handle("GET", "/v1/systemone", {}, b"").status == 405
    assert service.handle("POST", "/health", {}, b"").status == 405


def test_busy_returns_503() -> None:
    service, deciders = make(max_concurrent=1)
    decider = deciders["laya-multilingual"]
    decider.release.clear()

    first = threading.Thread(target=post, args=(service, {"state": "s", "questions": QUESTIONS}))
    first.start()
    while not decider.calls:
        pass

    r = post(service, {"state": "s", "questions": QUESTIONS})
    decider.release.set()
    first.join()

    assert r.status == 503 and r.headers["Retry-After"] == "1"


def test_clef_alias_budget_and_text_only() -> None:
    budget = ClefModel.max_token_budget
    fake = FakeDecider("clef")
    fake.model = SimpleNamespace(max_token_budget=budget)  # type: ignore[attr-defined]
    service = Service({"clef": fake}, "clef")

    assert service.resolve_model("Cloudflare/clef") == "clef"

    result = post(service, {"state": "s", "questions": QUESTIONS, "max_len": budget})
    assert result.status == 200
    assert fake.calls[-1]["max_len"] == budget

    assert post(service, {"state": "s", "questions": QUESTIONS, "max_len": budget + 1}).status == 422
    assert post(service, {"state": "s", "questions": QUESTIONS, "images": []}).status == 422


def test_clef_request_limits_are_used_for_single_and_batch() -> None:
    fake = FakeDecider("clef")
    fake.model = SimpleNamespace(request_limits=ClefModel.request_limits)  # type: ignore[attr-defined]
    service = Service({"clef": fake}, "clef")

    questions = {"intent": {"type": "choice", "criteria": {str(i): str(i) for i in range(MAX_CHOICE_OPTIONS + 1)}}}
    state = "x" * (MAX_STATE_CHARS + 1)  # over the default limits, within Clef's

    assert post(service, {"state": state, "questions": questions}).status == 200
    assert post(service, {"states": [state], "questions": questions}, "/v1/systemone/batch").status == 200
