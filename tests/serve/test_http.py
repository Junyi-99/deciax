"""End to end over a real socket with a real checkpoint (CPU fp32)."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from deciax.decider import Decider
from deciax.models.laya.model import LayaModel
from deciax.protocol.systemone import add_jev_confidence
from deciax.serve.app import Service, make_server
from tests.conftest import checkpoint_dir, laya_fixture


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[str, Decider]]:
    path = checkpoint_dir("laya")
    with jax.default_device(jax.devices("cpu")[0]):
        decider = Decider(LayaModel(path, jnp.float32, "jax"))

    httpd = make_server(Service({"laya": decider}, "laya"), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_address[1]}", decider

    httpd.shutdown()
    httpd.server_close()


def call(url: str, body: Any = None) -> tuple[int, Any, dict[str, str]]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def test_systemone_matches_decider(server: tuple[str, Decider]) -> None:
    url, decider = server
    case = laya_fixture("laya")["cases"][0]

    with jax.default_device(jax.devices("cpu")[0]):
        status, got, headers = call(
            url + "/v1/systemone",
            {"state": case["state"], "questions": case["questions"]},
        )
        want = add_jev_confidence(decider.decide(case["state"], case["questions"]))

    assert status == 200
    assert got == json.loads(json.dumps(want))
    assert float(headers["X-Inference-Time-Ms"]) > 0


def test_batch_and_health(server: tuple[str, Decider]) -> None:
    url, _ = server
    case = laya_fixture("laya")["cases"][0]

    with jax.default_device(jax.devices("cpu")[0]):
        status, got, _ = call(
            url + "/v1/systemone/batch",
            {"states": [case["state"], "hello"], "questions": case["questions"]},
        )
    assert status == 200 and len(got["results"]) == 2

    status, health, _ = call(url + "/health")
    assert status == 200 and health["loaded"] == ["laya"]


def test_validation_error_is_422(server: tuple[str, Decider]) -> None:
    url, _ = server
    status, got, _ = call(
        url + "/v1/systemone",
        {"state": "s", "questions": {"q": {"type": "nope", "instructions": "x"}}},
    )
    assert status == 422 and "unknown type" in got["detail"]
