"""HTTP server for the `/v1/systemone` protocol, wire-compatible with `laya-serve`.

Routes: `POST /v1/systemone`, `POST /v1/systemone/batch`, `GET /health`. Bodies, limits,
status codes, the `x_jev_confidence` field and the strict-Jev projection follow
laya/serve.py (laya 0.4.4). Differences: no language routing (a request that names no
checkpoint gets the default model), and the controls deciax does not implement
(`task`, `lang`, `lang_guess`, `min_confidence`, `batch_size`, `sort_by_length`) are
refused with 422 rather than ignored.

`Service` holds the protocol logic and is independent of the HTTP framework; the
stdlib `ThreadingHTTPServer` below is a thin adapter around it.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Protocol

from deciax.models import MODELS, ModelName
from deciax.ops.backend import ImplChoice
from deciax.protocol.systemone import (
    LONE_SURROGATE_DETAIL,
    MAX_BODY_BYTES,
    RequestError,
    RequestLimits,
    add_jev_confidence,
    check_batch_limits,
    check_request_limits,
    has_lone_surrogate,
    project_jev_strict,
)
from deciax.runtime.buckets import SEQ_BUCKETS

if TYPE_CHECKING:
    from deciax.decider import DtypeChoice

log = logging.getLogger("deciax.serve")

BUNDLE_REPO = "convaiinnovations/laya"
ALIASES: dict[str, ModelName] = {
    "clef": "clef",
    "clef-flash": "clef-flash",
    "english": "laya",
    "en": "laya",
    "default": "laya",
    "laya": "laya",
    "multilingual": "laya-multilingual",
    "multi": "laya-multilingual",
    "ml": "laya-multilingual",
    "laya-multilingual": "laya-multilingual",
}
UNAVAILABLE = {
    "typed-decisions",
    "typed",
    "typed_decisions",
    "laya-typed-decisions",
    "decisions",
}
REFUSED = (
    "hooks",
    "on_predict_start",
    "on_predict_end",
    "hooks_raise",
    "hooks_timeout",
)
UNSUPPORTED = ("task", "lang", "lang_guess", "min_confidence", "images", "videos", "media_kwargs")
UNSUPPORTED_BATCH = ("batch_size", "sort_by_length", "diagnostics")
MAX_TOKEN_BUDGET = max(SEQ_BUCKETS)
LIVENESS_ONLY = {"status": "ok"}


class Decides(Protocol):
    def decide(
        self,
        state: Any,
        questions: Mapping[Any, Any],
        *,
        max_len: int | None = ...,
        head_max_len: int | None = ...,
    ) -> dict[str, Any]: ...

    def decide_batch(
        self,
        states: Any,
        questions: Mapping[Any, Any],
        *,
        max_len: int | None = ...,
        head_max_len: int | None = ...,
    ) -> list[dict[str, Any]]: ...


@dataclass
class Response:
    status: int
    payload: Any
    headers: dict[str, str] = field(default_factory=dict)


def _error(status: int, detail: str, headers: dict[str, str] | None = None) -> Response:
    return Response(status, {"detail": detail}, headers or {})


class Service:
    def __init__(
        self,
        deciders: Mapping[ModelName, Decides],
        default_model: ModelName,
        *,
        api_key: str | None = None,
        max_concurrent: int = 16,
        jev_strict: bool = False,
        device: str = "unknown",
    ) -> None:
        if default_model not in deciders:
            raise ValueError(f"default model {default_model!r} is not loaded")

        self.deciders = dict(deciders)
        self.default_model = default_model
        self.jev_strict = jev_strict
        self.device = device
        self._expected = ("Bearer " + api_key).encode() if api_key else None
        self._admission = threading.BoundedSemaphore(max_concurrent)

    # -- routing ---------------------------------------------------------------------
    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> Response:
        path = path.split("?", 1)[0].rstrip("/") or "/"
        routes: dict[str, tuple[str, Callable[[bytes], Response]]] = {
            "/v1/systemone": ("POST", self._systemone),
            "/v1/systemone/batch": ("POST", self._batch),
        }

        authorization = headers.get("authorization")
        if path == "/health":
            if method != "GET":
                return _error(405, "Method Not Allowed")
            return Response(200, self._health(authorization))

        if path not in routes:
            return _error(404, "Not Found")
        allowed, handler = routes[path]
        if method != allowed:
            return _error(405, "Method Not Allowed")

        if not self.authorized(authorization):
            return _error(401, "invalid or missing bearer token")
        if not self._admission.acquire(blocking=False):
            return _error(503, "server busy, try again later", {"Retry-After": "1"})

        try:
            return handler(body)
        except RequestError as e:
            return _error(e.status, e.detail)
        finally:
            self._admission.release()

    def authorized(self, authorization: str | None) -> bool:
        if self._expected is None:
            return True
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        return hmac.compare_digest(supplied, self._expected)

    # -- endpoints -------------------------------------------------------------------
    def _health(self, authorization: str | None) -> dict[str, Any]:
        if not self.authorized(authorization):
            return dict(LIVENESS_ONLY)
        return {
            "status": "ok",
            "loaded": sorted(self.deciders),
            "default_model": self.default_model,
            "revisions": {name: MODELS[name].revision for name in sorted(self.deciders)},
            "device": self.device,
        }

    def _systemone(self, raw: bytes) -> Response:
        body = _parse_body(raw)
        if not isinstance(body, dict) or "questions" not in body:
            raise RequestError(400, "request body must be an object with a 'questions' field")

        state, questions = body.get("state"), body["questions"]
        decider = self.deciders[self.resolve_model(body.get("model"))]
        limits = getattr(getattr(decider, "model", None), "request_limits", RequestLimits())
        check_request_limits(state, questions, limits)
        _refuse_controls(body, UNSUPPORTED)
        if has_lone_surrogate(body):
            raise RequestError(400, LONE_SURROGATE_DETAIL)

        budgets = _budgets(body, getattr(getattr(decider, "model", None), "max_token_budget", MAX_TOKEN_BUDGET))
        result, ms = self._infer(lambda: decider.decide(state, questions, **budgets))

        return Response(200, self._finish(result), _timing(ms))

    def _batch(self, raw: bytes) -> Response:
        body = _parse_body(raw)
        if not isinstance(body, dict) or "questions" not in body or "states" not in body:
            raise RequestError(
                400,
                "request body must be an object with 'states' and 'questions' fields",
            )

        states, questions = body["states"], body["questions"]
        decider = self.deciders[self.resolve_model(body.get("model"))]
        limits = getattr(getattr(decider, "model", None), "request_limits", RequestLimits())
        check_batch_limits(states, questions, limits)
        _refuse_controls(body, UNSUPPORTED + UNSUPPORTED_BATCH)
        if has_lone_surrogate(body):
            raise RequestError(400, LONE_SURROGATE_DETAIL)

        budgets = _budgets(body, getattr(getattr(decider, "model", None), "max_token_budget", MAX_TOKEN_BUDGET))
        results, ms = self._infer(lambda: decider.decide_batch(states, questions, **budgets))

        payload = {
            "results": [self._finish(r) for r in results],
            "total_usage": {
                "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in results),
                "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in results),
            },
        }
        return Response(200, payload, _timing(ms))

    # -- helpers ---------------------------------------------------------------------
    def resolve_model(self, model: Any) -> ModelName:
        """A loaded checkpoint name for the request's `model` field.

        Absent, the root bundle id or an unknown id such as `jev-1` mean "let the
        server choose" (the default model). A known checkpoint this server does not
        serve, a path, or an unpublished Hub repo id is a 422.
        """
        if not model:
            return self.default_model

        text = str(model).strip()
        key = text.lower()
        published: dict[str, ModelName] = {c.repo.lower(): name for name, c in MODELS.items() if c.repo != BUNDLE_REPO}
        name = published.get(key) or ALIASES.get(key)

        if name is None and key in UNAVAILABLE:
            raise RequestError(422, f"model {text!r} is not available in deciax")

        if name is None:
            if key == BUNDLE_REPO:
                return self.default_model
            if text[:1] in ".~" or "/" in text or "\\" in text:
                raise RequestError(
                    422,
                    f"unknown model {text!r}; this server serves {sorted(self.deciders)}, or omit model to let the server choose",
                )
            return self.default_model

        if name not in self.deciders:
            raise RequestError(
                422,
                f"model {text!r} is not loaded; this server serves {sorted(self.deciders)}",
            )
        return name

    def _infer(self, run: Callable[[], Any]) -> tuple[Any, float]:
        t0 = time.perf_counter()
        try:
            result = run()
        except RequestError:
            raise
        except ValueError as e:
            raise RequestError(422, str(e)) from None
        except Exception:
            log.exception("inference failed")
            raise RequestError(500, "inference failed") from None

        return result, (time.perf_counter() - t0) * 1000.0

    def _finish(self, result: dict[str, Any]) -> dict[str, Any]:
        return project_jev_strict(result) if self.jev_strict else add_jev_confidence(result)


def _parse_body(raw: bytes) -> Any:
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, "request body too large")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        raise RequestError(400, "request body must be valid JSON") from None


def _refuse_controls(body: Mapping[str, Any], unsupported: tuple[str, ...]) -> None:
    refused = [k for k in REFUSED if k in body]
    if refused:
        raise RequestError(
            422,
            f"{', '.join(refused)} cannot be set from a request; hooks belong to the server",
        )

    sent = [k for k in unsupported if body.get(k) is not None]
    if sent:
        raise RequestError(422, f"{', '.join(sent)} is not supported by deciax")


def _budgets(body: Mapping[str, Any], limit: int = MAX_TOKEN_BUDGET) -> dict[str, int]:
    budgets = {}
    for key in ("max_len", "head_max_len"):
        value = body.get(key)
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, int):
            raise RequestError(422, f"{key} must be an integer")
        if value <= 0:
            raise RequestError(422, f"{key} must be a positive integer")
        if value > limit:
            raise RequestError(422, f"{key} exceeds the server cap ({value} > {limit})")

        budgets[key] = value

    return budgets


def _timing(ms: float) -> dict[str, str]:
    return {
        "Server-Timing": f"inference;dur={ms:.2f}",
        "X-Inference-Time-Ms": f"{ms:.2f}",
    }


# -- HTTP adapter ------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    server: _Server

    def _respond(self, response: Response) -> None:
        data = json.dumps(response.payload, ensure_ascii=False).encode("utf-8")

        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self) -> None:
        body = b""
        if self.command == "POST":
            length = self.headers.get("Content-Length")
            if length is None:
                self.close_connection = True
                return self._respond(_error(411, "Content-Length required"))

            try:
                n = int(length)
            except ValueError:
                self.close_connection = True
                return self._respond(_error(400, "invalid Content-Length"))

            if n > MAX_BODY_BYTES:
                self.close_connection = True
                return self._respond(_error(413, "request body too large"))
            body = self.rfile.read(n)

        headers = {k.lower(): v for k, v in self.headers.items()}
        self._respond(self.server.service.handle(self.command, self.path, headers, body))

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _dispatch

    def log_message(self, format: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), format % args)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: Service) -> None:
        super().__init__(address, _Handler)
        self.service = service


def make_server(service: Service, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """A bound, not yet serving, HTTP server (port 0 picks a free port)."""
    return _Server((host, port), service)


WARMUP_STATE = "warmup"
WARMUP_QUESTIONS = {"q": {"type": "noul", "instructions": "Is this a warmup request?"}}


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    models: tuple[ModelName, ...] = ("laya",),
    *,
    default_model: ModelName | None = None,
    api_key: str | None = None,
    max_concurrent: int = 16,
    jev_strict: bool = False,
    warmup: bool = True,
    dtype: DtypeChoice = "auto",
    impl: ImplChoice = "auto",
) -> None:
    import jax

    from deciax.decider import load

    deciders: dict[ModelName, Decides] = {}
    for name in models:
        log.info("loading %s", name)
        deciders[name] = load(name, dtype=dtype, impl=impl)
        if warmup:
            deciders[name].decide(WARMUP_STATE, WARMUP_QUESTIONS)

    service = Service(
        deciders,
        default_model or models[0],
        api_key=api_key,
        max_concurrent=max_concurrent,
        jev_strict=jev_strict,
        device=jax.default_backend(),
    )

    server = make_server(service, host, port)
    log.info("serving %s on http://%s:%d", sorted(deciders), *server.server_address[:2])

    try:
        server.serve_forever()
    finally:
        server.server_close()
