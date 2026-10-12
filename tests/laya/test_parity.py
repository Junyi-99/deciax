"""Outputs vs PyTorch laya 0.4.4 (fixtures from scripts/export_reference.py)."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deciax.decider import Decider
from deciax.models.laya.model import LayaModel
from deciax.protocol.answers import softmax
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import parse_questions
from deciax.runtime.engine import Engine
from tests.conftest import checkpoint_dir, laya_fixture

MODELS = ["laya", "laya-multilingual"]
BF16_PROB_TOLERANCE = (
    0.02  # laya-tpu's acceptance: probability max-abs error vs PyTorch fp32
)
ROUNDING = 2e-4  # answers are rounded to 4 dp; fp32 noise can flip the last digit


def _run(model: LayaModel, case: dict[str, Any]) -> DecisionOutput:
    rows = model.encode(case["state"], parse_questions(case["questions"]))
    return Engine(model).run(rows)


def _close(got: Any, want: Any, path: str = "") -> None:
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for k in want:
            _close(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _close(g, w, f"{path}[{i}]")
    elif isinstance(want, float) and not isinstance(got, str):
        assert abs(got - want) <= ROUNDING, (path, got, want)
    else:
        assert got == want, (path, got, want)


@pytest.fixture(scope="module", params=MODELS)
def cpu_fp32(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[LayaModel, dict[str, Any]]]:
    path = checkpoint_dir(request.param)

    # Load inside the context but yield outside it: a module-scoped fixture that yields
    # inside `default_device` pins every later test in the module to the CPU.
    with jax.default_device(jax.devices("cpu")[0]):
        model = LayaModel(path, jnp.float32, "jax")
    yield model, laya_fixture(request.param)


def test_fp32_logits(cpu_fp32: tuple[LayaModel, dict[str, Any]]) -> None:
    model, fixture = cpu_fp32

    with jax.default_device(jax.devices("cpu")[0]):
        for case in fixture["cases"]:
            out = _run(model, case)
            for i, want in enumerate(case["option_logits"]):
                np.testing.assert_allclose(
                    out.option_logits[i, : len(want)],
                    want,
                    atol=1e-3,
                    err_msg=case["name"],
                )

            # act logits run to ~4e3; compare relatively
            np.testing.assert_allclose(
                out.act_logits,
                case["act_logits"],
                rtol=1e-5,
                atol=1e-3,
                err_msg=case["name"],
            )


def test_fp32_responses(cpu_fp32: tuple[LayaModel, dict[str, Any]]) -> None:
    model, fixture = cpu_fp32
    decider = Decider(model)

    with jax.default_device(jax.devices("cpu")[0]):
        for case in fixture["cases"]:
            _close(
                decider.decide(case["state"], case["questions"]),
                case["response"],
                case["name"],
            )


@pytest.mark.tpu
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("impl", ["jax", "pallas"])
def test_bf16_tpu(model_name: str, impl: str) -> None:
    model = LayaModel(checkpoint_dir(model_name), jnp.bfloat16, impl)
    assert next(iter(model.params.values())).devices().pop().platform == "tpu"

    for case in laya_fixture(model_name)["cases"]:
        out = _run(model, case)

        for i, want in enumerate(case["option_logits"]):
            got = out.option_logits[i, : len(want)].astype(np.float64)
            ref = np.asarray(want, np.float64)
            assert np.abs(softmax(got) - softmax(ref)).max() <= BF16_PROB_TOLERANCE, (
                case["name"]
            )
            assert got.argmax() == ref.argmax(), case["name"]

        act = np.asarray(case["act_logits"], np.float64)
        for i in range(len(act)):
            diff = np.abs(
                softmax(out.act_logits[i].astype(np.float64)) - softmax(act[i])
            ).max()
            assert diff <= BF16_PROB_TOLERANCE, case["name"]
