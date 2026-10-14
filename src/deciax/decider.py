"""Public entry point: encode a request, run the engine, decode typed answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from deciax.models import MODELS, ModelName
from deciax.models.base import RequestModel
from deciax.models.laya.model import LayaModel
from deciax.ops.backend import ImplChoice
from deciax.protocol.types import State

type DtypeChoice = Literal["auto", "bf16", "fp32"]


class Decider:
    """One loaded checkpoint bound to a runtime engine."""

    def __init__(self, model: RequestModel) -> None:
        self.model = model
        self.engine = model.build_engine()

    def decide(
        self,
        state: State,
        questions: Mapping[Any, Mapping[str, Any]],
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> dict[str, Any]:
        """Answer every question about one state (Laya `system_one` response).

        `max_len` / `head_max_len` override the checkpoint's token budgets.
        """
        return self.decide_batch([state], questions, max_len=max_len, head_max_len=head_max_len)[0]

    def decide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[Any, Mapping[str, Any]],
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        """Answer the same questions about many states, sharing forward passes."""
        return self.engine.decide_batch(states, questions, max_len=max_len, head_max_len=head_max_len)


def load(
    model: ModelName = "laya",
    *,
    revision: str | None = None,
    dtype: DtypeChoice = "auto",
    impl: ImplChoice = "auto",
) -> Decider:
    """Load a registered checkpoint (see `deciax.models.MODELS`).

    `dtype="auto"` is bf16 on TPU and fp32 elsewhere.
    """
    ckpt = MODELS[model]
    path = Path(
        snapshot_download(
            ckpt.repo,
            revision=revision or ckpt.revision,
            allow_patterns=list(ckpt.files),
        )
    )

    if dtype == "auto":
        dtype = "bf16" if jax.default_backend() == "tpu" else "fp32"
    jdtype = {"bf16": jnp.bfloat16, "fp32": jnp.float32}[dtype]

    if ckpt.family == "laya":
        return Decider(LayaModel(path, jdtype, impl))

    if ckpt.family == "clef":
        if impl not in ("auto", "jax"):
            raise ValueError("Clef currently supports auto/jax kernels")
        from deciax.models.clef.model import ClefModel

        return Decider(ClefModel(path, jdtype, model))

    raise ValueError(f"unsupported model family: {ckpt.family}")
