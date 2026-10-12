"""Public entry point: encode a request, run the engine, decode typed answers."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from deciax.models import MODELS
from deciax.models.laya.model import LayaModel
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import State, parse_questions
from deciax.runtime.engine import Engine


class Decider:
    """One loaded checkpoint bound to a runtime engine."""

    def __init__(self, model: LayaModel) -> None:
        self.model = model
        self.engine = Engine(model)

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
        return self.decide_batch(
            [state], questions, max_len=max_len, head_max_len=head_max_len
        )[0]

    def decide_batch(
        self,
        states: Sequence[State],
        questions: Mapping[Any, Mapping[str, Any]],
        *,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        """Answer the same questions about many states, sharing forward passes."""
        parsed = parse_questions(questions)
        if not parsed:
            empty = {"input_tokens": 0, "output_tokens": 0}
            return [
                {"model": "laya-rl-agent", "answers": {}, "usage": dict(empty)}
                for _ in states
            ]

        per_state = [
            self.model.encode(s, parsed, max_len, head_max_len) for s in states
        ]
        out = self.engine.run([r for rows in per_state for r in rows])

        results, n = [], len(parsed)
        for i, rows in enumerate(per_state):
            part = slice(i * n, (i + 1) * n)
            sub = DecisionOutput(
                out.option_logits[part],
                out.option_mask[part],
                out.act_logits[part],
                out.input_tokens[part],
            )
            results.append(self.model.decode(rows, sub, parsed))

        return results


def load(
    model: str = "laya",
    *,
    revision: str | None = None,
    dtype: str = "auto",
    impl: str = "auto",
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

    return Decider(LayaModel(path, jdtype, impl))
