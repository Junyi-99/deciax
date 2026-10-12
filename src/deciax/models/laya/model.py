"""Laya = ModernBERT encoder + decision heads, implementing `DecisionModel`."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from deciax.models import modernbert
from deciax.models.base import EncodedRow
from deciax.models.laya.calibration import Calibration
from deciax.models.laya.config import LayaConfig
from deciax.models.laya.heads import decision_heads
from deciax.models.laya.input import encode_state
from deciax.models.laya.tokenizer import LayaTokenizer
from deciax.models.laya.weights import load_params
from deciax.protocol.answers import softmax, to_answer
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import Question, State

MODEL_NAME = "laya-rl-agent"


class LayaModel:
    def __init__(self, path: Path, dtype: jnp.dtype = jnp.bfloat16, impl: str = "auto") -> None:
        self.config = LayaConfig.from_dir(path)
        self.tokenizer = LayaTokenizer(path / "tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.calibration = Calibration(self.config.temperature, self.config.temperature_by_options)

        self.dtype = dtype
        self.params = load_params(path / "model.safetensors", dtype)
        self._forward = jax.jit(functools.partial(self._apply, impl=impl))

    def _apply(
        self,
        params: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array,
        marker_pos: jax.Array,
        marker_mask: jax.Array,
        qtype: jax.Array,
        impl: str,
    ) -> tuple[jax.Array, jax.Array]:
        h = modernbert.encode(params, self.config.encoder, input_ids, attention_mask, self.dtype, impl)

        return decision_heads(
            params,
            h,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
            self.config.head_layers,
            self.dtype,
            impl,
        )

    def encode(
        self,
        state: State,
        questions: Mapping[Any, Question],
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> list[EncodedRow]:
        return encode_state(
            self.tokenizer,
            state,
            questions,
            self.config.max_len if max_len is None else max_len,
            self.config.head_max_len if head_max_len is None else head_max_len,
        )

    def forward(self, arrays: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        return self._forward(  # type: ignore[no-any-return]
            self.params,
            arrays["input_ids"],
            arrays["attention_mask"],
            arrays["marker_pos"],
            arrays["marker_mask"],
            arrays["qtype"],
        )

    def decode(
        self,
        rows: list[EncodedRow],
        output: DecisionOutput,
        questions: Mapping[Any, Question],
    ) -> dict[str, Any]:
        """Laya's `system_one` response for one state's rows."""
        answers = {}
        for i, (qid, q) in enumerate(questions.items()):
            k = len(rows[i].marker_pos)
            t = self.calibration.temperature_for(q.type, k)
            probs = softmax(output.option_logits[i, :k].astype(np.float64) / t)
            act = softmax(output.act_logits[i].astype(np.float64))
            answers[qid] = to_answer(q, probs, float(act[0]))

        stats = [r.state_stats for r in rows]
        dropped = max(s["state_tokens_dropped"] for s in stats)
        usage: dict[str, Any] = {
            "input_tokens": int(output.input_tokens.sum()),
            "output_tokens": 0,
            "state_tokens": stats[0]["state_tokens"],
            "state_tokens_dropped": dropped,
            "truncated": dropped > 0,
            "truncated_questions": [qid for qid, s in zip(questions, stats, strict=True) if s["truncated"]],
        }

        collapsed = {
            qid: {
                "total": r.option_stats["options"],
                "distinct": r.option_stats["options_distinct"],
                "tokens_per_option": r.option_stats["tokens_per_option"],
            }
            for qid, r in zip(questions, rows, strict=True)
            if r.option_stats["options_distinct"] < r.option_stats["options"]
        }
        if collapsed:
            usage["options"] = collapsed

        return {"model": MODEL_NAME, "answers": answers, "usage": usage}
