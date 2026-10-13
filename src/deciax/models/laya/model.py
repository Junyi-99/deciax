"""Laya = ModernBERT encoder + decision heads, implementing `DecisionModel`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from deciax.models import modernbert
from deciax.models.base import EncodedRow, RequestEngine
from deciax.models.laya import heads
from deciax.models.laya.calibration import Calibration
from deciax.models.laya.config import LayaConfig
from deciax.models.laya.input import encode_state
from deciax.models.laya.tokenizer import LayaTokenizer
from deciax.models.laya.weights import load_params
from deciax.ops.backend import ImplChoice, resolve
from deciax.protocol.answers import softmax, to_answer
from deciax.protocol.output import DecisionOutput
from deciax.protocol.types import Question, State

MODEL_NAME = "laya-rl-agent"


class LayaModel:
    max_token_budget = 1024

    def build_engine(self) -> RequestEngine:
        from deciax.runtime.engine import Engine

        return Engine(self)

    def __init__(self, path: Path, dtype: jnp.dtype = jnp.bfloat16, impl: ImplChoice = "auto") -> None:
        self.config = LayaConfig.from_dir(path)
        self.tokenizer = LayaTokenizer(path / "tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.calibration = Calibration(self.config.temperature, self.config.temperature_by_options)

        self.dtype = dtype
        self.params = load_params(path / "model.safetensors", dtype)
        self.impl = resolve(impl)
        self.fused: dict[str, jax.Array] = {}  # weight layouts for the fused kernels
        if self.impl == "pallas":
            self.fused = {**modernbert.prepare_fused(self.params, self.config.encoder), **heads.prepare_fused(self.params, self.config.head_layers)}
        self._forward = jax.jit(self._apply)

    def _apply(
        self,
        params: dict[str, jax.Array],
        fused: dict[str, jax.Array],
        input_ids: jax.Array,
        attention_mask: jax.Array,
        marker_pos: jax.Array,
        marker_mask: jax.Array,
        qtype: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if self.impl == "pallas":
            h = modernbert.encode_fused(params, fused, self.config.encoder, input_ids, attention_mask)
            return heads.decision_heads_fused(params, fused, h, attention_mask, marker_pos, marker_mask, qtype, self.config.head_layers)

        h = modernbert.encode(params, self.config.encoder, input_ids, attention_mask, self.dtype, self.impl)

        return heads.decision_heads(
            params,
            h,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
            self.config.head_layers,
            self.dtype,
            self.impl,
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
            self.fused,
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
