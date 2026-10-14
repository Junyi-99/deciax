"""Text encoding ported from Cloudflare/clef joint_schema_model.py (Apache-2.0).

Reference revision: 0b331204bb13fbd2ca93a64df1956f5b55478ce5.
Question/option spans and serialization are part of the trained model contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from deciax.models import ModelName
from deciax.protocol.answers import softmax
from deciax.protocol.systemone import MAX_QUESTIONS, MAX_TOTAL_OPTIONS
from deciax.runtime.buckets import ROW_BUCKETS, bucket

SYSTEM_PROMPT = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."
CLEF_QTYPES = {"noul": 0, "choice": 1, "score": 2}  # the trained type-embedding order
SEQUENCE_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
MAX_TOKENS = SEQUENCE_BUCKETS[-1]
SCHEMA_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def options(question: Mapping[str, Any]) -> list[tuple[str, Any]]:
    if question["type"] == "noul":
        criteria = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
        criteria.update(question.get("criteria") or {})
        return [(k, criteria[k]) for k in ("true", "false")]

    if question["type"] == "choice":
        return sorted((str(k), v) for k, v in question["criteria"].items())

    return [(str(i), v) for i, v in enumerate(question["criteria"])]


@dataclass(frozen=True)
class Field:
    id: str
    kind: int
    span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class Record:
    ids: tuple[int, ...]
    fields: tuple[Field, ...]


def validate_questions(questions: Mapping[Any, Any]) -> None:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Clef requires at least one question")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"Clef supports at most {MAX_QUESTIONS} questions per request")

    total = 0
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ValueError("Clef question ids must be nonempty strings")
        if not isinstance(q, Mapping) or not isinstance(q.get("type"), str) or q.get("type") not in CLEF_QTYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")

        if any(k in q for k in ("labels", "option_order")):
            raise ValueError(f"{qid}: Clef does not support labels or option_order")

        criteria = q.get("criteria")
        if q["type"] == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise ValueError(f"{qid}: choice criteria must be a nonempty object")
        if q["type"] == "score" and (not isinstance(criteria, list) or not criteria):
            raise ValueError(f"{qid}: score criteria must be a nonempty list")
        if q["type"] == "noul" and criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
            raise ValueError(f"{qid}: noul criteria must use true/false keys")

        total += len(options(q))

    if total > MAX_TOTAL_OPTIONS:
        raise ValueError(f"Clef supports at most {MAX_TOTAL_OPTIONS} total options per request")


def encode(tokenizer: Tokenizer, state: Any, questions: Mapping[str, Any], max_length: int = MAX_TOKENS) -> Record:
    validate_questions(questions)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_TOKENS:
        raise ValueError(f"Clef max_len must be between 1 and {MAX_TOKENS}")

    def tokens(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)

    schema = tokens("\n\nSCHEMA FIELDS:\n")
    fields = []
    for i, (qid, q) in enumerate(questions.items()):
        schema += tokens(f"\nFIELD {i + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        start = len(schema)
        instruction = q.get("instructions")
        schema += tokens(qid if instruction in (None, "") else render(instruction))
        span = (start, len(schema))

        schema += tokens("\nALLOWED OPTIONS:\n")
        spans, ids = [], []
        for j, (oid, description) in enumerate(options(q)):
            schema += tokens(f"OPTION {j + 1}: ")
            start = len(schema)
            semantics = {"option_id": oid}
            if description is not None:
                semantics["description"] = description
            schema += tokens(render(semantics))
            spans.append((start, len(schema)))
            ids.append(oid)
            schema += tokens("\n")

        schema += tokens("END FIELD\n")
        fields.append(Field(qid, CLEF_QTYPES[q["type"]], span, tuple(spans), tuple(ids)))

    prefix = tokens(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = tokens("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")

    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")

    state_ids = tokens(render(state))[: max_length - fixed]
    offset = len(prefix) + len(state_ids)
    shifted = tuple(
        Field(f.id, f.kind, (f.span[0] + offset, f.span[1] + offset), tuple((s + offset, e + offset) for s, e in f.option_spans), f.option_ids) for f in fields
    )

    return Record(tuple(prefix + state_ids + schema + suffix), shifted)


def collate(records: Sequence[Record], pad_id: int) -> dict[str, NDArray[Any]]:
    """Pad complete records; options are flat with explicit field ownership."""
    b = bucket(len(records), ROW_BUCKETS)
    s = bucket(max(len(r.ids) for r in records), SEQUENCE_BUCKETS)
    q = bucket(max(len(r.fields) for r in records), SCHEMA_BUCKETS)
    o = bucket(max(sum(len(f.option_ids) for f in r.fields) for r in records), SCHEMA_BUCKETS)
    lex = [tuple(t for f in r.fields for start, end in f.option_spans for t in r.ids[start:end]) for r in records]
    lexical_length = bucket(max(map(len, lex)), SEQUENCE_BUCKETS)

    arrays = {
        "lexical_ids": np.full((b, lexical_length), pad_id, np.int32),
        "lexical_spans": np.zeros((b, o, 2), np.int32),
        "input_ids": np.full((b, s), pad_id, np.int32),
        "mask": np.zeros((b, s), bool),
        "question_spans": np.zeros((b, q, 2), np.int32),
        "question_mask": np.zeros((b, q), bool),
        "types": np.zeros((b, q), np.int32),
        "option_spans": np.zeros((b, o, 2), np.int32),
        "option_mask": np.zeros((b, o), bool),
        "option_field": np.zeros((b, o), np.int32),
    }

    for i, record in enumerate(records):
        arrays["input_ids"][i, : len(record.ids)] = record.ids
        arrays["mask"][i, : len(record.ids)] = True
        arrays["lexical_ids"][i, : len(lex[i])] = lex[i]

        lexical_offset = 0
        j = 0
        for k, f in enumerate(record.fields):
            arrays["question_spans"][i, k] = f.span
            arrays["question_mask"][i, k] = True
            arrays["types"][i, k] = f.kind
            n = len(f.option_ids)
            arrays["option_spans"][i, j : j + n] = f.option_spans
            arrays["option_mask"][i, j : j + n] = True
            arrays["option_field"][i, j : j + n] = k
            for t, (start, end) in enumerate(f.option_spans):
                arrays["lexical_spans"][i, j + t] = (lexical_offset, lexical_offset + end - start)
                lexical_offset += end - start
            j += n

    return arrays


def decode(record: Record, logits: NDArray[Any], questions: Mapping[str, Any], model_name: ModelName) -> dict[str, Any]:
    answers, offset = {}, 0
    for f in record.fields:
        n = len(f.option_ids)
        values = logits[offset : offset + n].astype(np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite Clef option logits")
        probs = dict(zip(f.option_ids, softmax(values).tolist(), strict=True))

        q = questions[f.id]
        if q["type"] == "noul":
            answer: dict[str, Any] = {"type": "noul", "noul": round(probs["true"], 4)}
        elif q["type"] == "choice":
            keys = [str(k) for k in q["criteria"]]
            chosen = max(keys, key=probs.__getitem__)
            answer = {"type": "choice", "choice": chosen, "confidence": round(probs[chosen], 4), "probabilities": {k: round(probs[k], 4) for k in keys}}
        else:
            keys = list(f.option_ids)
            answer = {
                "type": "score",
                "score": round(sum(i * probs[k] for i, k in enumerate(keys)), 4),
                "confidence": round(max(probs.values()), 4),
                "legend": dict(zip(keys, q["criteria"], strict=True)),
                "probabilities": {k: round(probs[k], 4) for k in keys},
            }

        answers[f.id] = answer
        offset += n

    return {"model": model_name, "answers": answers, "usage": {"input_tokens": len(record.ids), "output_tokens": 0}}
