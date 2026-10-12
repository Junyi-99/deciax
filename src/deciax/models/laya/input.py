"""Sequence construction.

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

Port of laya `serialize_state`, `render_options`, `build_head`, `build_sequence` and
`Agent._encode_state` (laya/common.py, laya/agent.py, laya 0.4.4, Apache-2.0).
Sequential option layout only.
"""

import json
from collections.abc import Mapping
from typing import Any

from deciax.models.base import EncodedRow
from deciax.models.laya.tokenizer import LayaTokenizer
from deciax.protocol.types import (
    QTYPES,
    Question,
    State,
    render_criterion,
    resolve_noul_labels,
)

OPTION_MAX_TOKENS = 48


def serialize_state(state: State) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_options(q: Question) -> list[str]:
    """Option texts in label-index order; noul is always [false, true]."""
    if q.type == "choice":
        return [str(k) if v is None or v == "" else f"{k}: {render_criterion(v)}" for k, v in q.criteria.items()]

    if q.type == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(q.criteria)]

    crit = q.criteria or {}
    false_label, true_label = resolve_noul_labels(q.labels)
    false_crit, true_crit = crit.get("false"), crit.get("true")

    return [
        false_label + ": " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_head(tok: LayaTokenizer, q: Question, head_max_len: int) -> tuple[list[int], list[int], dict[str, Any]]:
    """(head ids, marker positions, option stats) for the question half."""
    mask = tok.mask_token
    opts = render_options(q)
    order = q.option_order if q.option_order is not None else range(len(opts))
    head_ids = tok.encode(f"{q.type} question: {q.instructions.replace(mask, ' ')}")
    opt_ids = [[tok.mask_token_id] + tok.encode(" " + opts[i].replace(mask, " "), OPTION_MAX_TOKENS) for i in order]

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per_option] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]

    ids = [tok.cls_token_id, *head_ids, tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)

    stats = {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }
    return ids, markers, stats


def build_sequence(
    tok: LayaTokenizer,
    q: Question,
    state_ids: list[int],
    max_len: int,
    head_max_len: int,
    truncate_left: bool = False,
) -> EncodedRow:
    ids, markers, option_stats = build_head(tok, q, head_max_len)

    room = max(0, max_len - len(ids) - 1)
    st = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    ids = [*ids, *st, tok.sep_token_id][:max_len]

    return EncodedRow(
        input_ids=ids,
        marker_pos=[m for m in markers if m < max_len],
        qtype=QTYPES[q.type],
        state_stats={
            "state_tokens": len(state_ids),
            "state_tokens_used": len(st),
            "state_tokens_dropped": len(state_ids) - len(st),
            "truncated": len(st) < len(state_ids),
        },
        option_stats=option_stats,
    )


def encode_state(
    tok: LayaTokenizer,
    state: State,
    questions: Mapping[Any, Question],
    max_len: int,
    head_max_len: int,
) -> list[EncodedRow]:
    """One row per question; the state is tokenized once and shared. A list state
    (a conversation, newest last) is truncated from the left."""
    state_ids = tok.encode(serialize_state(state).replace(tok.mask_token, " "))

    rows = []
    for qid, q in questions.items():
        row = build_sequence(tok, q, state_ids, max_len, head_max_len, isinstance(state, list))
        if len(row.marker_pos) != q.option_count:
            raise ValueError(
                f"question {qid!r}: only {len(row.marker_pos)} of its {q.option_count} option "
                f"markers fit in max_len={max_len} with head_max_len={head_max_len} spent on "
                "the question; lower head_max_len, raise max_len, or use fewer options"
            )
        rows.append(row)

    return rows
