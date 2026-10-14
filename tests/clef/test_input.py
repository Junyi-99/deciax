from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from deciax.models.clef.input import collate, decode, encode, validate_questions


@pytest.fixture
def tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def test_record_spans_options_and_compact_lexical_tokens(tokenizer: Tokenizer) -> None:
    questions = {
        "sentiment": {"type": "choice", "criteria": {"z": "world", "a": "hello"}},
        "yes": {"type": "noul"},
        "score": {"type": "score", "criteria": ["hello", "world", "hello world"]},
    }

    records = [encode(tokenizer, s, questions, 512) for s in ("hello", "hello world " * 20)]
    arrays = collate(records, 0)

    assert records[0].fields[0].option_ids == ("a", "z")
    assert records[0].fields[1].option_ids == ("true", "false")

    for i, record in enumerate(records):
        j = 0
        for owner, field in enumerate(record.fields):
            assert field.span[0] < field.span[1] < len(record.ids)
            for start, end in field.option_spans:
                ls, le = arrays["lexical_spans"][i, j]
                np.testing.assert_array_equal(record.ids[start:end], arrays["lexical_ids"][i, ls:le])
                assert arrays["option_field"][i, j] == owner
                j += 1

        assert arrays["option_mask"][i].sum() == 7

        result = decode(record, np.zeros(8), questions, "clef")
        assert result["answers"]["yes"]["noul"] == 0.5
        assert result["answers"]["score"]["score"] == 1.0
        assert result["usage"]["input_tokens"] == len(record.ids)


def test_truncation_keeps_schema_complete(tokenizer: Tokenizer) -> None:
    q = {"x": {"type": "choice", "criteria": {"a": "hello", "b": "world"}}}
    fixed = encode(tokenizer, "", q, 512)
    limit = len(fixed.ids) + 5

    actual = encode(tokenizer, "hello world " * 500, q, limit)

    assert len(actual.ids) == limit
    for a, b in zip(actual.fields, fixed.fields, strict=True):
        assert a.span == (b.span[0] + 5, b.span[1] + 5)

    with pytest.raises(ValueError, match="schema requires"):
        encode(tokenizer, "", q, len(fixed.ids) - 1)


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": {"type": []}},
        {"q": {"type": "score", "criteria": []}},
        {"q": {"type": "noul", "criteria": {"yes": "yes"}}},
        {"q": {"type": "choice", "criteria": {"a": "a"}, "option_order": ["a"]}},
    ],
)
def test_invalid_schema(questions: Any) -> None:
    with pytest.raises(ValueError):
        validate_questions(questions)


def test_decode_rejects_nonfinite_logits(tokenizer: Tokenizer) -> None:
    questions = {"yes": {"type": "noul"}}
    record = encode(tokenizer, "hello", questions, 512)

    with pytest.raises(FloatingPointError, match="non-finite"):
        decode(record, np.array([np.nan, 0.0]), questions, "clef")
