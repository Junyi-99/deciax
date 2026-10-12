"""input_ids / marker_pos identical to laya 0.4.4 `Agent._encode_state`, token for token."""

import pytest

from deciax.models.laya.config import LayaConfig
from deciax.models.laya.input import encode_state
from deciax.models.laya.tokenizer import LayaTokenizer
from deciax.protocol.types import parse_questions
from tests.conftest import checkpoint_dir, laya_fixture

MODELS = ["laya", "laya-multilingual"]


@pytest.mark.parametrize("model", MODELS)
def test_sequences_match_reference(model: str) -> None:
    path = checkpoint_dir(model)
    tok, cfg = LayaTokenizer(path / "tokenizer"), LayaConfig.from_dir(path)
    for case in laya_fixture(model)["cases"]:
        rows = encode_state(
            tok,
            case["state"],
            parse_questions(case["questions"]),
            cfg.max_len,
            cfg.head_max_len,
        )
        assert [r.input_ids for r in rows] == [r["input_ids"] for r in case["rows"]], (
            case["name"]
        )
        assert [r.marker_pos for r in rows] == [
            r["marker_pos"] for r in case["rows"]
        ], case["name"]
