"""`tokenizers` ids identical to laya's AutoTokenizer."""

import pytest

from deciax.models import ModelName
from deciax.models.laya.tokenizer import LayaTokenizer
from tests.conftest import checkpoint_dir, laya_fixture

MODELS: list[ModelName] = ["laya", "laya-multilingual"]


@pytest.mark.parametrize("model", MODELS)
def test_ids_match_autotokenizer(model: ModelName) -> None:
    tok = LayaTokenizer(checkpoint_dir(model) / "tokenizer")
    for case in laya_fixture(model)["tokenizer"]:
        assert tok.encode(case["text"]) == case["ids"], case["text"]
        assert tok.encode(case["text"], 48) == case["ids_max48"], case["text"]
