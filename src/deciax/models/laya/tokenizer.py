"""Laya tokenization with the `tokenizers` library (no transformers dependency).

Must reproduce laya's AutoTokenizer ids exactly, including the replacement of
[MASK] inside state text (tests/laya/test_tokenizer.py).
"""

from pathlib import Path


class LayaTokenizer:
    def __init__(self, tokenizer_dir: Path) -> None:
        raise NotImplementedError

    def encode_text(self, text: str, max_len: int | None = None) -> list[int]:
        raise NotImplementedError
