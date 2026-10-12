"""Laya tokenization with the `tokenizers` library (no transformers dependency)."""

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


def _token(value: Any) -> str:
    return str(value["content"] if isinstance(value, dict) else value)


class LayaTokenizer:
    def __init__(self, tokenizer_dir: Path) -> None:
        self._tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self._tok.no_truncation()
        self._tok.no_padding()
        cfg = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
        self.mask_token = _token(cfg["mask_token"])
        self.mask_token_id = self._id(self.mask_token)
        self.cls_token_id = self._id(_token(cfg["cls_token"]))
        self.sep_token_id = self._id(_token(cfg["sep_token"]))
        self.pad_token_id = self._id(_token(cfg["pad_token"]))

    def _id(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(f"special token {token!r} is not in the vocabulary")
        return int(token_id)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Ids without special tokens; `max_length` keeps the first tokens (laya's
        `truncation=True, max_length=...` on a single sequence)."""
        ids: list[int] = self._tok.encode(text, add_special_tokens=False).ids
        return ids if max_length is None else ids[:max_length]
