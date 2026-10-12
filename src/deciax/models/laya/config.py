"""Checkpoint configuration: `rl_agent_config.json` + `encoder/config.json`."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deciax.models.modernbert import ModernBertConfig


@dataclass(frozen=True)
class LayaConfig:
    encoder: ModernBertConfig
    max_len: int = 512
    head_max_len: int = 192
    head_layers: int = 2
    temperature: tuple[float, ...] = (1.0, 1.0, 1.0)
    temperature_by_options: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, path: Path) -> LayaConfig:
        raw = json.loads((path / "rl_agent_config.json").read_text())
        if raw.get("option_layout", "sequential") != "sequential":
            raise NotImplementedError("only the sequential option layout is supported")

        encoder = ModernBertConfig.from_hf(json.loads((path / "encoder" / "config.json").read_text()))

        return cls(
            encoder=encoder,
            max_len=raw.get("max_len", 512),
            head_max_len=raw.get("head_max_len", 192),
            head_layers=raw.get("head_layers", 2),
            temperature=tuple(raw.get("temperature", (1.0, 1.0, 1.0))),
            temperature_by_options=dict(raw.get("temperature_by_options", {})),
            raw=raw,
        )
