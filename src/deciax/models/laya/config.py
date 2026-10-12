"""Checkpoint configuration: `rl_agent_config.json` + `encoder/config.json`."""

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
    act_costs: dict[str, float] = field(default_factory=dict)
    temperature: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature_by_options: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, path: Path) -> LayaConfig:
        raise NotImplementedError
