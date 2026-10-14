"""Registered checkpoints. One entry until a second model family exists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type ModelName = Literal["laya", "laya-multilingual"]


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    family: Literal["laya"]
    # Hub files the model reads; the repo also holds files deciax never loads.
    files: tuple[str, ...] = (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/*",
        "tokenizer/*",
    )


MODELS: dict[ModelName, Checkpoint] = {
    "laya": Checkpoint(
        "convaiinnovations/laya",
        "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "laya",
    ),
    "laya-multilingual": Checkpoint(
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        "laya",
    ),
}
