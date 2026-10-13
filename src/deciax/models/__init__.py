"""Registered checkpoints and their decision-model adapter families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type ModelName = Literal["laya", "laya-multilingual", "clef", "clef-flash"]


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    family: Literal["laya", "clef"]
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

CLEF_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model*.safetensors*", "joint_head_config.json", "joint_head.safetensors")
MODELS.update(
    {
        "clef": Checkpoint("Cloudflare/clef", "0b331204bb13fbd2ca93a64df1956f5b55478ce5", "clef", CLEF_FILES),
        "clef-flash": Checkpoint("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3", "clef", CLEF_FILES),
    }
)
