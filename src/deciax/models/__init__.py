"""Registered checkpoints. One entry until a second model family exists."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    family: str


MODELS: dict[str, Checkpoint] = {
    "laya": Checkpoint(
        "convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", "laya"
    ),
    "laya-multilingual": Checkpoint(
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        "laya",
    ),
}
