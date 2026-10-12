"""Fetch pinned checkpoints from the Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import snapshot_download


def fetch(repo: str, revision: str) -> Path:
    """Local snapshot directory of `repo` at `revision` (downloaded if missing)."""
    return Path(snapshot_download(repo, revision=revision))
