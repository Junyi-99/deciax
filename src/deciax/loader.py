"""Fetch pinned checkpoints from the Hugging Face Hub."""

from pathlib import Path

from huggingface_hub import snapshot_download

FILES = ["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]


def fetch(repo: str, revision: str, local_files_only: bool = False) -> Path:
    """Local snapshot of the files a checkpoint needs (downloaded if missing)."""
    return Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=FILES,
            local_files_only=local_files_only,
        )
    )
