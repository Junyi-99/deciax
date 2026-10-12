import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `tpu` tests unless JAX's default backend is a TPU."""
    if not any("tpu" in item.keywords for item in items):
        return
    import jax

    if jax.default_backend() == "tpu":
        return

    skip = pytest.mark.skip(reason="needs a TPU")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip)


def laya_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "laya" / f"{name}.json").read_text())  # type: ignore[no-any-return]


def checkpoint_dir(name: str) -> Path:
    """The cached checkpoint, or skip: tests never download weights."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from deciax.models import MODELS

    ckpt = MODELS[name]

    try:
        return Path(
            snapshot_download(
                ckpt.repo,
                revision=ckpt.revision,
                allow_patterns=list(ckpt.files),
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"{ckpt.repo}@{ckpt.revision[:8]} is not in the local HF cache")
