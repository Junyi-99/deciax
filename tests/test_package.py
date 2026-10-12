import json
from pathlib import Path

import deciax
from deciax.models import MODELS


def test_version() -> None:
    assert deciax.__version__ == "0.1.1.dev2"


def test_registry_matches_fixture_manifest() -> None:
    manifest = json.loads((Path(__file__).parent / "fixtures" / "manifest.json").read_text())
    for name, entry in manifest["checkpoints"].items():
        assert MODELS[name].repo == entry["repo"]
        assert MODELS[name].revision == entry["revision"]
