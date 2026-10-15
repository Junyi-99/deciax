import json

import deciax
from deciax.models import MODELS
from tests.conftest import FIXTURES


def test_version() -> None:
    assert deciax.__version__ == "0.1.2"


def test_registry_matches_fixture_manifest() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, entry in manifest["checkpoints"].items():
        assert MODELS[name].repo == entry["repo"]
        assert MODELS[name].revision == entry["revision"]
