from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import openlrc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANNED_METADATA_TEXT = ("deepfilter", "onnxruntime", "spacy", "torch", "torchaudio")


def test_runtime_version_matches_installed_distribution() -> None:
    assert openlrc.__version__ == "0.4.2"
    assert importlib.metadata.version("openlrc-mac") == openlrc.__version__


def test_project_metadata_has_only_supported_python_and_extras() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata_text = repr(project).lower()

    assert project["version"] == openlrc.__version__
    assert project["requires-python"] == ">=3.11, <3.15"
    assert set(project["optional-dependencies"]) == {"litellm"}
    assert all(name not in metadata_text for name in BANNED_METADATA_TEXT)
