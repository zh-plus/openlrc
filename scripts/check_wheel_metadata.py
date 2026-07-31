"""Validate release-critical OpenLRC Mac wheel metadata."""

from __future__ import annotations

import argparse
import re
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from openlrc import __version__

EXPECTED_NAME = "openlrc-mac"
EXPECTED_PYTHON = {">=3.11", "<3.15"}
EXPECTED_EXTRA = "litellm"
BANNED_DEPENDENCIES = {"deepfilterlib", "deepfilternet", "onnxruntime", "spacy", "torch", "torchaudio"}
EXPECTED_ENTRY_POINTS = {
    "openlrc = openlrc.cli.main:main",
    "openlrc-mac = openlrc.cli.main:main",
    "openlrc-tui = openlrc.tui.app:run",
}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"Invalid Requires-Dist value: {requirement!r}")
    return match.group(0).lower().replace("_", "-").replace(".", "-")


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt"))
        metadata = BytesParser(policy=default).parsebytes(wheel.read(metadata_name))
        entry_points = wheel.read(entry_points_name).decode("utf-8")

    assert metadata["Name"] == EXPECTED_NAME
    assert metadata["Version"] == __version__
    assert {specifier.strip() for specifier in metadata["Requires-Python"].split(",")} == EXPECTED_PYTHON
    assert set(metadata.get_all("Provides-Extra", [])) == {EXPECTED_EXTRA}

    dependency_names = {_requirement_name(value) for value in metadata.get_all("Requires-Dist", [])}
    assert dependency_names.isdisjoint(BANNED_DEPENDENCIES)
    assert EXPECTED_ENTRY_POINTS.issubset(set(entry_points.splitlines()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)


if __name__ == "__main__":
    main()
