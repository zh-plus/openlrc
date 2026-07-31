from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from openlrc.application.resources import _whisper_cli_version_status, _whisper_metal_status


def test_whisper_cli_version_probe_reports_version() -> None:
    completed = subprocess.CompletedProcess(
        args=["/tmp/whisper-cli", "--version"], returncode=0, stdout="whisper.cpp version: 1.9.1\n", stderr=""
    )

    with patch("openlrc.application.resources.subprocess.run", return_value=completed) as run:
        status = _whisper_cli_version_status("/tmp/whisper-cli")

    assert status.available is True
    assert status.detail == "whisper.cpp version: 1.9.1"
    run.assert_called_once_with(
        ["/tmp/whisper-cli", "--version"], capture_output=True, text=True, timeout=5, check=False
    )


def test_whisper_cli_version_probe_failure_is_actionable() -> None:
    completed = subprocess.CompletedProcess(
        args=["/tmp/whisper-cli", "--version"], returncode=2, stdout="", stderr="unsupported option"
    )

    with patch("openlrc.application.resources.subprocess.run", return_value=completed):
        status = _whisper_cli_version_status("/tmp/whisper-cli")

    assert status.available is False
    assert "code 2" in status.detail
    assert "unsupported option" in status.detail
    assert "setup whisper" in status.hint


def test_vendor_metal_build_reads_cmake_cache_without_running_inference(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "whisper.cpp"
    cli_path = vendor_dir / "build" / "bin" / "whisper-cli"
    cache_path = vendor_dir / "build" / "CMakeCache.txt"
    cli_path.parent.mkdir(parents=True)
    cli_path.touch()
    cache_path.write_text("GGML_METAL:BOOL=ON\nGGML_METAL_EMBED_LIBRARY:BOOL=ON\n", encoding="utf-8")

    with patch("openlrc.application.resources.whisper_vendor_dir", return_value=vendor_dir):
        status = _whisper_metal_status(str(cli_path))

    assert status.available is True
    assert "Metal enabled at build time" in status.detail
    assert "embedded library enabled" in status.detail
    assert "runtime not probed" in status.detail
    assert "--no-whisper-gpu --no-whisper-flash-attn" in status.hint


def test_metal_disabled_is_informational_because_cpu_remains_valid(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "whisper.cpp"
    cli_path = vendor_dir / "build" / "bin" / "whisper-cli"
    cache_path = vendor_dir / "build" / "CMakeCache.txt"
    cli_path.parent.mkdir(parents=True)
    cli_path.touch()
    cache_path.write_text("GGML_METAL:BOOL=OFF\n", encoding="utf-8")

    with patch("openlrc.application.resources.whisper_vendor_dir", return_value=vendor_dir):
        status = _whisper_metal_status(str(cli_path))

    assert status.available is True
    assert "Metal disabled at build time" in status.detail
    assert "CPU remains usable" in status.detail


def test_non_vendor_whisper_binary_reports_unknown_build_capability(tmp_path: Path) -> None:
    vendor_dir = tmp_path / "vendor"
    external_cli = tmp_path / "bundle" / "whisper-cli"
    external_cli.parent.mkdir()
    external_cli.touch()

    with patch("openlrc.application.resources.whisper_vendor_dir", return_value=vendor_dir):
        status = _whisper_metal_status(str(external_cli))

    assert status.available is True
    assert "unknown" in status.detail
    assert "non-vendor" in status.detail
    assert "runtime not probed" in status.detail


def test_missing_whisper_binary_makes_version_and_metal_unavailable() -> None:
    version = _whisper_cli_version_status(None)
    metal = _whisper_metal_status(None)

    assert version.available is False
    assert metal.available is False
    assert "whisper-cli is missing" in version.detail
    assert "whisper-cli is missing" in metal.detail
