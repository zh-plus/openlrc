from __future__ import annotations

from pathlib import Path

import pytest

from openlrc.setup import whisper_cpp


def _configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    vendor_dir = repo_root / "vendor" / "whisper.cpp"
    vendor_dir.mkdir(parents=True)
    monkeypatch.setattr(whisper_cpp, "REPO_ROOT", repo_root)
    monkeypatch.setattr(whisper_cpp, "VENDOR_DIR", vendor_dir)
    return repo_root, vendor_dir


def test_ensure_submodule_skips_existing_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _repo_root, vendor_dir = _configure_paths(monkeypatch, tmp_path)
    (vendor_dir / "CMakeLists.txt").touch()
    calls: list[tuple[list[str], Path]] = []

    whisper_cpp.ensure_submodule(runner=lambda command, cwd: calls.append((command, cwd)))

    assert calls == []


def test_ensure_submodule_invokes_git_and_checks_postcondition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root, vendor_dir = _configure_paths(monkeypatch, tmp_path)
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], cwd: Path) -> None:
        calls.append((command, cwd))
        (vendor_dir / "CMakeLists.txt").touch()

    whisper_cpp.ensure_submodule(runner=runner)

    assert calls == [(["git", "submodule", "update", "--init", "--recursive", "vendor/whisper.cpp"], repo_root)]


def test_ensure_submodule_fails_when_checkout_is_still_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_paths(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="missing after submodule initialization"):
        whisper_cpp.ensure_submodule(runner=lambda _command, _cwd: None)


def test_build_whisper_cpp_uses_cmake_and_requires_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root, vendor_dir = _configure_paths(monkeypatch, tmp_path)
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(whisper_cpp.os, "cpu_count", lambda: 6)

    def runner(command: list[str], cwd: Path) -> None:
        calls.append((command, cwd))
        if command[:2] == ["cmake", "--build"]:
            cli_path = vendor_dir / "build" / "bin" / "whisper-cli"
            cli_path.parent.mkdir(parents=True)
            cli_path.touch()

    cli_path = whisper_cpp.build_whisper_cpp(runner=runner)

    build_dir = vendor_dir / "build"
    assert cli_path == build_dir / "bin" / "whisper-cli"
    assert calls == [
        (["cmake", "-S", ".", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], vendor_dir),
        (["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", "6"], repo_root),
    ]


def test_build_whisper_cpp_rejects_missing_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_paths(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="whisper-cli was not found"):
        whisper_cpp.build_whisper_cpp(runner=lambda _command, _cwd: None)


def test_download_models_invokes_both_scripts_and_requires_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _repo_root, vendor_dir = _configure_paths(monkeypatch, tmp_path)
    model_dir = tmp_path / "models"
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], cwd: Path) -> None:
        calls.append((command, cwd))
        if any(item.endswith("download-ggml-model.sh") for item in command):
            (model_dir / "ggml-small.bin").touch()
        if any(item.endswith("download-vad-model.sh") for item in command):
            (model_dir / "ggml-silero-v6.2.0.bin").touch()

    whisper_model, vad_model = whisper_cpp.download_models(model_dir, "small", "silero-v6.2.0", runner=runner)

    assert whisper_model == model_dir / "ggml-small.bin"
    assert vad_model == model_dir / "ggml-silero-v6.2.0.bin"
    assert calls == [
        (["sh", "models/download-ggml-model.sh", "small", str(model_dir)], vendor_dir),
        (["sh", "models/download-vad-model.sh", "silero-v6.2.0", str(model_dir)], vendor_dir),
    ]


def test_download_models_reports_missing_expected_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_paths(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match=r"ggml-base\.bin.*ggml-silero-v6\.2\.0\.bin"):
        whisper_cpp.download_models(tmp_path / "models", "base", "silero-v6.2.0", runner=lambda _command, _cwd: None)


def test_setup_skip_flags_return_no_artifacts_without_build_or_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _repo_root, vendor_dir = _configure_paths(monkeypatch, tmp_path)
    (vendor_dir / "CMakeLists.txt").touch()
    calls: list[tuple[list[str], Path]] = []

    result = whisper_cpp.setup_whisper_cpp(
        skip_build=True, skip_models=True, runner=lambda command, cwd: calls.append((command, cwd))
    )

    assert result.cli_path is None
    assert result.whisper_model is None
    assert result.vad_model is None
    assert calls == []
