#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Prepare the vendored whisper.cpp CLI and default model files."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor" / "whisper.cpp"
DEFAULT_MODEL = "base"
DEFAULT_VAD_MODEL = "silero-v6.2.0"


@dataclass(frozen=True)
class WhisperSetupResult:
    cli_path: Path | None = None
    whisper_model: Path | None = None
    vad_model: Path | None = None


def default_model_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "OpenLRC" / "models"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_submodule() -> None:
    if (VENDOR_DIR / "CMakeLists.txt").exists():
        return
    run(["git", "submodule", "update", "--init", "--recursive", "vendor/whisper.cpp"])
    if not (VENDOR_DIR / "CMakeLists.txt").exists():
        raise RuntimeError("vendor/whisper.cpp is missing after submodule initialization.")


def build_whisper_cpp() -> Path:
    build_dir = VENDOR_DIR / "build"
    run(["cmake", "-S", ".", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], cwd=VENDOR_DIR)
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", str(os.cpu_count() or 1)])

    cli_path = build_dir / "bin" / "whisper-cli"
    if not cli_path.exists():
        raise RuntimeError(f"Build completed, but whisper-cli was not found at {cli_path}")
    return cli_path


def download_models(model_dir: Path, model: str, vad_model: str) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    run(["sh", "models/download-ggml-model.sh", model, str(model_dir)], cwd=VENDOR_DIR)
    run(["sh", "models/download-vad-model.sh", vad_model, str(model_dir)], cwd=VENDOR_DIR)

    whisper_model = model_dir / f"ggml-{model}.bin"
    vad_model_path = model_dir / f"ggml-{vad_model}.bin"
    missing = [str(path) for path in (whisper_model, vad_model_path) if not path.exists()]
    if missing:
        raise RuntimeError(f"Model download finished, but expected files are missing: {', '.join(missing)}")
    return whisper_model, vad_model_path


def setup_whisper_cpp(
    *,
    model: str = DEFAULT_MODEL,
    vad_model: str = DEFAULT_VAD_MODEL,
    model_dir: Path | None = None,
    skip_build: bool = False,
    skip_models: bool = False,
) -> WhisperSetupResult:
    ensure_submodule()

    cli_path: Path | None = None
    if not skip_build:
        cli_path = build_whisper_cpp()

    whisper_model: Path | None = None
    vad_model_path: Path | None = None
    if not skip_models:
        whisper_model, vad_model_path = download_models(
            (model_dir or default_model_dir()).expanduser(), model, vad_model
        )

    return WhisperSetupResult(cli_path=cli_path, whisper_model=whisper_model, vad_model=vad_model_path)
