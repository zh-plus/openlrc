#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Prepare the vendored llama.cpp server and default local LLM model."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_REPO,
    default_model_url,
    user_llm_model_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class LlamaSetupResult:
    server_path: Path | None = None
    cli_path: Path | None = None
    model_path: Path | None = None


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_submodule() -> None:
    if (VENDOR_DIR / "CMakeLists.txt").exists():
        return
    run(["git", "submodule", "update", "--init", "--recursive", "vendor/llama.cpp"])
    if not (VENDOR_DIR / "CMakeLists.txt").exists():
        raise RuntimeError("vendor/llama.cpp is missing after submodule initialization.")


def build_llama_cpp() -> tuple[Path, Path]:
    build_dir = VENDOR_DIR / "build"
    run(["cmake", "-S", ".", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], cwd=VENDOR_DIR)
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "llama-cli",
            "--parallel",
            str(os.cpu_count() or 1),
        ],
        cwd=VENDOR_DIR,
    )

    server_path = build_dir / "bin" / "llama-server"
    cli_path = build_dir / "bin" / "llama-cli"
    missing = [str(path) for path in (server_path, cli_path) if not path.exists()]
    if missing:
        raise RuntimeError(f"Build completed, but expected binaries are missing: {', '.join(missing)}")
    return server_path, cli_path


def download_model(
    *, model_dir: Path, model_repo: str, model_file: str, model_url: str | None, revision: str, force: bool
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / model_file
    if destination.exists() and not force:
        print(f"Model already exists: {destination}")
        return destination

    if force and destination.exists():
        destination.unlink()

    url = model_url or default_model_url(model_repo=model_repo, model_file=model_file, revision=revision)
    run(["curl", "-L", "--fail", "--continue-at", "-", "--output", str(destination), url])

    if not destination.exists():
        raise RuntimeError(f"Model download finished, but expected file is missing: {destination}")
    return destination


def setup_llama_cpp(
    *,
    model_repo: str = DEFAULT_LLAMA_MODEL_REPO,
    model_file: str = DEFAULT_LLAMA_MODEL_FILE,
    revision: str = "main",
    model_url: str | None = None,
    model_dir: Path | None = None,
    skip_build: bool = False,
    skip_models: bool = False,
    force: bool = False,
) -> LlamaSetupResult:
    ensure_submodule()

    server_path: Path | None = None
    cli_path: Path | None = None
    if not skip_build:
        server_path, cli_path = build_llama_cpp()

    model_path: Path | None = None
    if not skip_models:
        model_path = download_model(
            model_dir=(model_dir or user_llm_model_dir()).expanduser(),
            model_repo=model_repo,
            model_file=model_file,
            model_url=model_url,
            revision=revision,
            force=force,
        )

    return LlamaSetupResult(server_path=server_path, cli_path=cli_path, model_path=model_path)
