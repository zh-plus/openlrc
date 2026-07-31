"""Shared resource checks for CLI diagnostics and interactive model status."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_FILE,
    HY_MT2_7B_MODEL_FILE,
    resolve_llama_cli,
    resolve_llama_model_path,
    resolve_llama_server,
    user_llm_model_dir,
)
from openlrc.llama_resources import vendor_dir as llama_vendor_dir
from openlrc.whisper_resources import (
    DEFAULT_MODEL_NAME,
    DEFAULT_VAD_MODEL_NAME,
    resolve_vad_model_path,
    resolve_whisper_cli,
    resolve_whisper_model_path,
    user_model_dir,
)
from openlrc.whisper_resources import vendor_dir as whisper_vendor_dir


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    name: str
    available: bool
    detail: str
    hint: str = ""
    role: str = "dependency"


def _path_status(name: str, path: Path, hint: str, role: str = "dependency") -> ResourceStatus:
    return ResourceStatus(name, path.exists(), str(path) if path.exists() else f"Missing: {path}", hint, role)


def _executable_status(name: str, executable: str, hint: str) -> ResourceStatus:
    resolved = shutil.which(executable)
    return ResourceStatus(name, bool(resolved), resolved or f"{executable} not found on PATH", hint)


def _resolver_status(name: str, resolver, hint: str, role: str = "model") -> ResourceStatus:
    try:
        resolved = resolver()
    except Exception as exc:
        return ResourceStatus(name, False, str(exc), hint, role)
    return ResourceStatus(name, True, str(resolved), hint, role)


def _whisper_cli_version_status(cli_path: str | None) -> ResourceStatus:
    hint = "Run `openlrc setup whisper`, then retry `openlrc doctor`."
    if cli_path is None:
        return ResourceStatus("whisper-cli version", False, "Unavailable because whisper-cli is missing.", hint)
    try:
        completed = subprocess.run([cli_path, "--version"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ResourceStatus("whisper-cli version", False, f"Version probe failed: {exc}", hint)

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "No version output")
    if completed.returncode:
        return ResourceStatus(
            "whisper-cli version",
            False,
            f"Version probe exited with code {completed.returncode}: {first_line[:500]}",
            hint,
        )
    return ResourceStatus("whisper-cli version", True, first_line[:500], "Runtime inference is not run by doctor.")


def _cmake_cache_bool(cache_text: str, key: str) -> bool | None:
    prefix = f"{key}:BOOL="
    for line in cache_text.splitlines():
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip().upper()
            if value in {"ON", "TRUE", "YES", "1"}:
                return True
            if value in {"OFF", "FALSE", "NO", "0"}:
                return False
    return None


def _whisper_metal_status(cli_path: str | None) -> ResourceStatus:
    cpu_hint = (
        "Doctor does not run inference. Use the opt-in Metal smoke to verify runtime acceleration, "
        "or pass `--no-whisper-gpu --no-whisper-flash-attn` for an explicit CPU run."
    )
    if cli_path is None:
        return ResourceStatus(
            "Whisper Metal build",
            False,
            "Unavailable because whisper-cli is missing; CPU and Metal runtime were not probed.",
            "Run `openlrc setup whisper`.",
        )

    resolved_cli = Path(cli_path).expanduser().resolve(strict=False)
    vendor_cli = whisper_vendor_dir() / "build" / "bin" / "whisper-cli"
    if resolved_cli != vendor_cli.resolve(strict=False):
        return ResourceStatus(
            "Whisper Metal build",
            True,
            "Build capability unknown for the resolved non-vendor binary; runtime not probed and CPU remains usable.",
            cpu_hint,
        )

    cache_path = whisper_vendor_dir() / "build" / "CMakeCache.txt"
    try:
        cache_text = cache_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ResourceStatus(
            "Whisper Metal build",
            True,
            f"Build capability unknown because {cache_path} is unavailable; runtime not probed and CPU remains usable.",
            cpu_hint,
        )

    metal = _cmake_cache_bool(cache_text, "GGML_METAL")
    embedded = _cmake_cache_bool(cache_text, "GGML_METAL_EMBED_LIBRARY")
    if metal is True:
        embedded_detail = (
            "embedded library enabled"
            if embedded is True
            else "embedded library disabled"
            if embedded is False
            else "embedded library setting unknown"
        )
        detail = f"Metal enabled at build time ({embedded_detail}); runtime not probed and CPU remains usable."
    elif metal is False:
        detail = "Metal disabled at build time; runtime not probed and CPU remains usable."
    else:
        detail = "GGML_METAL is absent from CMakeCache; runtime not probed and CPU remains usable."
    return ResourceStatus("Whisper Metal build", True, detail, cpu_hint)


class ResourceStatusService:
    def doctor(self) -> list[ResourceStatus]:
        try:
            whisper_cli = resolve_whisper_cli("")
        except Exception:
            whisper_cli = None
        return [
            _executable_status("ffmpeg", "ffmpeg", "Install ffmpeg and make sure it is on PATH."),
            _executable_status("cmake", "cmake", "Install CMake and Xcode Command Line Tools."),
            _path_status(
                "whisper.cpp submodule",
                whisper_vendor_dir() / "CMakeLists.txt",
                "Run `git submodule update --init --recursive`.",
            ),
            _resolver_status("whisper-cli", lambda: resolve_whisper_cli(""), "Run `openlrc setup whisper`."),
            _whisper_cli_version_status(whisper_cli),
            _whisper_metal_status(whisper_cli),
            _resolver_status(
                "Whisper model",
                lambda: resolve_whisper_model_path(DEFAULT_MODEL_NAME),
                "Run `openlrc setup whisper --model base`.",
            ),
            _resolver_status(
                "Whisper VAD model",
                lambda: resolve_vad_model_path(DEFAULT_VAD_MODEL_NAME),
                "Run `openlrc setup whisper`.",
            ),
            _path_status(
                "llama.cpp submodule",
                llama_vendor_dir() / "CMakeLists.txt",
                "Run `git submodule update --init --recursive`.",
            ),
            _resolver_status("llama-server", lambda: resolve_llama_server(""), "Run `openlrc setup llama`."),
            _resolver_status("llama-cli", lambda: resolve_llama_cli(""), "Run `openlrc setup llama`."),
            _resolver_status(
                "Qwen GGUF model",
                lambda: resolve_llama_model_path(DEFAULT_LLAMA_MODEL_FILE),
                "Run `openlrc setup llama`.",
            ),
        ]

    def models(self, settings=None) -> list[ResourceStatus]:
        whisper_model = getattr(settings, "whisper_model", DEFAULT_MODEL_NAME)
        vad_model = getattr(settings, "vad_model", DEFAULT_VAD_MODEL_NAME)
        whisper_cli = getattr(settings, "whisper_cli", "")
        qwen_model = getattr(settings, "qwen_model", DEFAULT_LLAMA_MODEL_FILE)
        hymt2_model = getattr(settings, "hymt2_model", "") or HY_MT2_7B_MODEL_FILE
        llama_server = getattr(settings, "llama_server", "")
        return [
            _resolver_status(
                "whisper-cli", lambda: resolve_whisper_cli(whisper_cli), "Run `openlrc setup whisper`.", "transcription"
            ),
            _resolver_status(
                "Whisper default model",
                lambda: resolve_whisper_model_path(whisper_model),
                f"Default directory: {user_model_dir()}",
                "transcription",
            ),
            _resolver_status(
                "Whisper VAD model",
                lambda: resolve_vad_model_path(vad_model),
                f"Default directory: {user_model_dir()}",
                "transcription",
            ),
            _resolver_status(
                "Local Qwen GGUF",
                lambda: resolve_llama_model_path(qwen_model),
                f"Default directory: {user_llm_model_dir()}",
                "translation",
            ),
            _resolver_status(
                "Local Hy-MT2 7B Q6_K GGUF",
                lambda: resolve_llama_model_path(hymt2_model),
                "Optional. Run `openlrc setup llama --local-model-profile hy-mt2-7b-q6-k`.",
                "translation",
            ),
            _resolver_status(
                "llama-server", lambda: resolve_llama_server(llama_server), "Run `openlrc setup llama`.", "translation"
            ),
        ]
