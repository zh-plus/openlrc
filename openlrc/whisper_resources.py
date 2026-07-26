#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""Resolve whisper.cpp binaries and model files for local dev and packaging."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

OPENLRC_WHISPER_CLI = "OPENLRC_WHISPER_CLI"
OPENLRC_WHISPER_MODEL = "OPENLRC_WHISPER_MODEL"
OPENLRC_WHISPER_VAD_MODEL = "OPENLRC_WHISPER_VAD_MODEL"
OPENLRC_WHISPER_MODEL_DIR = "OPENLRC_WHISPER_MODEL_DIR"

DEFAULT_MODEL_NAME = "ggml-base.bin"
DEFAULT_VAD_MODEL_NAME = "ggml-silero-v6.2.0.bin"
SETUP_COMMAND = "uv run python scripts/setup_whisper_cpp.py"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vendor_dir() -> Path:
    return repo_root() / "vendor" / "whisper.cpp"


def vendor_cli_path() -> Path:
    return vendor_dir() / "build" / "bin" / "whisper-cli"


def vendor_model_dir() -> Path:
    return vendor_dir() / "models"


def user_model_dir() -> Path:
    env_dir = os.environ.get(OPENLRC_WHISPER_MODEL_DIR)
    if env_dir:
        return Path(env_dir).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenLRC" / "models"

    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")).expanduser()
    return data_home / "openlrc" / "models"


def _has_path_separator(value: str) -> bool:
    return "/" in value or (os.sep != "/" and os.sep in value) or (os.altsep is not None and os.altsep in value)


def _resolve_existing_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    return None


def _resolve_executable(value: str) -> str | None:
    if not value:
        return None

    if _has_path_separator(value) or Path(value).is_absolute():
        path = _resolve_existing_path(value)
        if path is not None and os.access(path, os.X_OK):
            return str(path)
        return None

    return shutil.which(value)


def _app_bundle_cli_path() -> str | None:
    """Return a bundled whisper-cli from a macOS .app, when present."""
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.name != "Contents":
            continue
        resources = parent / "Resources"
        for candidate in (resources / "bin" / "whisper-cli", resources / "whisper-cli"):
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _format_missing_cli(value: str | None = None) -> str:
    requested = f" Requested: {value!r}." if value else ""
    return (
        f"whisper-cli could not be found.{requested} Run `{SETUP_COMMAND}` after "
        "`git submodule update --init --recursive`, or set "
        f"{OPENLRC_WHISPER_CLI} / TranscriptionConfig.cli_path to a whisper-cli executable."
    )


def resolve_whisper_cli(cli_path: str | None = "") -> str:
    """Resolve whisper-cli using config, env, app bundle, submodule build, then PATH."""
    if cli_path:
        resolved = _resolve_executable(cli_path)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_cli(cli_path))

    env_cli = os.environ.get(OPENLRC_WHISPER_CLI, "")
    if env_cli:
        resolved = _resolve_executable(env_cli)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_cli(env_cli))

    bundled_cli = _app_bundle_cli_path()
    if bundled_cli:
        return bundled_cli

    vendor_cli = vendor_cli_path()
    if vendor_cli.exists() and os.access(vendor_cli, os.X_OK):
        return str(vendor_cli.resolve())

    path_cli = shutil.which("whisper-cli")
    if path_cli:
        return path_cli

    raise FileNotFoundError(_format_missing_cli())


def _normalize_whisper_model_name(value: str) -> str:
    if not value or value.endswith(".bin") or _has_path_separator(value) or Path(value).is_absolute():
        return value
    return f"ggml-{value}.bin"


def _normalize_vad_model_name(value: str) -> str:
    if not value or value.endswith(".bin") or _has_path_separator(value) or Path(value).is_absolute():
        return value
    return f"ggml-{value}.bin"


def _model_candidates(model_value: str, normalizer) -> list[Path]:
    normalized = normalizer(model_value)
    path = Path(normalized).expanduser()
    if path.is_absolute() or _has_path_separator(normalized):
        return [path]
    return [user_model_dir() / normalized, vendor_model_dir() / normalized]


def _resolve_model_file(model_value: str | None, *, env_var: str, default_name: str, label: str, normalizer) -> str:
    value = model_value if model_value is not None else default_name
    env_value = os.environ.get(env_var)

    if env_value and value in ("", default_name):
        value = env_value

    for candidate in _model_candidates(value, normalizer):
        if candidate.exists():
            return str(candidate.resolve())

    checked = ", ".join(str(path) for path in _model_candidates(value, normalizer))
    return_hint = (
        f"{label} model could not be found. Checked: {checked}. Run `{SETUP_COMMAND}` "
        f"or set {env_var} / TranscriptionConfig.{label} to an existing model file. "
        f"Set {OPENLRC_WHISPER_MODEL_DIR} to change the default model directory."
    )
    raise FileNotFoundError(return_hint)


def resolve_whisper_model_path(model_path: str | None = DEFAULT_MODEL_NAME) -> str:
    return _resolve_model_file(
        model_path,
        env_var=OPENLRC_WHISPER_MODEL,
        default_name=DEFAULT_MODEL_NAME,
        label="whisper_model",
        normalizer=_normalize_whisper_model_name,
    )


def resolve_vad_model_path(vad_model_path: str | None = DEFAULT_VAD_MODEL_NAME) -> str:
    if vad_model_path == "":
        return ""
    return _resolve_model_file(
        vad_model_path,
        env_var=OPENLRC_WHISPER_VAD_MODEL,
        default_name=DEFAULT_VAD_MODEL_NAME,
        label="vad_model",
        normalizer=_normalize_vad_model_name,
    )
