#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Resolve llama.cpp binaries and GGUF model files for local LLM translation."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

OPENLRC_LLAMA_SERVER = "OPENLRC_LLAMA_SERVER"
OPENLRC_LLAMA_CLI = "OPENLRC_LLAMA_CLI"
OPENLRC_LLAMA_MODEL = "OPENLRC_LLAMA_MODEL"
OPENLRC_LLAMA_MODEL_DIR = "OPENLRC_LLAMA_MODEL_DIR"

DEFAULT_LLAMA_MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_LLAMA_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_LLAMA_MODEL_ALIAS = "qwen3.5-9b-local"
QWEN35_9B_PROFILE = "qwen3.5-9b"
HY_MT2_7B_PROFILE = "hy-mt2-7b"
HY_MT2_30B_A3B_PROFILE = "hy-mt2-30b-a3b"
HY_MT2_PROMPT_PROFILE = "hy-mt2"
HY_MT2_7B_MODEL_REPO = "tencent/Hy-MT2-7B-GGUF"
HY_MT2_7B_MODEL_FILE = "HY-MT2-7B-Q6_K.gguf"
HY_MT2_7B_Q4_MODEL_FILE = "Hy-MT2-7B-Q4_K_M.gguf"
HY_MT2_7B_Q8_MODEL_FILE = "HY-MT2-7B-Q8_0.gguf"
HY_MT2_7B_MODEL_ALIAS = "hy-mt2-7b-local"
DEFAULT_LLAMA_HOST = "127.0.0.1"
DEFAULT_LLAMA_PORT = 8088
DEFAULT_LLAMA_CONTEXT_SIZE = 32768
DEFAULT_LLAMA_IDLE_TIMEOUT = 300
DEFAULT_LLAMA_STARTUP_TIMEOUT = 120
LOCAL_LLAMA_API_KEY = "openlrc-local"
SETUP_COMMAND = "uv run python scripts/setup_llama_cpp.py"


@dataclass(frozen=True)
class LocalLLMProfile:
    """Registered local llama.cpp model profile."""

    name: str
    server_alias: str
    model_repo: str | None
    model_file: str | None
    prompt_profile: str = "default"
    context_window: int = DEFAULT_LLAMA_CONTEXT_SIZE
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None


LOCAL_LLM_PROFILES: dict[str, LocalLLMProfile] = {
    QWEN35_9B_PROFILE: LocalLLMProfile(
        name=QWEN35_9B_PROFILE,
        server_alias=DEFAULT_LLAMA_MODEL_ALIAS,
        model_repo=DEFAULT_LLAMA_MODEL_REPO,
        model_file=DEFAULT_LLAMA_MODEL_FILE,
    ),
    HY_MT2_7B_PROFILE: LocalLLMProfile(
        name=HY_MT2_7B_PROFILE,
        server_alias=HY_MT2_7B_MODEL_ALIAS,
        model_repo=HY_MT2_7B_MODEL_REPO,
        model_file=HY_MT2_7B_MODEL_FILE,
        prompt_profile=HY_MT2_PROMPT_PROFILE,
        temperature=0.7,
        top_p=0.6,
        top_k=20,
        repeat_penalty=1.05,
        max_tokens=4096,
    ),
    HY_MT2_30B_A3B_PROFILE: LocalLLMProfile(
        name=HY_MT2_30B_A3B_PROFILE,
        server_alias="hy-mt2-30b-a3b-local",
        model_repo=None,
        model_file=None,
        prompt_profile=HY_MT2_PROMPT_PROFILE,
        temperature=0.7,
        top_p=1.0,
        top_k=-1,
        repeat_penalty=1.0,
        max_tokens=4096,
    ),
}

_PROFILE_ALIASES = {
    "qwen3.5-9b": QWEN35_9B_PROFILE,
    "qwen35-9b": QWEN35_9B_PROFILE,
    DEFAULT_LLAMA_MODEL_ALIAS: QWEN35_9B_PROFILE,
    "hy-mt2-7b": HY_MT2_7B_PROFILE,
    "hymt2-7b": HY_MT2_7B_PROFILE,
    "hy_mt2_7b": HY_MT2_7B_PROFILE,
    "hy-mt2-7b-q6": HY_MT2_7B_PROFILE,
    "hy-mt2-7b-q6-k": HY_MT2_7B_PROFILE,
    HY_MT2_7B_MODEL_ALIAS: HY_MT2_7B_PROFILE,
    "hy-mt2-30b-a3b": HY_MT2_30B_A3B_PROFILE,
    "hymt2-30b-a3b": HY_MT2_30B_A3B_PROFILE,
    "hy_mt2_30b_a3b": HY_MT2_30B_A3B_PROFILE,
    "hy-mt2-30b-a3b-local": HY_MT2_30B_A3B_PROFILE,
}

_MODEL_ALIASES = {
    "qwen3.5-9b": DEFAULT_LLAMA_MODEL_FILE,
    "qwen35-9b": DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_ALIAS: DEFAULT_LLAMA_MODEL_FILE,
    "hy-mt2-7b": HY_MT2_7B_MODEL_FILE,
    "hymt2-7b": HY_MT2_7B_MODEL_FILE,
    "hy_mt2_7b": HY_MT2_7B_MODEL_FILE,
    "hy-mt2-7b-q6": HY_MT2_7B_MODEL_FILE,
    "hy-mt2-7b-q6-k": HY_MT2_7B_MODEL_FILE,
    HY_MT2_7B_MODEL_ALIAS: HY_MT2_7B_MODEL_FILE,
    "hy-mt2-7b-q4": HY_MT2_7B_Q4_MODEL_FILE,
    "hy-mt2-7b-q4-k-m": HY_MT2_7B_Q4_MODEL_FILE,
    "hy-mt2-7b-q8": HY_MT2_7B_Q8_MODEL_FILE,
    "hy-mt2-7b-q8-0": HY_MT2_7B_Q8_MODEL_FILE,
}


def get_local_llm_profile(name: str) -> LocalLLMProfile:
    normalized = _PROFILE_ALIASES.get(name.lower(), name.lower())
    try:
        return LOCAL_LLM_PROFILES[normalized]
    except KeyError:
        supported = ", ".join(sorted(LOCAL_LLM_PROFILES))
        raise ValueError(f"Unsupported local LLM profile {name!r}. Choose from: {supported}.") from None


def infer_local_llm_profile(model: str) -> str | None:
    """Infer a profile from a semantic model alias when that mapping is unambiguous."""
    profile_name = _PROFILE_ALIASES.get(model.lower())
    if profile_name == HY_MT2_30B_A3B_PROFILE:
        return None
    return profile_name


def is_hy_mt2_30b_profile_alias(value: str) -> bool:
    return _PROFILE_ALIASES.get(value.lower()) == HY_MT2_30B_A3B_PROFILE


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vendor_dir() -> Path:
    return repo_root() / "vendor" / "llama.cpp"


def vendor_server_path() -> Path:
    return vendor_dir() / "build" / "bin" / "llama-server"


def vendor_cli_path() -> Path:
    return vendor_dir() / "build" / "bin" / "llama-cli"


def vendor_model_dir() -> Path:
    return vendor_dir() / "models"


def user_llm_model_dir() -> Path:
    env_dir = os.environ.get(OPENLRC_LLAMA_MODEL_DIR)
    if env_dir:
        return Path(env_dir).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenLRC" / "models" / "llm"

    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")).expanduser()
    return data_home / "openlrc" / "models" / "llm"


def default_model_url(
    model_repo: str = DEFAULT_LLAMA_MODEL_REPO, model_file: str = DEFAULT_LLAMA_MODEL_FILE, revision: str = "main"
) -> str:
    quoted_file = quote(model_file)
    quoted_revision = quote(revision, safe="")
    return f"https://huggingface.co/{model_repo}/resolve/{quoted_revision}/{quoted_file}"


def normalize_llama_model_name(value: str) -> str:
    if not value:
        return DEFAULT_LLAMA_MODEL_FILE

    lowered = value.lower()
    if lowered in _MODEL_ALIASES:
        return _MODEL_ALIASES[lowered]

    if value.endswith(".gguf") or _has_path_separator(value) or Path(value).is_absolute():
        return value

    return f"{value}.gguf"


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


def _app_bundle_binary_path(binary_name: str) -> str | None:
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.name != "Contents":
            continue
        resources = parent / "Resources"
        for candidate in (resources / "bin" / binary_name, resources / binary_name):
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _format_missing_binary(binary_name: str, env_var: str, value: str | None = None) -> str:
    requested = f" Requested: {value!r}." if value else ""
    return (
        f"{binary_name} could not be found.{requested} Run `{SETUP_COMMAND}` after "
        "`git submodule update --init --recursive`, or set "
        f"{env_var} / LocalLLMConfig.{binary_name.replace('-', '_')}_path to an executable."
    )


def resolve_llama_server(server_path: str | None = "") -> str:
    """Resolve llama-server using config, env, app bundle, submodule build, then PATH."""
    if server_path:
        resolved = _resolve_executable(server_path)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_binary("llama-server", OPENLRC_LLAMA_SERVER, server_path))

    env_server = os.environ.get(OPENLRC_LLAMA_SERVER, "")
    if env_server:
        resolved = _resolve_executable(env_server)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_binary("llama-server", OPENLRC_LLAMA_SERVER, env_server))

    bundled_server = _app_bundle_binary_path("llama-server")
    if bundled_server:
        return bundled_server

    vendor_server = vendor_server_path()
    if vendor_server.exists() and os.access(vendor_server, os.X_OK):
        return str(vendor_server.resolve())

    path_server = shutil.which("llama-server")
    if path_server:
        return path_server

    raise FileNotFoundError(_format_missing_binary("llama-server", OPENLRC_LLAMA_SERVER))


def resolve_llama_cli(cli_path: str | None = "") -> str:
    """Resolve llama-cli using config, env, app bundle, submodule build, then PATH."""
    if cli_path:
        resolved = _resolve_executable(cli_path)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_binary("llama-cli", OPENLRC_LLAMA_CLI, cli_path))

    env_cli = os.environ.get(OPENLRC_LLAMA_CLI, "")
    if env_cli:
        resolved = _resolve_executable(env_cli)
        if resolved:
            return resolved
        raise FileNotFoundError(_format_missing_binary("llama-cli", OPENLRC_LLAMA_CLI, env_cli))

    bundled_cli = _app_bundle_binary_path("llama-cli")
    if bundled_cli:
        return bundled_cli

    vendor_cli = vendor_cli_path()
    if vendor_cli.exists() and os.access(vendor_cli, os.X_OK):
        return str(vendor_cli.resolve())

    path_cli = shutil.which("llama-cli")
    if path_cli:
        return path_cli

    raise FileNotFoundError(_format_missing_binary("llama-cli", OPENLRC_LLAMA_CLI))


def _model_candidates(model_value: str) -> list[Path]:
    normalized = normalize_llama_model_name(model_value)
    path = Path(normalized).expanduser()
    if path.is_absolute() or _has_path_separator(normalized):
        return [path]
    return [user_llm_model_dir() / normalized, vendor_model_dir() / normalized]


def resolve_llama_model_path(model_path: str | None = DEFAULT_LLAMA_MODEL_FILE) -> str:
    """Resolve a GGUF model path using config, env, user model dir, then vendor model dir."""
    value = model_path if model_path is not None else DEFAULT_LLAMA_MODEL_FILE
    value = normalize_llama_model_name(value)

    env_model = os.environ.get(OPENLRC_LLAMA_MODEL)
    if env_model and value == DEFAULT_LLAMA_MODEL_FILE:
        value = env_model

    for candidate in _model_candidates(value):
        if candidate.exists():
            return str(candidate.resolve())

    checked = ", ".join(str(path) for path in _model_candidates(value))
    raise FileNotFoundError(
        f"llama.cpp GGUF model could not be found. Checked: {checked}. Run `{SETUP_COMMAND}` "
        f"or set {OPENLRC_LLAMA_MODEL} / LocalLLMConfig.model_path to an existing .gguf file. "
        f"Set {OPENLRC_LLAMA_MODEL_DIR} to change the default model directory."
    )
