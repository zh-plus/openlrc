"""Versioned, secret-free local settings for interactive frontends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from openlrc.application.paths import app_support_dir, atomic_write_json, backup_corrupt_file
from openlrc.llama_resources import (
    DEFAULT_LLAMA_CONTEXT_SIZE,
    DEFAULT_LLAMA_HOST,
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_PORT,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    HY_MT2_7B_PROFILE,
    QWEN35_9B_PROFILE,
)
from openlrc.whisper_resources import DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME


@dataclass(slots=True)
class GeneralSettings:
    theme: str = "openlrc-dark"
    language: str = "en"
    logo_animation: bool = True
    reduce_motion: bool = False
    ascii_status: str = "auto"


@dataclass(slots=True)
class ProviderSettings:
    enabled: bool = False
    model: str = ""
    base_url: str = ""
    proxy: str = ""


def _default_providers() -> dict[str, ProviderSettings]:
    return {
        "openai": ProviderSettings(enabled=True, model="gpt-4.1-nano"),
        "anthropic": ProviderSettings(),
        "google": ProviderSettings(),
        "litellm": ProviderSettings(),
        "third_party": ProviderSettings(),
    }


@dataclass(slots=True)
class LocalModelSettings:
    whisper_model: str = DEFAULT_MODEL_NAME
    vad_model: str = DEFAULT_VAD_MODEL_NAME
    whisper_cli: str = ""
    qwen_model: str = QWEN35_9B_PROFILE
    hymt2_profile: str = HY_MT2_7B_PROFILE
    hymt2_model: str = ""
    llama_server: str = ""
    host: str = DEFAULT_LLAMA_HOST
    port: int = DEFAULT_LLAMA_PORT
    context_size: int = DEFAULT_LLAMA_CONTEXT_SIZE
    gpu_layers: str = "all"
    idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT
    startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT


@dataclass(slots=True)
class TranscriptionDefaults:
    source_language: str = ""
    noise_suppress: bool = False
    skip_preprocess: bool = False
    use_gpu: bool = True
    flash_attn: bool = True


@dataclass(slots=True)
class WorkflowDefaults:
    target_language: str = "zh-cn"
    bilingual_subtitle: bool = False
    subtitle_optimization: str = "aggressive"
    clear_temp: bool = True
    clear_checkpoint: bool = True
    glossary_strict: bool = True
    force_glossary: bool = False
    edit_rounds: int = 1
    enable_restore: bool = False
    default_glossary: str = ""


@dataclass(slots=True)
class AppSettings:
    schema_version: int = 1
    general: GeneralSettings = field(default_factory=GeneralSettings)
    providers: dict[str, ProviderSettings] = field(default_factory=_default_providers)
    local_models: LocalModelSettings = field(default_factory=LocalModelSettings)
    transcription: TranscriptionDefaults = field(default_factory=TranscriptionDefaults)
    workflow: WorkflowDefaults = field(default_factory=WorkflowDefaults)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AppSettings:
        if payload.get("schema_version", 1) != 1:
            raise ValueError("Unsupported settings schema version.")
        general_raw = _mapping(payload.get("general"))
        local_raw = _mapping(payload.get("local_models"))
        trans_raw = _mapping(payload.get("transcription"))
        workflow_raw = _mapping(payload.get("workflow"))
        providers = _default_providers()
        provider_raw = payload.get("providers")
        if isinstance(provider_raw, dict):
            for name, value in provider_raw.items():
                if name in providers and isinstance(value, dict):
                    providers[name] = _construct(ProviderSettings, value)
        general = _construct(GeneralSettings, general_raw)
        if general.theme not in {"openlrc-dark", "textual-dark"}:
            general.theme = "openlrc-dark"
        if general.language not in {"en", "zh-cn"}:
            general.language = "en"
        return cls(
            general=general,
            providers=providers,
            local_models=_construct(LocalModelSettings, local_raw),
            transcription=_construct(TranscriptionDefaults, trans_raw),
            workflow=_construct(WorkflowDefaults, workflow_raw),
        )


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _construct(model_type, raw: dict[str, object]):
    allowed = model_type.__dataclass_fields__
    values = {key: value for key, value in raw.items() if key in allowed}
    return cast(Any, model_type)(**values)


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_support_dir() / "settings.json"
        self.last_warning: str | None = None

    def load(self) -> AppSettings:
        self.last_warning = None
        if not self.path.exists():
            return AppSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Settings root must be an object.")
            return AppSettings.from_dict(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            try:
                backup = backup_corrupt_file(self.path)
                self.last_warning = f"Invalid settings were backed up to {backup}: {exc}"
            except OSError:
                self.last_warning = f"Invalid settings could not be loaded: {exc}"
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        atomic_write_json(self.path, settings.to_dict())
