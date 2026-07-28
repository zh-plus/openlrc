"""Secret-free editable Workflow recipes used by interactive frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from openlrc.application.credentials import CredentialStore
from openlrc.application.settings import AppSettings
from openlrc.config import (
    ContextAssistance,
    ContextLLMConfig,
    EditConfig,
    GlossaryOptions,
    HyMT2Mode,
    SubtitleOptimizationMode,
    TranscriptionConfig,
    context_model_required,
)
from openlrc.context import CharacterBrief, TranslationBriefInput
from openlrc.llama_resources import QWEN35_9B_PROFILE
from openlrc.models import ModelConfig, ModelProvider
from openlrc.workflow import (
    RunRequest,
    TranscribeRequest,
    TranslateRequest,
    TranslationMode,
    WorkflowKind,
    WorkflowRequest,
    WorkflowTranslationFactory,
)

PROVIDER_MAP = {
    "openai": ModelProvider.OPENAI,
    "anthropic": ModelProvider.ANTHROPIC,
    "google": ModelProvider.GOOGLE,
    "litellm": ModelProvider.LITELLM,
    "third_party": ModelProvider.THIRD_PARTY,
}


def parse_brief_characters(value: str) -> list[CharacterBrief]:
    """Parse TUI character mappings without requiring a complete Workflow Draft."""
    characters: list[CharacterBrief] = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        source, separator, target = line.partition("=")
        if not separator or not source.strip() or not target.strip():
            raise ValueError(f"Brief character line {line_number} must use 'Source Name = Target Name'.")
        characters.append(CharacterBrief(source_name=source.strip(), target_name=target.strip()))
    return characters


@dataclass(slots=True)
class WorkflowDraft:
    task: str = ""
    workflow: str = WorkflowKind.RUN.value
    paths: list[str] = field(default_factory=list)
    source_language: str = ""
    target_language: str = "zh-cn"
    whisper_model: str = ""
    vad_model: str = ""
    noise_suppress: bool = False
    skip_preprocess: bool = False
    whisper_use_gpu: bool = True
    whisper_flash_attn: bool = True
    translation_backend: str = ""
    mode: str = TranslationMode.FAST.value
    provider: str = "openai"
    primary_model: str = ""
    retry_provider: str = ""
    retry_model: str = ""
    reviewer_provider: str = ""
    reviewer_model: str = ""
    fee_limit: float = 0.8
    consumer_thread: int = 4
    local_profile: str = ""
    qwen_model: str = ""
    hymt2_model: str = ""
    context_provider: str = ""
    context_model: str = ""
    context_base_url: str = ""
    context_fee_limit: float = 0.8
    context_assistance: str = ContextAssistance.AUTO.value
    glossary_path: str = ""
    glossary_strict: bool = True
    force_glossary: bool = False
    brief_summary: str = ""
    brief_characters: str = ""
    brief_tone_style: str = ""
    edit_rounds: int = 1
    enable_restore: bool = False
    bilingual_subtitle: bool = False
    subtitle_optimization: str = SubtitleOptimizationMode.AGGRESSIVE.value
    clear_temp: bool = True
    clear_checkpoint: bool = True

    @classmethod
    def defaults(cls, settings: AppSettings, workflow: WorkflowKind = WorkflowKind.RUN) -> WorkflowDraft:
        return cls(
            workflow=workflow.value,
            source_language=settings.transcription.source_language,
            target_language=settings.workflow.target_language,
            whisper_model=settings.local_models.whisper_model,
            vad_model=settings.local_models.vad_model,
            noise_suppress=settings.transcription.noise_suppress,
            skip_preprocess=settings.transcription.skip_preprocess,
            whisper_use_gpu=settings.transcription.use_gpu,
            whisper_flash_attn=settings.transcription.flash_attn,
            local_profile=settings.local_models.hymt2_profile,
            qwen_model=settings.local_models.qwen_model,
            hymt2_model=settings.local_models.hymt2_model,
            context_provider="local",
            context_model=settings.local_models.qwen_model or QWEN35_9B_PROFILE,
            glossary_strict=settings.workflow.glossary_strict,
            force_glossary=settings.workflow.force_glossary,
            edit_rounds=settings.workflow.edit_rounds,
            enable_restore=settings.workflow.enable_restore,
            bilingual_subtitle=settings.workflow.bilingual_subtitle,
            subtitle_optimization=settings.workflow.subtitle_optimization,
            clear_temp=settings.workflow.clear_temp,
            clear_checkpoint=settings.workflow.clear_checkpoint,
            glossary_path=settings.workflow.default_glossary,
        )

    def to_recipe(self) -> dict[str, object]:
        recipe = asdict(self)
        if self.workflow == WorkflowKind.TRANSCRIBE.value:
            for key in _TRANSLATION_RECIPE_FIELDS:
                recipe.pop(key, None)
        elif self.context_assistance == ContextAssistance.OFF.value:
            for key in ("context_provider", "context_model", "context_base_url", "context_fee_limit"):
                recipe.pop(key, None)
        return recipe

    @classmethod
    def from_recipe(cls, payload: dict[str, object]) -> WorkflowDraft:
        payload = dict(payload)
        if "local_model" in payload and "hymt2_model" not in payload:
            payload["hymt2_model"] = payload.pop("local_model")
        allowed = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**cast(Any, values))

    def validate(self, settings: AppSettings, credentials: CredentialStore) -> None:
        self.build_request(settings, credentials)

    def build_request(self, settings: AppSettings, credentials: CredentialStore) -> WorkflowRequest:
        kind = WorkflowKind(self.workflow)
        paths = tuple(Path(path).expanduser() for path in self.paths)
        if not paths:
            raise ValueError("Select at least one input file.")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Input file not found: " + ", ".join(missing))
        transcription = TranscriptionConfig(
            whisper_model=self.whisper_model or settings.local_models.whisper_model,
            cli_path=settings.local_models.whisper_cli,
            vad_model=self.vad_model,
            asr_options={"use_gpu": self.whisper_use_gpu, "flash_attn": self.whisper_flash_attn},
        )
        source_language = self.source_language.strip() or None
        optimization = SubtitleOptimizationMode(self.subtitle_optimization)
        if kind is WorkflowKind.TRANSCRIBE:
            return TranscribeRequest(
                paths,
                transcription=transcription,
                src_lang=source_language,
                noise_suppress=self.noise_suppress,
                skip_preprocess=self.skip_preprocess,
                subtitle_output=self.task != "transcribe-json",
                subtitle_optimization=optimization,
                clear_temp=self.clear_temp,
            )

        if kind is WorkflowKind.RUN and self.translation_backend == "none":
            return RunRequest(
                paths,
                transcription=transcription,
                translation=None,
                src_lang=source_language,
                target_lang=self.target_language,
                noise_suppress=self.noise_suppress,
                subtitle_optimization=optimization,
                clear_temp=self.clear_temp,
                clear_checkpoint=self.clear_checkpoint,
                skip_preprocess=self.skip_preprocess,
            )

        translation = self._build_translation(settings, credentials)
        if kind is WorkflowKind.TRANSLATE:
            if translation is None:
                raise ValueError("Translate requires a translation mode.")
            return TranslateRequest(
                paths,
                translation=translation,
                target_lang=self.target_language,
                bilingual_sub=self.bilingual_subtitle,
                subtitle_optimization=optimization,
                clear_checkpoint=self.clear_checkpoint,
            )
        if translation is None:
            raise ValueError("Total Run requires Online or Local translation.")
        return RunRequest(
            paths,
            transcription=transcription,
            translation=translation,
            src_lang=source_language,
            target_lang=self.target_language,
            noise_suppress=self.noise_suppress,
            bilingual_sub=self.bilingual_subtitle,
            subtitle_optimization=optimization,
            clear_temp=self.clear_temp,
            clear_checkpoint=self.clear_checkpoint,
            skip_preprocess=self.skip_preprocess,
        )

    def _build_translation(self, settings: AppSettings, credentials: CredentialStore):
        if self.translation_backend == "" and self.workflow == WorkflowKind.TRANSCRIBE.value:
            return None
        if self.translation_backend not in {"online", "local-qwen", "local"}:
            raise ValueError("Choose Online, Local Qwen, or Hy-MT2 translation before continuing.")
        glossary = self.glossary_path.strip() or None
        glossary_options = GlossaryOptions(strict=self.glossary_strict, force=self.force_glossary)
        mode = TranslationMode(self.mode)
        if self.translation_backend == "online" and mode is not TranslationMode.STANDARD:
            raise ValueError("Online translation supports Standard mode only.")
        if self.translation_backend == "local-qwen" and mode is not TranslationMode.STANDARD:
            raise ValueError("Local Qwen translation supports Standard mode only.")
        if self.translation_backend == "local" and mode is TranslationMode.STANDARD:
            raise ValueError("Local translation supports Fast, Normal, Normal Plus, or Pro; not Standard.")
        if not self.target_language.strip():
            raise ValueError("Target language is required for translation.")
        edit_config = EditConfig(
            enabled=False,
            max_rounds=(self.edit_rounds if mode in {TranslationMode.NORMAL_PLUS, TranslationMode.PRO} else 0),
            semantic_review=(self.edit_rounds > 0 and mode in {TranslationMode.NORMAL_PLUS, TranslationMode.PRO}),
            restore_enabled=self.enable_restore,
        )
        if mode is TranslationMode.STANDARD and self.translation_backend == "online":
            if self.fee_limit <= 0:
                raise ValueError("Fee limit must be greater than zero.")
            if self.consumer_thread < 1:
                raise ValueError("Consumers must be at least 1.")
            primary = self._provider_model(self.provider, self.primary_model, settings, credentials)
            retry = (
                self._provider_model(self.retry_provider, self.retry_model, settings, credentials)
                if self.retry_provider and self.retry_model
                else None
            )
            reviewer = (
                self._provider_model(self.reviewer_provider, self.reviewer_model, settings, credentials)
                if self.reviewer_provider and self.reviewer_model
                else None
            )
            return WorkflowTranslationFactory.standard_online(
                primary=primary,
                retry=retry,
                reviewer=reviewer,
                fee_limit=self.fee_limit,
                consumer_thread=self.consumer_thread,
                glossary=glossary,
                glossary_options=glossary_options,
                edit_config=edit_config,
            )
        local = settings.local_models
        if mode is TranslationMode.STANDARD and self.translation_backend == "local-qwen":
            return WorkflowTranslationFactory.standard_local_qwen(
                model=self.qwen_model or local.qwen_model,
                idle_timeout=local.idle_timeout,
                port=local.port,
                server_path=local.llama_server,
                host=local.host,
                ctx_size=local.context_size,
                gpu_layers=local.gpu_layers,
                startup_timeout=local.startup_timeout,
                glossary=glossary,
                glossary_options=glossary_options,
                edit_config=edit_config,
            )
        brief = self._brief(mode)
        context_llm = self._context_model(mode, brief, edit_config, settings, credentials)
        return WorkflowTranslationFactory.hymt2(
            mode=mode,
            profile=self.local_profile or local.hymt2_profile,
            model=self.hymt2_model or local.hymt2_model or None,
            context_llm=context_llm,
            context_assistance=self.context_assistance,
            translation_brief=brief,
            idle_timeout=local.idle_timeout,
            port=local.port,
            server_path=local.llama_server,
            host=local.host,
            ctx_size=local.context_size,
            gpu_layers=local.gpu_layers,
            startup_timeout=local.startup_timeout,
            glossary=glossary,
            glossary_options=glossary_options,
            edit_config=edit_config,
        )

    def _provider_model(
        self,
        provider: str,
        model: str,
        settings: AppSettings,
        credentials: CredentialStore,
        *,
        base_url_override: str = "",
    ) -> ModelConfig:
        if provider not in PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider or '<empty>'}.")
        profile = settings.providers[provider]
        if not profile.enabled:
            raise ValueError(f"Provider {provider} is disabled in Settings.")
        model_name = model.strip() or profile.model.strip()
        if not model_name:
            raise ValueError(f"Configure a model for provider {provider}.")
        credential = credentials.resolve(provider)
        if credential.value is None:
            raise ValueError(f"No API key configured for provider {provider}.")
        return ModelConfig(
            provider=PROVIDER_MAP[provider],
            name=model_name,
            base_url=base_url_override.strip() or profile.base_url.strip() or None,
            api_key=credential.value,
            proxy=profile.proxy.strip() or None,
        )

    def _brief(self, mode: TranslationMode | None = None) -> TranslationBriefInput | None:
        if mode is TranslationMode.FAST:
            return None
        assistance = ContextAssistance(self.context_assistance)
        summary = self.brief_summary.strip()
        if assistance is ContextAssistance.OFF and not summary:
            raise ValueError("Context assistance off requires a Brief summary.")
        if assistance is ContextAssistance.AUTO and not any((summary, self.brief_characters, self.brief_tone_style)):
            return None
        characters = [] if assistance is ContextAssistance.OFF else None
        if self.brief_characters.strip():
            characters = parse_brief_characters(self.brief_characters)
        tone_style = self.brief_tone_style.strip()
        return TranslationBriefInput(
            summary=summary or None,
            characters=characters,
            tone_style=tone_style if assistance is ContextAssistance.OFF else (tone_style or None),
        )

    def requires_context_model(self) -> bool:
        """Return the effective Context requirement for the current Hy-MT2 Draft."""
        mode = TranslationMode(self.mode)
        edit_config = EditConfig(
            enabled=False,
            max_rounds=(self.edit_rounds if mode in {TranslationMode.NORMAL_PLUS, TranslationMode.PRO} else 0),
            semantic_review=(self.edit_rounds > 0 and mode in {TranslationMode.NORMAL_PLUS, TranslationMode.PRO}),
            restore_enabled=self.enable_restore,
        )
        return context_model_required(
            mode=HyMT2Mode(mode.value),
            translation_brief=self._brief(mode),
            edit_config=edit_config,
            context_assistance=self.context_assistance,
        )

    def _context_model(
        self,
        mode: TranslationMode,
        brief: TranslationBriefInput | None,
        edit_config: EditConfig,
        settings: AppSettings,
        credentials: CredentialStore,
    ) -> ContextLLMConfig | None:
        required = context_model_required(
            mode=HyMT2Mode(mode.value),
            translation_brief=brief,
            edit_config=edit_config,
            context_assistance=self.context_assistance,
        )
        if not required:
            return None
        if not self.context_provider or not self.context_model:
            raise ValueError(f"{mode.value} requires a context provider and model for this Brief/review configuration.")
        local = settings.local_models
        if self.context_provider == "local":
            return ContextLLMConfig.local_qwen35_9b(
                model=self.context_model,
                port=local.port,
                server_path=local.llama_server,
                host=local.host,
                ctx_size=local.context_size,
                gpu_layers=local.gpu_layers,
                startup_timeout=local.startup_timeout,
            )
        model = self._provider_model(
            self.context_provider, self.context_model, settings, credentials, base_url_override=self.context_base_url
        )
        return ContextLLMConfig(chatbot=model, fee_limit=self.context_fee_limit)


_TRANSLATION_RECIPE_FIELDS = {
    "target_language",
    "translation_backend",
    "mode",
    "provider",
    "primary_model",
    "retry_provider",
    "retry_model",
    "reviewer_provider",
    "reviewer_model",
    "fee_limit",
    "consumer_thread",
    "local_profile",
    "qwen_model",
    "hymt2_model",
    "context_provider",
    "context_model",
    "context_base_url",
    "context_fee_limit",
    "context_assistance",
    "glossary_path",
    "glossary_strict",
    "force_glossary",
    "brief_summary",
    "brief_characters",
    "brief_tone_style",
    "edit_rounds",
    "enable_restore",
    "bilingual_subtitle",
    "clear_checkpoint",
}


def normalize_input_paths(raw_paths, workflow: str | WorkflowKind) -> tuple[list[str], list[str]]:
    """Normalize, validate, and de-duplicate input paths while preserving order."""
    kind = WorkflowKind(workflow)
    normalized: list[str] = []
    issues: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        value = str(raw_path).strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve(strict=False)
        identity = str(path)
        if identity in seen:
            issues.append(f"Duplicate ignored: {path}")
            continue
        seen.add(identity)
        normalized.append(identity)
        if not path.is_file():
            issues.append(f"Missing file: {path}")
            continue
        if kind is WorkflowKind.TRANSLATE:
            if path.suffix.lower() != ".json":
                issues.append(f"Translation requires a transcription JSON: {path}")
            continue
        try:
            from openlrc.media_utils import get_file_type

            get_file_type(path)
        except RuntimeError as exc:
            issues.append(str(exc))
    return normalized, issues
