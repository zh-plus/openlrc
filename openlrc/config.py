#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from openlrc.context import TranslationBriefInput, normalize_translation_brief_input
from openlrc.glossary import GlossaryCatalog
from openlrc.llama_resources import (
    DEFAULT_LLAMA_CONTEXT_SIZE,
    DEFAULT_LLAMA_HOST,
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_ALIAS,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_PORT,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    HY_MT2_7B_PROFILE,
    HY_MT2_30B_A3B_PROFILE,
    LOCAL_LLAMA_API_KEY,
    LocalLLMProfile,
    get_local_llm_profile,
    normalize_llama_model_name,
)
from openlrc.models import ModelConfig, ModelProvider
from openlrc.whisper_resources import DEFAULT_MODEL_NAME, DEFAULT_VAD_MODEL_NAME


@dataclass
class TranscriptionConfig:
    """
    Configuration for the transcription stage.

    Args:
        whisper_model: Path to whisper GGML model file, or semantic model filename.
            Default: ``ggml-base.bin``
        cli_path: Path to the whisper-cli executable. Empty string resolves automatically.
            Default: ``""``
        vad_model: Path to the Silero VAD model for whisper.cpp.
            Empty string disables native VAD. Default: ``ggml-silero-v6.2.0.bin``
        asr_options: Parameters for whisper.cpp CLI transcription.
    """

    whisper_model: str = DEFAULT_MODEL_NAME
    cli_path: str = ""
    vad_model: str = DEFAULT_VAD_MODEL_NAME
    asr_options: dict | None = None


@dataclass
class LocalLLMConfig:
    """
    Configuration for the local llama.cpp translation server.

    Args:
        enabled: Whether OpenLRC should manage a local llama-server.
        model_path: Path, filename, or supported alias for a GGUF model.
        server_path: Path to llama-server. Empty string resolves automatically.
        host: Host for the local server.
        port: Port for the local server.
        alias: OpenAI-compatible model alias served by llama-server.
        ctx_size: Context size passed to llama-server.
        gpu_layers: GPU layer offload setting for llama-server.
        idle_timeout: Seconds to keep an owned server alive after translation.
            Values <= 0 disable automatic idle shutdown.
        startup_timeout: Seconds to wait for llama-server readiness.
        extra_args: Additional llama-server CLI arguments.
    """

    enabled: bool = True
    model_path: str = DEFAULT_LLAMA_MODEL_FILE
    server_path: str = ""
    host: str = DEFAULT_LLAMA_HOST
    port: int = DEFAULT_LLAMA_PORT
    alias: str = DEFAULT_LLAMA_MODEL_ALIAS
    ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE
    gpu_layers: str | int = "all"
    idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT
    startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT
    extra_args: list[str] | None = None


class HyMT2Mode(StrEnum):
    """Execution modes for the Hy-MT2 translation pipeline."""

    FAST = "fast"
    NORMAL = "normal"
    NORMAL_PLUS = "normal-plus"
    # Deprecated public aliases retained for the 0.3.x compatibility window.
    CONTEXT = "context"
    CONTEXT_PLUS = "context-plus"
    PRO = "pro"

    @property
    def canonical(self) -> "HyMT2Mode":
        if self is HyMT2Mode.CONTEXT:
            return HyMT2Mode.NORMAL
        if self is HyMT2Mode.CONTEXT_PLUS:
            return HyMT2Mode.NORMAL_PLUS
        return self


class ContextAssistance(StrEnum):
    """Whether Hy-MT2 may use a context model to complete a Translation Brief."""

    AUTO = "auto"
    OFF = "off"


def normalize_hymt2_mode(mode: HyMT2Mode | str) -> HyMT2Mode:
    """Return the canonical product mode for public and checkpoint use."""
    selected = HyMT2Mode(mode)
    if selected in {HyMT2Mode.CONTEXT, HyMT2Mode.CONTEXT_PLUS}:
        warnings.warn(
            f"Hy-MT2 mode {selected.value!r} is deprecated; use {selected.canonical.value!r}.",
            FutureWarning,
            stacklevel=2,
        )
    return selected.canonical


class SubtitleOptimizationMode(StrEnum):
    """Subtitle cleanup profiles applied around translation."""

    AGGRESSIVE = "aggressive"
    RELAXED = "relaxed"


@dataclass
class GlossaryOptions:
    """Glossary validation and reporting behavior."""

    strict: bool = True
    force: bool = False
    report_matches: bool = True


@dataclass
class EditConfig:
    """Deterministic and semantic subtitle editing behavior."""

    enabled: bool = False
    max_rounds: int = 1
    semantic_review: bool = True
    deterministic_checks: bool = True
    restore_enabled: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_rounds, bool) or not 0 <= self.max_rounds <= 3:
            raise ValueError("EditConfig.max_rounds must be between 0 and 3.")


def context_model_required(
    *,
    mode: HyMT2Mode | str,
    translation_brief: TranslationBriefInput | dict | None,
    edit_config: EditConfig,
    context_assistance: ContextAssistance | str = ContextAssistance.AUTO,
) -> bool:
    """Resolve the single Hy-MT2 Context requirement used by every frontend."""
    selected_mode = normalize_hymt2_mode(mode)
    assistance = ContextAssistance(context_assistance)
    brief_input = normalize_translation_brief_input(translation_brief)

    if selected_mode is HyMT2Mode.FAST:
        return False

    if selected_mode is HyMT2Mode.PRO:
        if assistance is ContextAssistance.OFF:
            raise ValueError("Context assistance cannot be off in Hy-MT2 pro mode.")
        return True

    semantic_review_requires_context = bool(
        selected_mode is HyMT2Mode.NORMAL_PLUS and edit_config.semantic_review and edit_config.max_rounds > 0
    )
    if semantic_review_requires_context:
        if assistance is ContextAssistance.OFF:
            raise ValueError("Context assistance cannot be off while Normal Plus semantic review is enabled.")
        return True

    if assistance is ContextAssistance.OFF:
        if brief_input is None or not brief_input.is_complete:
            raise ValueError(
                "Context assistance off requires a complete manual Translation Brief "
                "(summary, characters, and tone/style)."
            )
        return False

    return brief_input is None or not brief_input.is_complete


@dataclass
class ContextLLMConfig:
    """General-purpose model used to prepare and review Hy-MT2 translations."""

    chatbot: ModelConfig
    local_llm: LocalLLMConfig | None = None
    fee_limit: float = 0.8

    @classmethod
    def online(
        cls,
        *,
        provider: ModelProvider | str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        fee_limit: float = 0.8,
        context_window: int | None = None,
        max_tokens: int | None = None,
    ) -> "ContextLLMConfig":
        """Create an online or externally managed context-model configuration."""
        return cls(
            chatbot=ModelConfig(
                provider=provider,
                name=model,
                base_url=base_url,
                api_key=api_key,
                context_window=context_window,
                max_tokens=max_tokens,
            ),
            fee_limit=fee_limit,
        )

    @classmethod
    def local_qwen35_9b(
        cls,
        *,
        model: str = "qwen3.5-9b",
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
    ) -> "ContextLLMConfig":
        """Create a managed local Qwen configuration for staged Hy-MT2 context work."""
        return cls(
            chatbot=ModelConfig(
                provider=ModelProvider.LOCAL_LLAMA,
                name=DEFAULT_LLAMA_MODEL_ALIAS,
                api_key=LOCAL_LLAMA_API_KEY,
                context_window=ctx_size,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            local_llm=LocalLLMConfig(
                model_path=normalize_llama_model_name(model),
                server_path=server_path,
                host=host,
                port=port,
                alias=DEFAULT_LLAMA_MODEL_ALIAS,
                ctx_size=ctx_size,
                gpu_layers=gpu_layers,
                idle_timeout=0,
                startup_timeout=startup_timeout,
                extra_args=extra_args,
            ),
            fee_limit=0.0,
        )


@dataclass
class TranslationConfig:
    """
    Configuration for the translation stage.

    All fields use primitive, serialization-friendly types so that the config
    can be parsed by CLI frameworks (simple_parsing, HfArgumentParser, Hydra)
    and serialized to JSON/YAML without custom encoders.

    For programmatic use with richer types (e.g. ``ModelConfig``), pass them
    directly to ``LRCer.__init__`` or ``LLMTranslator.__init__`` instead.

    Args:
        chatbot: Configuration for the primary chatbot model, or None.
            Default: ``None`` (which defaults to OpenAI's ``gpt-4.1-nano``)
        retry_chatbot: Configuration for the fallback chatbot model for translation retries, or None.
        cr_chatbot: Configuration for the Context Review chatbot model, or None.
            When None and lean mode is active, the primary ``chatbot`` is used for CR.
            The classic pipeline uses the primary chatbot when this is None.
        fee_limit: Maximum fee per translation call in USD. Default: ``0.8``
        consumer_thread: Number of parallel translation threads. Default: ``4``
        glossary: Path to a JSON glossary file mapping source words to
            translations, or None.
        translation_brief: Optional fixed summary, character mappings, and
            tone/style sections for a full or partial Hy-MT2 Translation Brief.
            An empty mapping is treated as no manual Brief.
        is_force_glossary_used: Force glossary usage in context. Default: ``False``
        enable_cr: Whether to run Context Review in lean mode.
            Default: ``True``. The classic pipeline always runs Context Review.
        chunked_guideline: Enable chunked guideline generation for long texts.
            When True, texts exceeding the CR model's context window are
            automatically split and merged. Default: ``False``
        prompt_profile: Prompt family used by lean translation and atomic
            fallback. Default: ``"default"``.
        local_llm: Local llama.cpp server configuration. When provided and
            enabled, OpenLRC starts or reuses a local OpenAI-compatible server.
    """

    chatbot: ModelConfig | None = None
    retry_chatbot: ModelConfig | None = None
    cr_chatbot: ModelConfig | None = None
    fee_limit: float = 0.8
    consumer_thread: int = 4
    glossary: dict | str | Path | GlossaryCatalog | None = None
    is_force_glossary_used: bool = False
    glossary_options: GlossaryOptions = field(default_factory=GlossaryOptions)
    edit_config: EditConfig = field(default_factory=EditConfig)
    translation_brief: TranslationBriefInput | dict | None = None
    enable_cr: bool = True
    chunked_guideline: bool = False
    prompt_profile: str = "default"
    local_llm: LocalLLMConfig | None = None
    hy_mt2_mode: HyMT2Mode = HyMT2Mode.FAST
    context_llm: ContextLLMConfig | None = None
    context_assistance: ContextAssistance = ContextAssistance.AUTO
    _translator_engine: str = field(default="classic", init=False, repr=False)

    def __post_init__(self) -> None:
        self.context_assistance = ContextAssistance(self.context_assistance)
        if self.context_assistance is ContextAssistance.OFF and self.context_llm is not None:
            raise ValueError("Context assistance off conflicts with an explicit context_llm configuration.")

    @classmethod
    def local_qwen35_9b(
        cls,
        *,
        model: str = "qwen3.5-9b",
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
        glossary: dict | str | Path | GlossaryCatalog | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
    ) -> "TranslationConfig":
        """Create the recommended local Qwen3.5 9B translation configuration."""
        local_llm = LocalLLMConfig(
            model_path=normalize_llama_model_name(model),
            server_path=server_path,
            host=host,
            port=port,
            alias=DEFAULT_LLAMA_MODEL_ALIAS,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            idle_timeout=idle_timeout,
            startup_timeout=startup_timeout,
            extra_args=extra_args,
        )
        return cls(
            chatbot=ModelConfig(
                provider=ModelProvider.LOCAL_LLAMA,
                name=DEFAULT_LLAMA_MODEL_ALIAS,
                api_key=LOCAL_LLAMA_API_KEY,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            fee_limit=0.0,
            consumer_thread=1,
            enable_cr=True,
            local_llm=local_llm,
            glossary=glossary,
            glossary_options=glossary_options or GlossaryOptions(),
            edit_config=edit_config or EditConfig(),
        )

    @classmethod
    def local_hy_mt2_7b(
        cls,
        *,
        model: str | None = None,
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
        mode: HyMT2Mode | str = HyMT2Mode.FAST,
        context_llm: ContextLLMConfig | None = None,
        context_assistance: ContextAssistance | str = ContextAssistance.AUTO,
        glossary: dict | str | Path | GlossaryCatalog | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
        translation_brief: TranslationBriefInput | dict | None = None,
    ) -> "TranslationConfig":
        """Create the recommended local Hy-MT2 7B Q6_K translation configuration."""
        return cls.local_hy_mt2(
            size=HY_MT2_7B_PROFILE,
            model=model,
            idle_timeout=idle_timeout,
            port=port,
            server_path=server_path,
            host=host,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            startup_timeout=startup_timeout,
            extra_args=extra_args,
            mode=mode,
            context_llm=context_llm,
            context_assistance=context_assistance,
            glossary=glossary,
            glossary_options=glossary_options,
            edit_config=edit_config,
            translation_brief=translation_brief,
        )

    @classmethod
    def local_hy_mt2(
        cls,
        *,
        size: str = HY_MT2_7B_PROFILE,
        model: str | None = None,
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
        mode: HyMT2Mode | str = HyMT2Mode.FAST,
        context_llm: ContextLLMConfig | None = None,
        context_assistance: ContextAssistance | str = ContextAssistance.AUTO,
        glossary: dict | str | Path | GlossaryCatalog | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
        translation_brief: TranslationBriefInput | dict | None = None,
    ) -> "TranslationConfig":
        """Create a local Hy-MT2 translation configuration.

        The 30B-A3B profile intentionally has no bundled/default GGUF file.
        Pass ``model`` as a local converted GGUF filename or path for that size.
        """
        profile = get_local_llm_profile(size)
        mode = normalize_hymt2_mode(mode)
        assistance = ContextAssistance(context_assistance)
        brief_input = normalize_translation_brief_input(translation_brief)
        if mode is HyMT2Mode.FAST and brief_input is not None:
            raise ValueError("Hy-MT2 fast mode does not use a Translation Brief.")
        resolved_edit_config = edit_config or EditConfig(enabled=mode in {HyMT2Mode.NORMAL_PLUS, HyMT2Mode.PRO})
        needs_context = context_model_required(
            mode=mode, translation_brief=brief_input, edit_config=resolved_edit_config, context_assistance=assistance
        )
        if assistance is ContextAssistance.OFF and context_llm is not None:
            raise ValueError("Context assistance off conflicts with an explicit context_llm configuration.")
        if needs_context and context_llm is None:
            raise ValueError(f"Hy-MT2 {mode.value!r} mode requires an explicit context_llm configuration.")
        if profile.name == HY_MT2_30B_A3B_PROFILE and not model:
            raise ValueError("Hy-MT2 30B-A3B requires an explicit local GGUF model path or filename.")

        model_path = model or profile.model_file
        if not model_path:
            raise ValueError(f"Local LLM profile {profile.name!r} does not define a default GGUF model.")

        local_llm = LocalLLMConfig(
            model_path=normalize_llama_model_name(model_path),
            server_path=server_path,
            host=host,
            port=port,
            alias=profile.server_alias,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            idle_timeout=idle_timeout,
            startup_timeout=startup_timeout,
            extra_args=extra_args,
        )
        config = cls(
            chatbot=_local_profile_model_config(profile, ctx_size=ctx_size),
            fee_limit=0.0,
            consumer_thread=1,
            enable_cr=False,
            prompt_profile=profile.prompt_profile,
            local_llm=local_llm,
            hy_mt2_mode=mode,
            context_llm=context_llm,
            context_assistance=assistance,
            glossary=glossary,
            glossary_options=glossary_options or GlossaryOptions(),
            edit_config=resolved_edit_config,
            translation_brief=brief_input,
        )
        config._translator_engine = "lean"
        return config


def _local_profile_model_config(profile: LocalLLMProfile, *, ctx_size: int) -> ModelConfig:
    extra_body: dict[str, object] = {}
    if profile.top_k is not None:
        extra_body["top_k"] = profile.top_k
    if profile.repeat_penalty is not None:
        extra_body["repeat_penalty"] = profile.repeat_penalty

    return ModelConfig(
        provider=ModelProvider.LOCAL_LLAMA,
        name=profile.server_alias,
        api_key=LOCAL_LLAMA_API_KEY,
        context_window=ctx_size,
        max_tokens=profile.max_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        extra_body=extra_body or None,
    )
