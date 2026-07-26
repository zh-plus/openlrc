"""Shared configuration factories for CLI, TUI, and future GUI clients."""

from __future__ import annotations

from pathlib import Path

from openlrc.config import ContextLLMConfig, EditConfig, GlossaryOptions, HyMT2Mode, TranslationConfig
from openlrc.context import TranslationBriefInput
from openlrc.llama_resources import (
    DEFAULT_LLAMA_CONTEXT_SIZE,
    DEFAULT_LLAMA_HOST,
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_PORT,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    HY_MT2_7B_PROFILE,
)
from openlrc.models import ModelConfig, ModelProvider
from openlrc.workflow.types import RunExecutionStrategy, RunRequest, TranslationMode, WorkflowTranslationConfig


class WorkflowTranslationFactory:
    """Build validated product-mode configurations without constructing an ``LRCer``."""

    @staticmethod
    def standard_online(
        *,
        primary: ModelConfig | None = None,
        retry: ModelConfig | None = None,
        reviewer: ModelConfig | None = None,
        fee_limit: float = 0.8,
        consumer_thread: int = 4,
        glossary: dict | str | Path | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
        chunked_guideline: bool = False,
    ) -> WorkflowTranslationConfig:
        config = TranslationConfig(
            chatbot=primary or ModelConfig(provider=ModelProvider.OPENAI, name="gpt-4.1-nano"),
            retry_chatbot=retry,
            cr_chatbot=reviewer,
            fee_limit=fee_limit,
            consumer_thread=consumer_thread,
            glossary=glossary,
            glossary_options=glossary_options or GlossaryOptions(),
            edit_config=edit_config or EditConfig(),
            chunked_guideline=chunked_guideline,
        )
        return WorkflowTranslationConfig(mode=TranslationMode.STANDARD, config=config)

    @staticmethod
    def standard_local_qwen(
        *,
        model: str = "qwen3.5-9b",
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        glossary: dict | str | Path | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
    ) -> WorkflowTranslationConfig:
        config = TranslationConfig.local_qwen35_9b(
            model=model,
            idle_timeout=idle_timeout,
            port=port,
            server_path=server_path,
            host=host,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            startup_timeout=startup_timeout,
            glossary=glossary,
            glossary_options=glossary_options,
            edit_config=edit_config,
        )
        return WorkflowTranslationConfig(mode=TranslationMode.STANDARD, config=config)

    @staticmethod
    def hymt2(
        *,
        mode: TranslationMode,
        profile: str = HY_MT2_7B_PROFILE,
        model: str | None = None,
        context_llm: ContextLLMConfig | None = None,
        translation_brief: TranslationBriefInput | None = None,
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        glossary: dict | str | Path | None = None,
        glossary_options: GlossaryOptions | None = None,
        edit_config: EditConfig | None = None,
    ) -> WorkflowTranslationConfig:
        mode = TranslationMode(mode)
        mode_map = {
            TranslationMode.FAST: HyMT2Mode.FAST,
            TranslationMode.NORMAL: HyMT2Mode.NORMAL,
            TranslationMode.NORMAL_PLUS: HyMT2Mode.NORMAL_PLUS,
            TranslationMode.PRO: HyMT2Mode.PRO,
        }
        if mode not in mode_map:
            raise ValueError("Hy-MT2 factory requires fast, normal, normal-plus, or pro mode.")
        config = TranslationConfig.local_hy_mt2(
            size=profile,
            model=model,
            idle_timeout=idle_timeout,
            port=port,
            server_path=server_path,
            host=host,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            startup_timeout=startup_timeout,
            mode=mode_map[mode],
            context_llm=context_llm,
            glossary=glossary,
            glossary_options=glossary_options,
            edit_config=edit_config,
            translation_brief=translation_brief,
        )
        return WorkflowTranslationConfig(mode=mode, config=config)


def resolve_run_execution_strategy(request: RunRequest) -> RunExecutionStrategy:
    """Resolve the fixed multi-file scheduling rule for a Workflow Run."""
    if request.translation is None:
        return RunExecutionStrategy.TRANSCRIBE_ONLY
    config = request.translation.config
    managed_local = bool(config.local_llm is not None and config.local_llm.enabled)
    local_provider = bool(config.chatbot is not None and config.chatbot.provider is ModelProvider.LOCAL_LLAMA)
    if managed_local or local_provider:
        return RunExecutionStrategy.MEMORY_SAVER
    return RunExecutionStrategy.PIPELINE
