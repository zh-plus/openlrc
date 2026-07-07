#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass

from openlrc.llama_resources import (
    DEFAULT_LLAMA_CONTEXT_SIZE,
    DEFAULT_LLAMA_HOST,
    DEFAULT_LLAMA_IDLE_TIMEOUT,
    DEFAULT_LLAMA_MODEL_ALIAS,
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_PORT,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    LOCAL_LLAMA_API_KEY,
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
        preprocess_options: Options for audio preprocessing.
    """

    whisper_model: str = DEFAULT_MODEL_NAME
    cli_path: str = ""
    vad_model: str = DEFAULT_VAD_MODEL_NAME
    asr_options: dict | None = None
    preprocess_options: dict | None = None


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
            Ignored in standard mode.
        fee_limit: Maximum fee per translation call in USD. Default: ``0.8``
        consumer_thread: Number of parallel translation threads. Default: ``4``
        glossary: Path to a JSON glossary file mapping source words to
            translations, or None.
        is_force_glossary_used: Force glossary usage in context. Default: ``False``
        translate_mode: Translation strategy. ``"standard"`` uses
            :class:`LLMTranslator`, ``"lean"`` uses :class:`LeanTranslator`.
            Default: ``"standard"``
        enable_cr: Whether to run Context Review in lean mode.
            Default: ``True``. Ignored in standard mode.
        chunked_guideline: Enable chunked guideline generation for long texts.
            When True, texts exceeding the CR model's context window are
            automatically split and merged. Default: ``False``
        local_llm: Local llama.cpp server configuration. When provided and
            enabled, OpenLRC starts or reuses a local OpenAI-compatible server.
    """

    chatbot: ModelConfig | None = None
    retry_chatbot: ModelConfig | None = None
    cr_chatbot: ModelConfig | None = None
    fee_limit: float = 0.8
    consumer_thread: int = 4
    glossary: str | None = None
    is_force_glossary_used: bool = False
    translate_mode: str = "standard"
    enable_cr: bool = True
    chunked_guideline: bool = False
    local_llm: LocalLLMConfig | None = None

    @classmethod
    def local_qwen35_9b(
        cls,
        *,
        model: str = "qwen3.5-9b",
        idle_timeout: int = DEFAULT_LLAMA_IDLE_TIMEOUT,
        port: int = DEFAULT_LLAMA_PORT,
        translate_mode: str = "lean",
        server_path: str = "",
        host: str = DEFAULT_LLAMA_HOST,
        ctx_size: int = DEFAULT_LLAMA_CONTEXT_SIZE,
        gpu_layers: str | int = "all",
        startup_timeout: int = DEFAULT_LLAMA_STARTUP_TIMEOUT,
        extra_args: list[str] | None = None,
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
            translate_mode=translate_mode,
            enable_cr=False,
            local_llm=local_llm,
        )
