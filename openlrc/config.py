#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass


@dataclass
class TranscriptionConfig:
    """
    Configuration for the transcription stage.

    Args:
        whisper_model: Name of whisper model. Default: ``large-v3``
        compute_type: Computation type (``default``, ``int8``, ``int8_float16``,
            ``int16``, ``float16``, ``float32``). Default: ``float16``
        device: Device for computation. Default: ``cuda``
        asr_options: Parameters for whisper model.
        vad_options: Parameters for VAD model.
        preprocess_options: Options for audio preprocessing.
    """

    whisper_model: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    asr_options: dict | None = None
    vad_options: dict | None = None
    preprocess_options: dict | None = None


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
        chatbot_model: The chatbot model to use, as a string like
            ``'gpt-4.1-nano'`` or ``'provider:model-name'``.
            Default: ``gpt-4.1-nano``
        fee_limit: Maximum fee per translation call in USD. Default: ``0.8``
        consumer_thread: Number of parallel translation threads. Default: ``4``
        proxy: Proxy for API requests. e.g. ``'http://127.0.0.1:7890'``
        base_url_config: Base URL dict for OpenAI & Anthropic.
        glossary: Path to a JSON glossary file mapping source words to
            translations, or None. To pass a dict directly, use the
            ``LRCer`` or ``LLMTranslator`` API instead.
        retry_model: Fallback model name for translation retries, or None.
        is_force_glossary_used: Force glossary usage in context. Default: ``False``
    """

    chatbot_model: str = "gpt-4.1-nano"
    fee_limit: float = 0.8
    consumer_thread: int = 4
    proxy: str | None = None
    base_url_config: dict | None = None
    glossary: str | None = None
    retry_model: str | None = None
    is_force_glossary_used: bool = False
