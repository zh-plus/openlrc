#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass

from openlrc.models import ModelConfig
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
