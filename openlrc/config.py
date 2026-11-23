#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass

from openlrc.models import ModelConfig


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
        chatbot: Configuration for the primary chatbot model, or None.
            Default: ``None`` (uses ``gpt-5.6-luna`` for native OpenAI; compatible gateways retain
            the previous default unless a model is specified).
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
