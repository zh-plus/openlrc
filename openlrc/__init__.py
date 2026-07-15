#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openlrc.config import ContextLLMConfig, HyMT2Mode, LocalLLMConfig, TranscriptionConfig, TranslationConfig
    from openlrc.models import ModelConfig, ModelProvider, list_chatbot_models
    from openlrc.openlrc import LRCer


__all__ = (
    "LRCer",
    "TranscriptionConfig",
    "TranslationConfig",
    "LocalLLMConfig",
    "ContextLLMConfig",
    "HyMT2Mode",
    "ModelConfig",
    "list_chatbot_models",
    "ModelProvider",
)
__version__ = "0.2.2"
__upstream_version__ = "1.7.0a1"
__app_name__ = "OpenLRC Mac"
__dist_name__ = "openlrc-mac"
__author__ = "OpenLRC Mac contributors"

_LAZY_EXPORTS = {
    "LRCer": ("openlrc.openlrc", "LRCer"),
    "TranscriptionConfig": ("openlrc.config", "TranscriptionConfig"),
    "TranslationConfig": ("openlrc.config", "TranslationConfig"),
    "LocalLLMConfig": ("openlrc.config", "LocalLLMConfig"),
    "ContextLLMConfig": ("openlrc.config", "ContextLLMConfig"),
    "HyMT2Mode": ("openlrc.config", "HyMT2Mode"),
    "ModelConfig": ("openlrc.models", "ModelConfig"),
    "ModelProvider": ("openlrc.models", "ModelProvider"),
    "list_chatbot_models": ("openlrc.models", "list_chatbot_models"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
