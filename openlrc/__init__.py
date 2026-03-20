#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openlrc.config import TranscriptionConfig, TranslationConfig
    from openlrc.models import ModelConfig, ModelProvider
    from openlrc.openlrc import LRCer


__all__ = ("LRCer", "TranscriptionConfig", "TranslationConfig", "ModelConfig", "list_chatbot_models", "ModelProvider")
__version__ = "1.6.2"
__author__ = "zh-plus"

_LAZY_EXPORTS = {
    "LRCer": ("openlrc.openlrc", "LRCer"),
    "TranscriptionConfig": ("openlrc.config", "TranscriptionConfig"),
    "TranslationConfig": ("openlrc.config", "TranslationConfig"),
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
