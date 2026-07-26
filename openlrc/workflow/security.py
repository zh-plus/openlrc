"""Shared redaction helpers for user-facing workflow and application messages."""

from __future__ import annotations

import re


def redact_sensitive_text(message: str) -> str:
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<redacted>", message)
    message = re.sub(r"\bAIza[A-Za-z0-9_-]{8,}\b", "<redacted>", message)
    message = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*)(bearer|basic|token)(\s+)([^\s,;]+)", r"\1\2\3<redacted>", message
    )
    message = re.sub(r"(?i)(\bauthorization\s*[:=]\s*)(?!(?:bearer|basic|token)\b)([^\s,;]+)", r"\1<redacted>", message)
    message = re.sub(
        r"(?i)(\b(?:x[-_ ]?api[-_ ]?key|api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token)\b)"
        r"(\s*[:=]\s*)([^\s,;&#]+)",
        r"\1\2<redacted>",
        message,
    )
    return re.sub(
        r"(?i)([?&](?:key|api[_-]?key|access[_-]?token|refresh[_-]?token|token)=)([^&#\s]+)", r"\1<redacted>", message
    )
