"""Protocol parsing and safe error mapping for the desktop sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from openlrc.application.workflow_queue import QueueFullError
from openlrc.workflow import redact_sensitive_text

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_REQUEST_ID = 128
MAX_METHOD_LENGTH = 128


@dataclass(frozen=True, slots=True)
class ProtocolRequest:
    request_id: str
    method: str
    params: dict[str, object]


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.hint = hint

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": redact_sensitive_text(str(self)),
            "retryable": self.retryable,
            "hint": self.hint,
        }


def parse_request(payload: object) -> ProtocolRequest:
    if not isinstance(payload, dict):
        raise ProtocolError("INVALID_FRAME", "Protocol frame must be an object.")
    if payload.get("type") != "request":
        raise ProtocolError("INVALID_FRAME", "Protocol frame type must be request.")
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            "PROTOCOL_MISMATCH", f"Unsupported protocol {payload.get('protocol')!r}; expected {PROTOCOL_VERSION}."
        )
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params", {})
    if not isinstance(request_id, str) or not request_id or len(request_id) > MAX_REQUEST_ID:
        raise ProtocolError("INVALID_FRAME", "Request id must be a non-empty bounded string.")
    if not isinstance(method, str) or not method or len(method) > MAX_METHOD_LENGTH:
        raise ProtocolError("INVALID_FRAME", "Method must be a non-empty bounded string.")
    if not isinstance(params, dict):
        raise ProtocolError("INVALID_FRAME", "Request params must be an object.")
    return ProtocolRequest(request_id, method, cast(dict[str, object], params))


def error_from_exception(exc: Exception) -> ProtocolError:
    if isinstance(exc, ProtocolError):
        return exc
    if isinstance(exc, QueueFullError):
        return ProtocolError("QUEUE_FULL", str(exc), retryable=True)
    if isinstance(exc, KeyError):
        return ProtocolError("NOT_FOUND", str(exc))
    if isinstance(exc, (TypeError, ValueError)):
        return ProtocolError("INVALID_REQUEST", str(exc))
    return ProtocolError("INTERNAL_ERROR", str(exc) or type(exc).__name__, retryable=False)


def success_response(request_id: str, result: object) -> dict[str, object]:
    return {"type": "response", "protocol": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result}


def error_response(request_id: str, error: ProtocolError) -> dict[str, object]:
    return {"type": "response", "protocol": PROTOCOL_VERSION, "id": request_id, "ok": False, "error": error.to_dict()}
