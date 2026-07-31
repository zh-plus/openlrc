"""Typed, cancellable setup operations for interactive frontends."""

from __future__ import annotations

import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from openlrc.application.operations import OperationGuard
from openlrc.llama_resources import DEFAULT_LLAMA_MODEL_FILE, DEFAULT_LLAMA_MODEL_REPO
from openlrc.setup.llama_cpp import LlamaSetupResult, setup_llama_cpp
from openlrc.setup.whisper_cpp import DEFAULT_MODEL, DEFAULT_VAD_MODEL, WhisperSetupResult, setup_whisper_cpp
from openlrc.workflow import CancellationToken, OwnedProcessRegistry, WorkflowCancelled, redact_sensitive_text


class SetupKind(StrEnum):
    WHISPER = "whisper"
    LLAMA = "llama"
    ALL = "all"


class SetupStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WhisperSetupRequest:
    model: str = DEFAULT_MODEL
    vad_model: str = DEFAULT_VAD_MODEL
    model_dir: Path | None = None
    skip_build: bool = False
    skip_models: bool = False
    kind: SetupKind = field(default=SetupKind.WHISPER, init=False)


@dataclass(frozen=True, slots=True)
class LlamaSetupRequest:
    model_repo: str = DEFAULT_LLAMA_MODEL_REPO
    model_file: str = DEFAULT_LLAMA_MODEL_FILE
    revision: str = "main"
    model_url: str | None = None
    model_dir: Path | None = None
    skip_build: bool = False
    skip_models: bool = False
    force: bool = False
    kind: SetupKind = field(default=SetupKind.LLAMA, init=False)


@dataclass(frozen=True, slots=True)
class SetupAllRequest:
    whisper: WhisperSetupRequest = field(default_factory=WhisperSetupRequest)
    llama: LlamaSetupRequest = field(default_factory=LlamaSetupRequest)
    kind: SetupKind = field(default=SetupKind.ALL, init=False)


SetupRequest: TypeAlias = WhisperSetupRequest | LlamaSetupRequest | SetupAllRequest


@dataclass(frozen=True, slots=True)
class SetupEventHeader:
    operation_id: str
    sequence: int
    kind: SetupKind


@dataclass(frozen=True, slots=True)
class SetupStartedEvent:
    header: SetupEventHeader


@dataclass(frozen=True, slots=True)
class SetupStageEvent:
    header: SetupEventHeader
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class SetupLogEvent:
    header: SetupEventHeader
    message: str


@dataclass(frozen=True, slots=True)
class SetupCompletedEvent:
    header: SetupEventHeader


@dataclass(frozen=True, slots=True)
class SetupFailedEvent:
    header: SetupEventHeader
    message: str


@dataclass(frozen=True, slots=True)
class SetupCancelledEvent:
    header: SetupEventHeader
    message: str = "Setup cancelled."


SetupEvent: TypeAlias = (
    SetupStartedEvent | SetupStageEvent | SetupLogEvent | SetupCompletedEvent | SetupFailedEvent | SetupCancelledEvent
)
SetupEventCallback = Callable[[SetupEvent], None]


@dataclass(frozen=True, slots=True)
class SetupResult:
    operation_id: str
    kind: SetupKind
    status: SetupStatus
    paths: tuple[Path, ...] = ()
    error: str | None = None


class SetupController:
    """Run setup services with structured stages and owned-process cancellation."""

    def __init__(self, *, operation_guard: OperationGuard | None = None) -> None:
        self.operation_guard = operation_guard or OperationGuard()
        self._lock = threading.RLock()
        self._active_token: CancellationToken | None = None
        self._active_kind: SetupKind | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_token is not None

    @property
    def active_kind(self) -> SetupKind | None:
        with self._lock:
            return self._active_kind

    def cancel(self) -> bool:
        with self._lock:
            token = self._active_token
        if token is None:
            return False
        token.cancel()
        return True

    def run(
        self,
        request: SetupRequest,
        *,
        on_event: SetupEventCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> SetupResult:
        operation_id = str(uuid.uuid4())
        token = cancellation_token or CancellationToken()
        processes = OwnedProcessRegistry()
        remove_callback = token.add_callback(processes.terminate_all)
        sequence = 0

        def emit(event_type, *args) -> None:
            nonlocal sequence
            sequence += 1
            if on_event is None:
                return
            event = event_type(SetupEventHeader(operation_id, sequence, request.kind), *args)
            try:
                on_event(event)
            except Exception:
                pass

        def runner(command: list[str], cwd: Path) -> None:
            token.raise_if_cancelled()
            process = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
            )
            processes.register(process, process_group=True)
            try:
                token.raise_if_cancelled()
                emit(SetupLogEvent, "$ " + redact_sensitive_text(" ".join(command)))
                assert process.stdout is not None
                for line in process.stdout:
                    stripped = line.rstrip()
                    if stripped:
                        emit(SetupLogEvent, redact_sensitive_text(stripped))
                return_code = process.wait()
            finally:
                processes.unregister(process)
            token.raise_if_cancelled()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

        with self._lock:
            if self._active_token is not None:
                remove_callback()
                raise RuntimeError("Another setup operation is already running.")
            self._active_token = token
            self._active_kind = request.kind

        try:
            with self.operation_guard.acquire("setup", f"{request.kind.value} setup"):
                token.raise_if_cancelled()
                emit(SetupStartedEvent)
                paths = self._execute(request, runner, emit)
                token.raise_if_cancelled()
                emit(SetupCompletedEvent)
                return SetupResult(operation_id, request.kind, SetupStatus.SUCCEEDED, paths)
        except WorkflowCancelled:
            emit(SetupCancelledEvent)
            return SetupResult(operation_id, request.kind, SetupStatus.CANCELLED)
        except Exception as exc:
            message = redact_sensitive_text(str(exc) or type(exc).__name__)
            emit(SetupFailedEvent, message)
            return SetupResult(operation_id, request.kind, SetupStatus.FAILED, error=message)
        finally:
            remove_callback()
            processes.terminate_all()
            with self._lock:
                self._active_token = None
                self._active_kind = None

    @staticmethod
    def _execute(request: SetupRequest, runner, emit) -> tuple[Path, ...]:
        results: list[Path] = []

        def whisper(selected: WhisperSetupRequest) -> None:
            emit(SetupStageEvent, "whisper", "Preparing whisper.cpp and transcription models")
            result = setup_whisper_cpp(
                model=selected.model,
                vad_model=selected.vad_model,
                model_dir=selected.model_dir,
                skip_build=selected.skip_build,
                skip_models=selected.skip_models,
                runner=runner,
            )
            results.extend(_whisper_paths(result))

        def llama(selected: LlamaSetupRequest) -> None:
            emit(SetupStageEvent, "llama", "Preparing llama.cpp and local translation model")
            result = setup_llama_cpp(
                model_repo=selected.model_repo,
                model_file=selected.model_file,
                revision=selected.revision,
                model_url=selected.model_url,
                model_dir=selected.model_dir,
                skip_build=selected.skip_build,
                skip_models=selected.skip_models,
                force=selected.force,
                runner=runner,
            )
            results.extend(_llama_paths(result))

        if isinstance(request, WhisperSetupRequest):
            whisper(request)
        elif isinstance(request, LlamaSetupRequest):
            llama(request)
        else:
            whisper(request.whisper)
            llama(request.llama)
        return tuple(results)


def _whisper_paths(result: WhisperSetupResult) -> tuple[Path, ...]:
    return tuple(path for path in (result.cli_path, result.whisper_model, result.vad_model) if path is not None)


def _llama_paths(result: LlamaSetupResult) -> tuple[Path, ...]:
    return tuple(path for path in (result.server_path, result.cli_path, result.model_path) if path is not None)
