"""Cancellation, event delivery, and owned-process lifecycle support."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from openlrc.workflow.types import (
    ArtifactCreatedEvent,
    EventHeader,
    EventSink,
    LogMessageEvent,
    ModelLifecycleEvent,
    StageCompletedEvent,
    StageOutcome,
    StageProgressEvent,
    StageStartedEvent,
    TranslationMode,
    WorkflowArtifact,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowError,
    WorkflowEvent,
    WorkflowFailedEvent,
    WorkflowKind,
    WorkflowStage,
    WorkflowStartedEvent,
    WorkflowStatus,
)


class WorkflowCancelled(RuntimeError):
    """Raised at a cooperative cancellation boundary."""


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback = 0

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = tuple(self._callbacks.values())
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise WorkflowCancelled("Workflow cancelled.")

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation and return True when cancelled."""
        return self._event.wait(timeout)

    def wait_or_raise(self, timeout: float) -> None:
        if self._event.wait(timeout):
            raise WorkflowCancelled("Workflow cancelled.")

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            if self._event.is_set():
                call_now = True
                callback_id = -1
            else:
                call_now = False
                callback_id = self._next_callback
                self._next_callback += 1
                self._callbacks[callback_id] = callback
        if call_now:
            callback()

        def remove() -> None:
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove


class OwnedProcessRegistry:
    """Tracks only subprocesses that the current workflow owns."""

    def __init__(self, terminate_timeout: float = 5.0) -> None:
        self._terminate_timeout = terminate_timeout
        self._lock = threading.Lock()
        self._processes: dict[subprocess.Popen, bool] = {}

    def register(self, process: subprocess.Popen, *, process_group: bool = False) -> None:
        with self._lock:
            self._processes[process] = process_group

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.pop(process, None)

    def terminate(self, process: subprocess.Popen) -> None:
        with self._lock:
            process_group = self._processes.get(process, False)
        if process.poll() is not None:
            self.unregister(process)
            return
        try:
            self._signal(process, process_group=process_group, force=False)
            process.wait(timeout=self._terminate_timeout)
        except subprocess.TimeoutExpired:
            self._signal(process, process_group=process_group, force=True)
            try:
                process.wait()
            except ChildProcessError:
                pass
        except (ChildProcessError, ProcessLookupError):
            pass
        finally:
            self.unregister(process)

    @staticmethod
    def _signal(process: subprocess.Popen, *, process_group: bool, force: bool) -> None:
        if process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except OSError:
                # Fall back to the direct child if process-group signalling is
                # unavailable; this still guarantees that the registered root
                # process participates in terminate/kill/wait cleanup.
                pass
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except OSError:
            pass

    def terminate_all(self) -> None:
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            self.terminate(process)


class ExecutionContext:
    """Single-use per-execution runtime shared by the workflow and core pipeline."""

    def __init__(
        self,
        workflow: WorkflowKind,
        *,
        event_sink: EventSink | None = None,
        cancellation_token: CancellationToken | None = None,
        job_id: str | None = None,
        translation_mode: TranslationMode | None = None,
    ) -> None:
        self.workflow = WorkflowKind(workflow)
        self.job_id = job_id or str(uuid.uuid4())
        self.translation_mode = translation_mode
        self.event_sink = event_sink
        self.cancellation_token = cancellation_token or CancellationToken()
        self.processes = OwnedProcessRegistry()
        self._lifecycle_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._sequence = 0
        self._event_lock = threading.RLock()
        self._terminal_lock = threading.Lock()
        self._terminal_sent = False
        self._artifacts_lock = threading.Lock()
        self._artifacts: list[WorkflowArtifact] = []
        self._owned_paths_lock = threading.Lock()
        self._owned_paths: set[Path] = set()
        self._failure_lock = threading.Lock()
        self.failed_stage: WorkflowStage | None = None
        self.failed_item: str | None = None
        self._state = threading.local()
        self._remove_cancel_callback = self.cancellation_token.add_callback(self.processes.terminate_all)

    @property
    def current_stage(self) -> WorkflowStage | None:
        return getattr(self._state, "stage", None)

    @property
    def current_item(self) -> str | None:
        return getattr(self._state, "item", None)

    @property
    def artifacts(self) -> tuple[WorkflowArtifact, ...]:
        with self._artifacts_lock:
            return tuple(self._artifacts)

    @property
    def owned_paths(self) -> frozenset[Path]:
        """Return the runtime-only files created by this execution."""
        with self._owned_paths_lock:
            return frozenset(self._owned_paths)

    def register_owned_path(self, path: str | Path) -> Path:
        """Mark a file as safe for this execution to clean up later."""
        resolved = Path(path).expanduser().resolve(strict=False)
        with self._owned_paths_lock:
            self._owned_paths.add(resolved)
        return resolved

    def unregister_owned_path(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve(strict=False)
        with self._owned_paths_lock:
            self._owned_paths.discard(resolved)

    def owns_path(self, path: str | Path) -> bool:
        resolved = Path(path).expanduser().resolve(strict=False)
        with self._owned_paths_lock:
            return resolved in self._owned_paths

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            remove_cancel_callback = self._remove_cancel_callback
        try:
            self.processes.terminate_all()
        finally:
            remove_cancel_callback()

    def check_cancelled(self) -> None:
        self.cancellation_token.raise_if_cancelled()

    def _publish(self, build_event: Callable[[EventHeader], WorkflowEvent], *, item: str | Path | None = None) -> None:
        # Allocate sequence numbers and invoke the sink under one re-entrant
        # lock. This guarantees observed delivery order across workers without
        # leaving a pending sequence gap if cancellation interrupts a caller.
        with self._event_lock:
            self._sequence += 1
            sequence = self._sequence
            resolved_item = self.current_item if item is None else str(item)
            header = EventHeader(
                job_id=self.job_id,
                sequence=sequence,
                timestamp=datetime.now(timezone.utc),
                workflow=self.workflow,
                item=resolved_item,
                translation_mode=self.translation_mode,
            )
            event = build_event(header)
            if self.event_sink is not None:
                try:
                    self.event_sink(event)
                except Exception:
                    # UI/event consumers must never be able to abort subtitle work.
                    pass

    def workflow_started(self) -> None:
        with self._lifecycle_lock:
            if self._started or self._closed:
                raise RuntimeError("ExecutionContext is single-use and has already started or closed.")
            self._started = True
        self._publish(WorkflowStartedEvent)

    def workflow_completed(self, status: WorkflowStatus, elapsed_seconds: float) -> None:
        with self._terminal_lock:
            if self._terminal_sent:
                return
            self._terminal_sent = True
        self._publish(lambda header: WorkflowCompletedEvent(header, status, elapsed_seconds))

    def workflow_failed(self, error: WorkflowError) -> None:
        with self._terminal_lock:
            if self._terminal_sent:
                return
            self._terminal_sent = True
        self._publish(lambda header: WorkflowFailedEvent(header, error), item=error.item)

    def workflow_cancelled(self, message: str = "Workflow cancelled.") -> None:
        with self._terminal_lock:
            if self._terminal_sent:
                return
            self._terminal_sent = True
        self._publish(lambda header: WorkflowCancelledEvent(header, message))

    @contextmanager
    def item(self, item: str | Path | None) -> Iterator[None]:
        previous = self.current_item
        self._state.item = None if item is None else str(item)
        try:
            yield
        finally:
            self._state.item = previous

    @contextmanager
    def stage(self, stage: WorkflowStage, *, item: str | Path | None = None) -> Iterator[None]:
        self.check_cancelled()
        previous_stage = self.current_stage
        previous_item = self.current_item
        self._state.stage = WorkflowStage(stage)
        if item is not None:
            self._state.item = str(item)
        self._publish(lambda header: StageStartedEvent(header, WorkflowStage(stage)))
        try:
            yield
            self.check_cancelled()
        except Exception:
            with self._failure_lock:
                if self.failed_stage is None:
                    self.failed_stage = WorkflowStage(stage)
                    self.failed_item = self.current_item
            raise
        else:
            self._publish(lambda header: StageCompletedEvent(header, WorkflowStage(stage)))
        finally:
            self._state.stage = previous_stage
            self._state.item = previous_item

    def stage_progress(
        self,
        stage: WorkflowStage,
        completed: float,
        total: float,
        *,
        item: str | Path | None = None,
        message: str | None = None,
    ) -> None:
        self.check_cancelled()
        self._publish(
            lambda header: StageProgressEvent(header, WorkflowStage(stage), completed, total, message), item=item
        )

    def stage_completed(
        self,
        stage: WorkflowStage,
        *,
        outcome: StageOutcome = StageOutcome.COMPLETED,
        item: str | Path | None = None,
        message: str | None = None,
    ) -> None:
        self._publish(lambda header: StageCompletedEvent(header, WorkflowStage(stage), outcome, message), item=item)

    def model_event(
        self, model: str, state: str, *, owned: bool | None = None, role: str | None = None, endpoint: str | None = None
    ) -> None:
        self._publish(lambda header: ModelLifecycleEvent(header, model, state, owned, role, endpoint))

    def artifact_created(self, artifact: WorkflowArtifact) -> None:
        with self._artifacts_lock:
            if any(existing.path == artifact.path and existing.kind is artifact.kind for existing in self._artifacts):
                return
            self._artifacts.append(artifact)
        self._publish(lambda header: ArtifactCreatedEvent(header, artifact), item=artifact.item)

    def log(self, level: str, message: str, *, item: str | Path | None = None) -> None:
        self._publish(lambda header: LogMessageEvent(header, level, message), item=item)
