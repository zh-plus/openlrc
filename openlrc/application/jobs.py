"""Single-active-job controller for interactive frontends."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from openlrc.application.credentials import CredentialStore
from openlrc.application.drafts import WorkflowDraft
from openlrc.application.history import JobItemState, JobRecord, JobRecordStatus, JobRepository
from openlrc.application.operations import OperationGuard
from openlrc.application.settings import AppSettings
from openlrc.workflow import (
    ArtifactCreatedEvent,
    CancellationToken,
    ExecutionContext,
    LogMessageEvent,
    ModelLifecycleEvent,
    StageCompletedEvent,
    StageProgressEvent,
    StageStartedEvent,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowEvent,
    WorkflowExecutor,
    WorkflowFailedEvent,
    WorkflowKind,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
    redact_sensitive_text,
)

EventCallback = Callable[[WorkflowEvent], None]


class JobController:
    """Own one executor and persist safe summaries around each execution."""

    def __init__(
        self,
        repository: JobRepository | None = None,
        *,
        operation_guard: OperationGuard | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> None:
        self.repository = repository or JobRepository()
        self._records = self.repository.load()
        self._executor = executor or WorkflowExecutor()
        self._lock = threading.RLock()
        self._active_context: ExecutionContext | None = None
        self._active_record: JobRecord | None = None
        self.persistence_warning: str | None = None
        self.operation_guard = operation_guard or OperationGuard()

    @property
    def records(self) -> tuple[JobRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def active_record(self) -> JobRecord | None:
        with self._lock:
            return self._active_record

    @property
    def is_running(self) -> bool:
        return self.active_record is not None

    def run(
        self,
        draft: WorkflowDraft,
        settings: AppSettings,
        credentials: CredentialStore,
        *,
        on_event: EventCallback | None = None,
        resumed_from: str | None = None,
    ) -> WorkflowResult:
        with self.operation_guard.acquire("workflow", "subtitle workflow"):
            return self._run_owned(draft, settings, credentials, on_event=on_event, resumed_from=resumed_from)

    def _run_owned(
        self,
        draft: WorkflowDraft,
        settings: AppSettings,
        credentials: CredentialStore,
        *,
        on_event: EventCallback | None,
        resumed_from: str | None,
    ) -> WorkflowResult:
        with self._lock:
            if self._active_context is not None:
                raise RuntimeError("Another OpenLRC job is already running.")
        request = draft.build_request(settings, credentials)
        kind = WorkflowKind(draft.workflow)
        token = CancellationToken()
        context = ExecutionContext(kind, cancellation_token=token)
        record = JobRecord(
            job_id=context.job_id,
            workflow=kind.value,
            name=_job_name(draft.paths),
            status=JobRecordStatus.RUNNING,
            input_paths=list(draft.paths),
            recipe=draft.to_recipe(),
            translation_mode=(draft.mode if draft.translation_backend not in {"", "none"} else None),
            resumed_from=resumed_from,
            items={
                str(Path(path).expanduser().resolve(strict=False)): JobItemState(
                    identity=str(Path(path).expanduser().resolve(strict=False)),
                    display_name=Path(path).name,
                    path=str(Path(path).expanduser().resolve(strict=False)),
                )
                for path in draft.paths
            },
        )

        def sink(event: WorkflowEvent) -> None:
            self._apply_event(record, event)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    pass

        context.event_sink = sink
        with self._lock:
            if self._active_context is not None:
                raise RuntimeError("Another OpenLRC job is already running.")
            self._active_context = context
            self._active_record = record
            try:
                self._records = self.repository.upsert(self._records, record)
            except Exception as exc:
                self._active_context = None
                self._active_record = None
                raise RuntimeError(f"Cannot save the starting job; workflow was not started: {exc}") from exc
        try:
            result = self._executor.execute(request, context)
            self._complete_record(record, result)
            return result
        finally:
            try:
                with self._lock:
                    try:
                        self._records = self.repository.upsert(self._records, record)
                        self.persistence_warning = None
                    except Exception as exc:
                        self.persistence_warning = f"History not saved: {exc}"
            finally:
                with self._lock:
                    self._active_context = None
                    self._active_record = None

    def cancel(self) -> bool:
        token = self.prepare_cancel()
        if token is None:
            return False
        token.cancel()
        return True

    def prepare_cancel(self) -> CancellationToken | None:
        """Mark the job cancelling and return its token for off-thread signalling."""
        with self._lock:
            context = self._active_context
            record = self._active_record
        if context is None:
            return None
        if record is not None:
            with self._lock:
                record.current_stage = "cancelling"
                record.event_log.append("job · cancelling; waiting for owned-resource cleanup")
                try:
                    self._records = self.repository.upsert(self._records, record)
                except Exception as exc:
                    self.persistence_warning = f"History not saved: {exc}"
        return context.cancellation_token

    def delete_record(self, job_id: str) -> None:
        with self._lock:
            if self._active_record is not None and self._active_record.job_id == job_id:
                raise RuntimeError("Cannot delete the active job record.")
            record = next((item for item in self._records if item.job_id == job_id), None)
            if record is not None and record.status is JobRecordStatus.RUNNING:
                raise RuntimeError("Cannot delete a running job record.")
            self._records = self.repository.delete(self._records, job_id)

    def can_resume(self, record: JobRecord) -> bool:
        return record.status in {
            JobRecordStatus.FAILED,
            JobRecordStatus.CANCELLED,
            JobRecordStatus.INTERRUPTED,
            JobRecordStatus.SUCCEEDED_WITH_WARNINGS,
        }

    @staticmethod
    def _apply_event(record: JobRecord, event: WorkflowEvent) -> None:
        item = Path(event.header.item).name if event.header.item else "job"
        raw_item = event.header.item
        if isinstance(event, WorkflowFailedEvent) and raw_item is None:
            raw_item = event.error.item
        item_state = _event_item(record, raw_item)
        if isinstance(event, StageStartedEvent):
            record.current_stage = event.stage.value
            if item_state is not None:
                item_state.state = "running"
                item_state.current_stage = event.stage.value
                item_state.progress = 0.0
            line = f"{item} · {event.stage.value} started"
        elif isinstance(event, StageProgressEvent):
            record.current_stage = event.stage.value
            record.progress = event.percent
            if item_state is not None:
                item_state.state = "running"
                item_state.current_stage = event.stage.value
                item_state.progress = event.percent
            line = f"{item} · {event.stage.value} {event.percent:.1f}%"
        elif isinstance(event, StageCompletedEvent):
            record.current_stage = event.stage.value
            if item_state is not None:
                item_state.current_stage = event.stage.value
                item_state.progress = 100.0
                item_state.state = "completed" if event.stage is WorkflowStage.EXPORT else "waiting"
            line = f"{item} · {event.stage.value} {event.outcome.value}"
        elif isinstance(event, ModelLifecycleEvent):
            key = f"{event.role or 'model'}:{event.model}"
            record.models[key] = {
                "model": event.model,
                "state": event.state,
                "owned": event.owned,
                "role": event.role,
                "endpoint": event.endpoint,
            }
            line = f"{event.role or 'model'} · {event.model} {event.state}"
        elif isinstance(event, ArtifactCreatedEvent):
            if item_state is not None:
                output = str(event.artifact.path)
                if output not in item_state.outputs:
                    item_state.outputs.append(output)
                if event.artifact.primary:
                    item_state.state = "completed"
                    item_state.progress = 100.0
            line = f"artifact · {event.artifact.kind.value} · {event.artifact.path}"
        elif isinstance(event, LogMessageEvent):
            line = f"{event.level} · {event.message}"
        elif isinstance(event, WorkflowFailedEvent):
            if item_state is not None:
                item_state.state = "failed"
                item_state.error = event.error.message
            else:
                for state in record.items.values():
                    if state.state == "running":
                        state.state = "failed"
                        state.error = event.error.message
            line = f"failed · {event.error.category.value} · {event.error.message}"
        elif isinstance(event, WorkflowCancelledEvent):
            for state in record.items.values():
                if state.state in {"running", "waiting"}:
                    state.state = "cancelled"
            line = f"cancelled · {event.message}"
        elif isinstance(event, WorkflowCompletedEvent):
            if event.status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.SUCCEEDED_WITH_WARNINGS}:
                for state in record.items.values():
                    if state.state not in {"failed", "cancelled"}:
                        state.state = "completed" if event.status is WorkflowStatus.SUCCEEDED else "warning"
                        state.progress = 100.0
            line = f"completed · {event.status.value}"
        else:
            line = "workflow started"
        record.event_log.append(redact_sensitive_text(line))
        del record.event_log[:-500]

    @staticmethod
    def _complete_record(record: JobRecord, result: WorkflowResult) -> None:
        status_map = {
            WorkflowStatus.SUCCEEDED: JobRecordStatus.SUCCEEDED,
            WorkflowStatus.SUCCEEDED_WITH_WARNINGS: JobRecordStatus.SUCCEEDED_WITH_WARNINGS,
            WorkflowStatus.FAILED: JobRecordStatus.FAILED,
            WorkflowStatus.CANCELLED: JobRecordStatus.CANCELLED,
        }
        record.status = status_map[result.status]
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record.progress = (
            100.0
            if result.status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.SUCCEEDED_WITH_WARNINGS}
            else record.progress
        )
        record.outputs = [str(path) for path in result.outputs if path.exists()]
        record.artifacts = [
            {
                "path": str(artifact.path),
                "kind": artifact.kind.value,
                "item": artifact.item,
                "primary": artifact.primary,
            }
            for artifact in result.artifacts
            if artifact.path.exists()
        ]
        record.reviews = [
            {
                "item": review.item,
                "incomplete": review.incomplete,
                "checkpoint": str(review.checkpoint) if review.checkpoint else None,
                "report": str(review.report) if review.report else None,
                "session": str(review.session) if review.session else None,
                "details": review.details,
            }
            for review in result.reviews
        ]
        record.api_fee = result.api_fee
        record.elapsed_seconds = result.elapsed_seconds
        if result.error is not None:
            record.error = {
                "category": result.error.category.value,
                "message": result.error.message,
                "stage": result.error.stage.value if result.error.stage else None,
                "item": result.error.item,
                "retryable": result.error.retryable,
                "hint": result.error.hint,
            }
            item_state = _event_item(record, result.error.item)
            if item_state is not None:
                item_state.state = "failed"
                item_state.error = result.error.message
            else:
                for state in record.items.values():
                    if state.state in {"running", "waiting"}:
                        state.state = "failed"
                        state.error = result.error.message


def _job_name(paths: list[str]) -> str:
    if not paths:
        return "Untitled job"
    first = Path(paths[0]).name
    return first if len(paths) == 1 else f"{first} +{len(paths) - 1}"


def _event_item(record: JobRecord, raw_item: str | None) -> JobItemState | None:
    if not raw_item:
        return None
    candidate = Path(raw_item).expanduser().resolve(strict=False)
    if str(candidate) in record.items:
        return record.items[str(candidate)]
    candidate_stem = candidate.stem
    for suffix in ("_preprocessed_transcribed", "_preprocessed", "_transcribed"):
        candidate_stem = candidate_stem.removesuffix(suffix)
    for state in record.items.values():
        source = Path(state.path)
        same_directory = candidate.parent == source.parent or candidate.parent.parent == source.parent
        if same_directory and candidate_stem == source.stem:
            return state
    return None
