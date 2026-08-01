"""Persistent single-worker queue shared by desktop frontends."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from openlrc.application.credentials import CredentialStore
from openlrc.application.drafts import WorkflowDraft
from openlrc.application.jobs import JobController
from openlrc.application.preflight import preflight
from openlrc.application.queue_repository import QueueEntry, QueueEntryState, QueueRepository, QueueState
from openlrc.application.settings import AppSettings
from openlrc.workflow import CancellationToken, WorkflowEvent, WorkflowResult

QueueChangedCallback = Callable[[dict[str, object]], None]
WorkflowEventCallback = Callable[[QueueEntry, WorkflowEvent], None]


class QueueFullError(RuntimeError):
    pass


class WorkflowQueueController:
    """Run immutable Workflow recipes in FIFO order with one active worker."""

    def __init__(
        self,
        repository: QueueRepository | None = None,
        *,
        job_controller: JobController | None = None,
        credentials: CredentialStore | None = None,
        on_queue_changed: QueueChangedCallback | None = None,
        on_workflow_event: WorkflowEventCallback | None = None,
    ) -> None:
        self.repository = repository or QueueRepository()
        recovered = self.repository.load()
        self._entries = list(recovered.entries)
        self._paused = recovered.paused
        self._job_controller = job_controller or JobController()
        self._credentials = credentials or CredentialStore()
        self._on_queue_changed = on_queue_changed
        self._on_workflow_event = on_workflow_event
        self._condition = threading.Condition(threading.RLock())
        self._active_queue_id: str | None = None
        self._active_token: CancellationToken | None = None
        self._closing = False
        self._worker: threading.Thread | None = None

    @property
    def job_controller(self) -> JobController:
        return self._job_controller

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run_worker, name="openlrc-workflow-queue", daemon=True)
            self._worker.start()
            self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            items = [entry.to_dict() for entry in self._entries]
            active = next((item for item in items if item["state"] in {"dispatching", "active"}), None)
            pending = [item for item in items if item["state"] in {"pending", "blocked"}]
            return {
                "paused": self._paused,
                "active": active,
                "entries": pending,
                "total": len(items),
                "pending_count": len(pending),
            }

    def enqueue(self, draft: WorkflowDraft, settings: AppSettings) -> QueueEntry:
        report = preflight(draft, settings, self._credentials)
        if report.blocked:
            message = next(
                (issue.message for issue in report.issues if issue.severity == "blocked"), "Preflight failed."
            )
            raise ValueError(message)
        recipe = deepcopy(draft.to_recipe())
        normalized = WorkflowDraft.from_recipe(recipe)
        display_name = _display_name(normalized.paths)
        entry = QueueEntry(
            queue_id=str(uuid.uuid4()),
            state=QueueEntryState.PENDING,
            workflow=normalized.workflow,
            display_name=display_name,
            draft_recipe=recipe,
            settings_snapshot=deepcopy(settings.to_dict()),
            credential_requirements=_credential_requirements(normalized),
        )
        with self._condition:
            if self._closing:
                raise RuntimeError("Queue is shutting down.")
            if len(self._entries) >= self.repository.MAX_ENTRIES:
                raise QueueFullError("Queue is full (maximum 50 pending or blocked tasks).")
            self._entries.append(entry)
            self._save_locked()
            self._condition.notify_all()
        self._emit_queue_changed()
        return entry

    def cancel(self, queue_id: str) -> dict[str, object]:
        with self._condition:
            entry = self._find_locked(queue_id)
            if entry.state in {QueueEntryState.PENDING, QueueEntryState.BLOCKED}:
                self._entries.remove(entry)
                self._save_locked()
                outcome = {"queue_id": queue_id, "cancelled": True, "removed": True}
            elif queue_id == self._active_queue_id and self._active_token is not None:
                self._active_token.cancel()
                outcome = {"queue_id": queue_id, "cancelled": True, "removed": False}
            else:
                outcome = {"queue_id": queue_id, "cancelled": False, "removed": False}
            self._condition.notify_all()
        self._emit_queue_changed()
        return outcome

    def reorder(self, ordered_queue_ids: list[str]) -> dict[str, object]:
        with self._condition:
            movable = [
                entry for entry in self._entries if entry.state in {QueueEntryState.PENDING, QueueEntryState.BLOCKED}
            ]
            if len(set(ordered_queue_ids)) != len(ordered_queue_ids):
                raise ValueError("Queue order contains duplicate IDs.")
            if set(ordered_queue_ids) != {entry.queue_id for entry in movable}:
                raise ValueError("Queue order must contain every pending or blocked task exactly once.")
            by_id = {entry.queue_id: entry for entry in movable}
            ordered = [by_id[queue_id] for queue_id in ordered_queue_ids]
            iterator = iter(ordered)
            self._entries = [
                next(iterator) if entry.state in {QueueEntryState.PENDING, QueueEntryState.BLOCKED} else entry
                for entry in self._entries
            ]
            self._save_locked()
        self._emit_queue_changed()
        return self.snapshot()

    def pause(self) -> dict[str, object]:
        with self._condition:
            self._paused = True
            self._save_locked()
        self._emit_queue_changed()
        return self.snapshot()

    def resume(self) -> dict[str, object]:
        with self._condition:
            for entry in self._entries:
                if entry.state is QueueEntryState.BLOCKED and entry.job_id is None:
                    entry.state = QueueEntryState.PENDING
                    entry.blocked_reason = None
                    break
            self._paused = False
            self._save_locked()
            self._condition.notify_all()
        self._emit_queue_changed()
        return self.snapshot()

    def active_operation(self) -> dict[str, object] | None:
        with self._condition:
            if self._active_queue_id is None:
                return None
            entry = self._find_locked(self._active_queue_id)
            record = self._job_controller.active_record
            return {"queue": entry.to_dict(), "job": record.to_dict() if record is not None else None}

    def close(self, *, cancel_active: bool = True, timeout: float = 15.0) -> None:
        worker: threading.Thread | None
        with self._condition:
            self._closing = True
            if cancel_active and self._active_token is not None:
                self._active_token.cancel()
            self._save_locked()
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)

    def _run_worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(self._can_dispatch_locked)
                if self._closing:
                    return
                entry = next(item for item in self._entries if item.state is QueueEntryState.PENDING)
                token = CancellationToken()
                entry.state = QueueEntryState.DISPATCHING
                entry.operation_id = str(uuid.uuid4())
                entry.blocked_reason = None
                self._active_queue_id = entry.queue_id
                self._active_token = token
                self._save_locked()
            self._emit_queue_changed()
            try:
                draft = WorkflowDraft.from_recipe(deepcopy(entry.draft_recipe))
                settings = AppSettings.from_dict(deepcopy(entry.settings_snapshot))
                report = preflight(draft, settings, self._credentials)
                if report.blocked:
                    reason = next(
                        (issue.message for issue in report.issues if issue.severity == "blocked"), "Preflight failed."
                    )
                    self._block_active(entry, reason)
                    continue

                def on_event(event: WorkflowEvent) -> None:
                    with self._condition:
                        entry.state = QueueEntryState.ACTIVE
                        entry.job_id = event.header.job_id
                        self._save_locked()
                    if self._on_workflow_event is not None:
                        self._on_workflow_event(entry, event)

                result = self._job_controller.run(
                    draft, settings, self._credentials, on_event=on_event, cancellation_token=token
                )
                self._finish_active(entry, result)
            except Exception as exc:
                self._block_active(entry, str(exc) or type(exc).__name__)

    def _finish_active(self, entry: QueueEntry, result: WorkflowResult) -> None:
        with self._condition:
            entry.job_id = result.job_id
            if entry in self._entries:
                self._entries.remove(entry)
            self._active_queue_id = None
            self._active_token = None
            self._save_locked()
            self._condition.notify_all()
        self._emit_queue_changed()

    def _block_active(self, entry: QueueEntry, reason: str) -> None:
        with self._condition:
            entry.state = QueueEntryState.BLOCKED
            entry.blocked_reason = reason
            entry.operation_id = None
            self._active_queue_id = None
            self._active_token = None
            self._paused = True
            self._save_locked()
            self._condition.notify_all()
        self._emit_queue_changed()

    def _can_dispatch_locked(self) -> bool:
        return self._closing or (
            not self._paused
            and self._active_queue_id is None
            and any(entry.state is QueueEntryState.PENDING for entry in self._entries)
        )

    def _find_locked(self, queue_id: str) -> QueueEntry:
        entry = next((item for item in self._entries if item.queue_id == queue_id), None)
        if entry is None:
            raise KeyError(f"Unknown queue ID: {queue_id}")
        return entry

    def _save_locked(self) -> None:
        self.repository.save(QueueState(paused=self._paused, entries=tuple(self._entries)))

    def _emit_queue_changed(self) -> None:
        if self._on_queue_changed is not None:
            try:
                self._on_queue_changed(self.snapshot())
            except Exception:
                pass


def _display_name(paths: list[str]) -> str:
    if not paths:
        return "Untitled task"
    first = Path(paths[0]).name
    return first if len(paths) == 1 else f"{first} +{len(paths) - 1}"


def _credential_requirements(draft: WorkflowDraft) -> list[str]:
    providers: list[str] = []
    if draft.translation_backend == "online":
        providers.extend(name for name in (draft.provider, draft.retry_provider, draft.reviewer_provider) if name)
    if draft.context_provider and draft.context_provider != "local":
        providers.append(draft.context_provider)
    return list(dict.fromkeys(providers))
