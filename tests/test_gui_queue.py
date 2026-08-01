from __future__ import annotations

import json
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openlrc.application import AppSettings, PreflightReport, WorkflowDraft
from openlrc.application.queue_repository import QueueEntry, QueueEntryState, QueueRepository, QueueState
from openlrc.application.workflow_queue import WorkflowQueueController
from openlrc.workflow import EventHeader, WorkflowKind, WorkflowResult, WorkflowStartedEvent, WorkflowStatus


class FakeJobController:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.records = ()
        self.active_record = None
        self.concurrent = 0
        self.max_concurrent = 0
        self.completed = threading.Event()

    def run(self, draft, _settings, _credentials, *, on_event, cancellation_token):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append(draft.paths[0])
        job_id = f"job-{len(self.calls)}"
        on_event(
            WorkflowStartedEvent(EventHeader(job_id, len(self.calls), datetime.now(UTC), WorkflowKind(draft.workflow)))
        )
        time.sleep(0.02)
        self.concurrent -= 1
        if len(self.calls) >= 2:
            self.completed.set()
        return WorkflowResult(
            job_id,
            WorkflowKind(draft.workflow),
            WorkflowStatus.CANCELLED if cancellation_token.is_cancelled else WorkflowStatus.SUCCEEDED,
        )


class CancellableJobController(FakeJobController):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = threading.Event()
        self.cancel_observed = threading.Event()

    def run(self, draft, _settings, _credentials, *, on_event, cancellation_token):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append(draft.paths[0])
        job_id = f"job-{len(self.calls)}"
        on_event(
            WorkflowStartedEvent(EventHeader(job_id, len(self.calls), datetime.now(UTC), WorkflowKind(draft.workflow)))
        )
        if len(self.calls) == 1:
            self.first_started.set()
            deadline = time.monotonic() + 2
            while not cancellation_token.is_cancelled and time.monotonic() < deadline:
                time.sleep(0.005)
            if cancellation_token.is_cancelled:
                self.cancel_observed.set()
        self.concurrent -= 1
        if len(self.calls) >= 2:
            self.completed.set()
        status = WorkflowStatus.CANCELLED if cancellation_token.is_cancelled else WorkflowStatus.SUCCEEDED
        return WorkflowResult(job_id, WorkflowKind(draft.workflow), status)


def _ready(*_args) -> PreflightReport:
    return PreflightReport()


def test_queue_repository_is_private_secret_free_and_recovers_active_as_paused(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    repository = QueueRepository(path)
    entry = QueueEntry(
        queue_id="queue-1",
        state=QueueEntryState.ACTIVE,
        workflow="transcribe",
        display_name="audio.wav",
        draft_recipe={"paths": ["/tmp/audio.wav"], "api_key": "must-not-leak"},
        settings_snapshot={"providers": {"openai": {"password": "must-not-leak"}}},
        operation_id="operation-1",
        job_id="job-1",
    )
    repository.save(QueueState(paused=False, entries=(entry,)))

    recovered = repository.load()

    assert recovered.paused is True
    assert recovered.recovered is True
    assert recovered.entries[0].state is QueueEntryState.BLOCKED
    assert recovered.entries[0].operation_id is None
    raw = path.read_text(encoding="utf-8")
    assert "must-not-leak" not in raw
    assert "api_key" not in raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_queue_runs_fifo_with_only_one_active_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("openlrc.application.workflow_queue.preflight", _ready)
    jobs = FakeJobController()
    controller = WorkflowQueueController(QueueRepository(tmp_path / "queue.json"), job_controller=jobs)  # type: ignore[arg-type]
    controller.start()

    controller.enqueue(WorkflowDraft(workflow="transcribe", paths=["first.wav"]), AppSettings())
    controller.enqueue(WorkflowDraft(workflow="transcribe", paths=["second.wav"]), AppSettings())

    assert jobs.completed.wait(2)
    controller.close()
    assert jobs.calls == ["first.wav", "second.wav"]
    assert jobs.max_concurrent == 1
    assert controller.snapshot()["total"] == 0


def test_clean_shutdown_with_empty_queue_does_not_pause_next_launch(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    first = WorkflowQueueController(QueueRepository(path), job_controller=FakeJobController())  # type: ignore[arg-type]
    first.start()
    first.close()

    second = WorkflowQueueController(QueueRepository(path), job_controller=FakeJobController())  # type: ignore[arg-type]

    assert second.snapshot()["paused"] is False
    second.close()


def test_pending_cancel_removes_entry_without_starting_job(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("openlrc.application.workflow_queue.preflight", _ready)
    jobs = FakeJobController()
    controller = WorkflowQueueController(QueueRepository(tmp_path / "queue.json"), job_controller=jobs)  # type: ignore[arg-type]
    controller.pause()
    controller.start()
    entry = controller.enqueue(WorkflowDraft(workflow="transcribe", paths=["pending.wav"]), AppSettings())

    result = controller.cancel(entry.queue_id)
    controller.close()

    assert result == {"queue_id": entry.queue_id, "cancelled": True, "removed": True}
    assert jobs.calls == []
    payload = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert payload["entries"] == []


def test_reorder_requires_every_movable_id_once_and_persists_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("openlrc.application.workflow_queue.preflight", _ready)
    controller = WorkflowQueueController(QueueRepository(tmp_path / "queue.json"), job_controller=FakeJobController())  # type: ignore[arg-type]
    controller.pause()
    entries = [
        controller.enqueue(WorkflowDraft(workflow="transcribe", paths=[f"{name}.wav"]), AppSettings())
        for name in ("first", "second", "third")
    ]

    reordered = controller.reorder([entries[2].queue_id, entries[0].queue_id, entries[1].queue_id])

    reordered_entries = reordered["entries"]
    assert isinstance(reordered_entries, list)
    assert [item["display_name"] for item in reordered_entries if isinstance(item, dict)] == [
        "third.wav",
        "first.wav",
        "second.wav",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        controller.reorder([entries[0].queue_id, entries[0].queue_id, entries[1].queue_id])
    with pytest.raises(ValueError, match="every pending"):
        controller.reorder([entries[0].queue_id])
    controller.close()

    reloaded = QueueRepository(tmp_path / "queue.json").load()
    assert reloaded.paused is True
    assert [entry.display_name for entry in reloaded.entries] == ["third.wav", "first.wav", "second.wav"]


def test_active_cancel_is_cooperative_and_next_pending_task_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("openlrc.application.workflow_queue.preflight", _ready)
    jobs = CancellableJobController()
    controller = WorkflowQueueController(QueueRepository(tmp_path / "queue.json"), job_controller=jobs)  # type: ignore[arg-type]
    controller.start()
    first = controller.enqueue(WorkflowDraft(workflow="transcribe", paths=["first.wav"]), AppSettings())
    controller.enqueue(WorkflowDraft(workflow="transcribe", paths=["second.wav"]), AppSettings())

    assert jobs.first_started.wait(1)
    result = controller.cancel(first.queue_id)
    assert result == {"queue_id": first.queue_id, "cancelled": True, "removed": False}
    assert jobs.cancel_observed.wait(1)
    assert jobs.completed.wait(2)
    controller.close()

    assert jobs.calls == ["first.wav", "second.wav"]
    assert jobs.max_concurrent == 1
    assert controller.snapshot()["total"] == 0


def test_restart_with_pending_entries_defaults_to_paused_until_resume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("openlrc.application.workflow_queue.preflight", _ready)
    path = tmp_path / "queue.json"
    first_jobs = FakeJobController()
    first = WorkflowQueueController(QueueRepository(path), job_controller=first_jobs)  # type: ignore[arg-type]
    first.pause()
    first.enqueue(WorkflowDraft(workflow="transcribe", paths=["pending.wav"]), AppSettings())
    first.close()

    second_jobs = FakeJobController()
    second = WorkflowQueueController(QueueRepository(path), job_controller=second_jobs)  # type: ignore[arg-type]
    second.start()
    time.sleep(0.05)
    assert second.snapshot()["paused"] is True
    assert second_jobs.calls == []

    second.resume()
    deadline = time.monotonic() + 2
    while not second_jobs.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    second.close()
    assert second_jobs.calls == ["pending.wav"]


def test_corrupt_or_duplicate_queue_is_backed_up_and_not_executed(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    path.write_text("{broken", encoding="utf-8")
    repository = QueueRepository(path)

    assert repository.load().entries == ()
    assert repository.last_warning is not None
    assert list(tmp_path.glob("queue.json.corrupt-*"))

    duplicate = QueueEntry(
        queue_id="duplicate",
        state=QueueEntryState.PENDING,
        workflow="transcribe",
        display_name="audio.wav",
        draft_recipe={"workflow": "transcribe", "paths": ["audio.wav"]},
        settings_snapshot={},
    )
    repository.save(QueueState(paused=True, entries=(duplicate, duplicate)))
    assert repository.load().entries == ()
    assert "duplicate" in (repository.last_warning or "").lower()
