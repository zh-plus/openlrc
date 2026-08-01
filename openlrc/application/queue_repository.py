"""Atomic persistence for the GUI workflow queue."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from openlrc.application.history import sanitize_recipe
from openlrc.application.paths import app_support_dir, atomic_write_json, backup_corrupt_file


class QueueEntryState(StrEnum):
    PENDING = "pending"
    DISPATCHING = "dispatching"
    ACTIVE = "active"
    BLOCKED = "blocked"


@dataclass(slots=True)
class QueueEntry:
    queue_id: str
    state: QueueEntryState
    workflow: str
    display_name: str
    draft_recipe: dict[str, object]
    settings_snapshot: dict[str, object]
    credential_requirements: list[str] = field(default_factory=list)
    enqueued_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    operation_id: str | None = None
    job_id: str | None = None
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["draft_recipe"] = sanitize_recipe(self.draft_recipe)
        payload["settings_snapshot"] = sanitize_recipe(self.settings_snapshot)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> QueueEntry:
        values = dict(payload)
        values["state"] = QueueEntryState(str(values["state"]))
        values["draft_recipe"] = _mapping(values.get("draft_recipe"))
        values["settings_snapshot"] = _mapping(values.get("settings_snapshot"))
        raw_requirements = values.get("credential_requirements", [])
        values["credential_requirements"] = (
            [str(item) for item in raw_requirements] if isinstance(raw_requirements, list) else []
        )
        allowed = cls.__dataclass_fields__
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**cast(Any, filtered))


@dataclass(frozen=True, slots=True)
class QueueState:
    paused: bool
    entries: tuple[QueueEntry, ...]
    recovered: bool = False


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


class QueueRepository:
    """Persist a bounded, secret-free ordered queue using atomic replacement."""

    SCHEMA_VERSION = 1
    MAX_ENTRIES = 50

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_support_dir() / "queue.json"
        self.last_warning: str | None = None

    def load(self) -> QueueState:
        self.last_warning = None
        if not self.path.exists():
            return QueueState(paused=False, entries=())
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("Unsupported queue schema.")
            raw_entries = payload.get("entries", [])
            if not isinstance(raw_entries, list):
                raise TypeError("Queue entries must be an array.")
            entries = [QueueEntry.from_dict(item) for item in raw_entries if isinstance(item, dict)]
            queue_ids = [entry.queue_id for entry in entries]
            if len(queue_ids) != len(set(queue_ids)):
                raise ValueError("Queue contains duplicate IDs.")
            if len(entries) > self.MAX_ENTRIES:
                entries = entries[: self.MAX_ENTRIES]
                self.last_warning = "Queue exceeded its limit and was truncated to 50 entries."
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            try:
                backup = backup_corrupt_file(self.path)
                self.last_warning = f"Invalid queue was backed up to {backup}: {exc}"
            except OSError:
                self.last_warning = f"Invalid queue could not be loaded: {exc}"
            return QueueState(paused=False, entries=())

        recovered = False
        for entry in entries:
            if entry.state in {QueueEntryState.DISPATCHING, QueueEntryState.ACTIVE}:
                entry.state = QueueEntryState.BLOCKED
                entry.blocked_reason = (
                    "OpenLRC closed while this task was starting or active. Review it and resume as a new task."
                )
                entry.operation_id = None
                recovered = True
        paused = bool(entries) or bool(payload.get("paused", False))
        state = QueueState(paused=paused, entries=tuple(entries), recovered=recovered)
        if recovered or paused != bool(payload.get("paused", False)):
            try:
                self.save(state)
            except OSError as exc:
                self.last_warning = f"Recovered queue in memory, but could not save it: {exc}"
        return state

    def save(self, state: QueueState) -> None:
        if len(state.entries) > self.MAX_ENTRIES:
            raise ValueError("Queue cannot contain more than 50 entries.")
        atomic_write_json(
            self.path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "paused": state.paused,
                "entries": [entry.to_dict() for entry in state.entries],
            },
        )
