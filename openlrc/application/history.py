"""Persistent, secret-free TUI job history."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

from openlrc.application.paths import app_support_dir, atomic_write_json, backup_corrupt_file
from openlrc.workflow import redact_sensitive_text


class JobRecordStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class JobItemState:
    identity: str
    display_name: str
    path: str
    state: str = "waiting"
    current_stage: str | None = None
    progress: float = 0.0
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    recovery: bool = False


@dataclass(slots=True)
class JobRecord:
    job_id: str
    workflow: str
    name: str
    status: JobRecordStatus
    input_paths: list[str]
    recipe: dict[str, object]
    translation_mode: str | None = None
    current_stage: str | None = None
    progress: float = 0.0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    outputs: list[str] = field(default_factory=list)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    reviews: list[dict[str, object]] = field(default_factory=list)
    models: dict[str, dict[str, object]] = field(default_factory=dict)
    event_log: list[str] = field(default_factory=list)
    api_fee: float = 0.0
    elapsed_seconds: float = 0.0
    error: dict[str, object] | None = None
    resumed_from: str | None = None
    items: dict[str, JobItemState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        result["recipe"] = sanitize_recipe(self.recipe)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> JobRecord:
        values = dict(payload)
        values["status"] = JobRecordStatus(str(values["status"]))
        raw_items = values.get("items")
        if isinstance(raw_items, dict):
            values["items"] = {
                str(key): JobItemState(**cast(Any, item)) for key, item in raw_items.items() if isinstance(item, dict)
            }
        allowed = cls.__dataclass_fields__
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**cast(Any, filtered))


_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|secret|password)")


def sanitize_recipe(value):
    if isinstance(value, dict):
        return {key: sanitize_recipe(item) for key, item in value.items() if not _SECRET_KEY.search(str(key))}
    if isinstance(value, list):
        return [sanitize_recipe(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_recipe(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class JobRepository:
    MAX_RECORDS = 100

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_support_dir() / "jobs.json"
        self.last_warning: str | None = None

    def load(self, *, mark_interrupted: bool = True) -> list[JobRecord]:
        self.last_warning = None
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("Unsupported job-history schema.")
            records = [JobRecord.from_dict(item) for item in payload.get("jobs", []) if isinstance(item, dict)]
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            try:
                backup = backup_corrupt_file(self.path)
                self.last_warning = f"Invalid job history was backed up to {backup}: {exc}"
            except OSError:
                self.last_warning = f"Invalid job history could not be loaded: {exc}"
            return []
        changed = False
        if mark_interrupted:
            for record in records:
                if record.status is JobRecordStatus.RUNNING:
                    record.status = JobRecordStatus.INTERRUPTED
                    record.completed_at = datetime.now(timezone.utc).isoformat()
                    changed = True
        if changed:
            self.save(records)
        return records[: self.MAX_RECORDS]

    def save(self, records: list[JobRecord]) -> None:
        ordered = sorted(records, key=lambda record: record.started_at, reverse=True)[: self.MAX_RECORDS]
        atomic_write_json(self.path, {"schema_version": 1, "jobs": [record.to_dict() for record in ordered]})

    def upsert(self, records: list[JobRecord], record: JobRecord) -> list[JobRecord]:
        updated = [item for item in records if item.job_id != record.job_id]
        updated.insert(0, record)
        updated = updated[: self.MAX_RECORDS]
        self.save(updated)
        return updated

    def delete(self, records: list[JobRecord], job_id: str) -> list[JobRecord]:
        updated = [record for record in records if record.job_id != job_id]
        self.save(updated)
        return updated
