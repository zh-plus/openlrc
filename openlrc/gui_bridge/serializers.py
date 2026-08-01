"""Explicit wire DTO serializers for GUI protocol v1."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import cast

from openlrc.application.history import JobRecord, JobRecordStatus
from openlrc.application.preflight import PreflightReport
from openlrc.application.queue_repository import QueueEntry
from openlrc.workflow import (
    ArtifactCreatedEvent,
    LogMessageEvent,
    ModelLifecycleEvent,
    StageCompletedEvent,
    StageProgressEvent,
    StageStartedEvent,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowEvent,
    WorkflowFailedEvent,
    WorkflowStartedEvent,
)


def json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Path, datetime)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    return str(value)


def preflight_dto(report: PreflightReport) -> dict[str, object]:
    return {
        "status": report.status.lower(),
        "blocked": report.blocked,
        "issues": [cast(dict[str, object], json_value(issue)) for issue in report.issues],
        "summary": dict(report.summary),
    }


def queue_entry_dto(entry: QueueEntry) -> dict[str, object]:
    return entry.to_dict()


def job_summary_dto(record: JobRecord) -> dict[str, object]:
    return {
        "job_id": record.job_id,
        "workflow": record.workflow,
        "name": record.name,
        "status": record.status.value,
        "input_paths": list(record.input_paths),
        "translation_mode": record.translation_mode,
        "progress": record.progress,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "outputs": list(record.outputs),
        "elapsed_seconds": record.elapsed_seconds,
        "error": json_value(record.error),
        "resumed_from": record.resumed_from,
    }


def job_detail_dto(record: JobRecord) -> dict[str, object]:
    payload = record.to_dict()
    payload["artifacts"] = [
        {"artifact_id": f"artifact-{index}", **cast(dict[str, object], json_value(artifact))}
        for index, artifact in enumerate(record.artifacts)
    ]
    return payload


def history_job(record: JobRecord) -> bool:
    return record.status not in {JobRecordStatus.RUNNING, JobRecordStatus.CANCELLED}


def workflow_event_frame(entry: QueueEntry, event: WorkflowEvent) -> dict[str, object]:
    event_name = {
        WorkflowStartedEvent: "workflow.started",
        WorkflowCompletedEvent: "workflow.completed",
        WorkflowFailedEvent: "workflow.failed",
        WorkflowCancelledEvent: "workflow.cancelled",
        StageStartedEvent: "workflow.stage_started",
        StageProgressEvent: "workflow.stage_progress",
        StageCompletedEvent: "workflow.stage_completed",
        ModelLifecycleEvent: "workflow.model_lifecycle",
        ArtifactCreatedEvent: "workflow.artifact_created",
        LogMessageEvent: "workflow.log",
    }[type(event)]
    payload = {field.name: json_value(getattr(event, field.name)) for field in fields(event) if field.name != "header"}
    if isinstance(event, StageProgressEvent):
        payload["percent"] = event.percent
    payload["item"] = event.header.item
    payload["workflow"] = event.header.workflow.value
    payload["timestamp"] = event.header.timestamp.isoformat()
    return {
        "type": "event",
        "protocol": 1,
        "event": event_name,
        "queue_id": entry.queue_id,
        "operation_id": entry.operation_id,
        "job_id": event.header.job_id,
        "sequence": event.header.sequence,
        "payload": payload,
    }
