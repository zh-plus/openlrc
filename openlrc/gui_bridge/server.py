"""OpenLRC desktop JSON Lines sidecar server."""

from __future__ import annotations

import importlib.metadata
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, TextIO, cast

from openlrc.application import (
    AppSettings,
    CredentialStore,
    JobRecord,
    ResourceStatusService,
    SettingsStore,
    WorkflowDraft,
    WorkflowQueueController,
    preflight,
)
from openlrc.gui_bridge.framing import JsonLineWriter, read_json_lines
from openlrc.gui_bridge.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    ProtocolRequest,
    error_from_exception,
    error_response,
    parse_request,
    success_response,
)
from openlrc.gui_bridge.serializers import (
    history_job,
    job_detail_dto,
    job_summary_dto,
    json_value,
    preflight_dto,
    workflow_event_frame,
)

CAPABILITIES = (
    "workflow.transcribe",
    "workflow.translate",
    "workflow.run",
    "queue",
    "settings",
    "credentials",
    "resources",
    "jobs",
)
PROVIDERS = frozenset({"openai", "anthropic", "google", "litellm", "third_party"})


def hello_payload() -> dict[str, object]:
    try:
        version = importlib.metadata.version("openlrc-mac")
    except importlib.metadata.PackageNotFoundError:
        version = "0.5.0"
    return {
        "service_version": version,
        "min_protocol": PROTOCOL_VERSION,
        "max_protocol": PROTOCOL_VERSION,
        "runtime_mode": "development",
        "capabilities": list(CAPABILITIES),
    }


class DesktopService:
    def __init__(self, writer: JsonLineWriter) -> None:
        self.writer = writer
        self.settings_store = SettingsStore()
        self.credentials = CredentialStore()
        self.resources = ResourceStatusService()
        self.queue = WorkflowQueueController(
            credentials=self.credentials, on_queue_changed=self._queue_changed, on_workflow_event=self._workflow_event
        )
        self.shutdown_requested = False

    def start(self) -> None:
        self.queue.start()

    def close(self) -> None:
        self.queue.close(cancel_active=True)

    def handle(self, request: ProtocolRequest) -> object:
        handlers = {
            "app.hello": self._app_hello,
            "app.shutdown": self._app_shutdown,
            "operation.active": self._operation_active,
            "queue.snapshot": self._queue_snapshot,
            "queue.enqueue": self._queue_enqueue,
            "queue.cancel": self._queue_cancel,
            "queue.reorder": self._queue_reorder,
            "queue.pause": self._queue_pause,
            "queue.resume": self._queue_resume,
            "settings.get": self._settings_get,
            "settings.update": self._settings_update,
            "credentials.status": self._credentials_status,
            "credentials.set": self._credentials_set,
            "credentials.delete": self._credentials_delete,
            "resources.status": self._resources_status,
            "workflow.preflight": self._workflow_preflight,
            "jobs.list": self._jobs_list,
            "jobs.get": self._jobs_get,
            "jobs.delete": self._jobs_delete,
            "jobs.resume_draft": self._jobs_resume_draft,
            "artifacts.resolve": self._artifacts_resolve,
        }
        handler = handlers.get(request.method)
        if handler is None:
            raise ProtocolError("UNKNOWN_METHOD", f"Unknown method: {request.method}")
        return handler(request.params)

    @staticmethod
    def _app_hello(_params: dict[str, object]) -> object:
        return hello_payload()

    def _app_shutdown(self, _params: dict[str, object]) -> object:
        self.shutdown_requested = True
        return {"accepted": True}

    def _operation_active(self, _params: dict[str, object]) -> object:
        return self.queue.active_operation()

    def _queue_snapshot(self, _params: dict[str, object]) -> object:
        return self.queue.snapshot()

    def _queue_enqueue(self, params: dict[str, object]) -> object:
        draft = _draft(params)
        settings = self.settings_store.load()
        entry = self.queue.enqueue(draft, settings)
        return {"queue_id": entry.queue_id, "entry": entry.to_dict()}

    def _queue_cancel(self, params: dict[str, object]) -> object:
        return self.queue.cancel(_required_string(params, "queue_id"))

    def _queue_reorder(self, params: dict[str, object]) -> object:
        raw = params.get("ordered_queue_ids")
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("ordered_queue_ids must be an array of strings.")
        return self.queue.reorder(cast(list[str], raw))

    def _queue_pause(self, _params: dict[str, object]) -> object:
        return self.queue.pause()

    def _queue_resume(self, _params: dict[str, object]) -> object:
        return self.queue.resume()

    def _settings_get(self, _params: dict[str, object]) -> object:
        return self.settings_store.load().to_dict()

    def _settings_update(self, params: dict[str, object]) -> object:
        patch = params.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("settings patch must be an object.")
        merged = _deep_merge(self.settings_store.load().to_dict(), cast(dict[str, object], patch))
        settings = AppSettings.from_dict(merged)
        self.settings_store.save(settings)
        return settings.to_dict()

    def _credentials_status(self, params: dict[str, object]) -> object:
        provider = _provider(params)
        resolved = self.credentials.resolve(provider)
        return {
            "provider": provider,
            "present": resolved.value is not None,
            "source": resolved.source.value,
            "environment_name": resolved.environment_name,
            "warning": self.credentials.last_error,
        }

    def _credentials_set(self, params: dict[str, object]) -> object:
        provider = _provider(params)
        secret = _required_string(params, "secret", max_length=16_384)
        self.credentials.set(provider, secret)
        return self._credentials_status({"provider": provider})

    def _credentials_delete(self, params: dict[str, object]) -> object:
        provider = _provider(params)
        self.credentials.delete(provider)
        return {"provider": provider, "deleted": True}

    def _resources_status(self, _params: dict[str, object]) -> object:
        settings = self.settings_store.load()
        statuses = [
            *(dict(asdict(item), group="runtime") for item in self.resources.doctor()),
            *(dict(asdict(item), group="models") for item in self.resources.models(settings.local_models)),
        ]
        return statuses

    def _workflow_preflight(self, params: dict[str, object]) -> object:
        draft = _draft(params)
        return preflight_dto(preflight(draft, self.settings_store.load(), self.credentials))

    def _jobs_list(self, _params: dict[str, object]) -> object:
        return [job_summary_dto(record) for record in self.queue.job_controller.records if history_job(record)]

    def _jobs_get(self, params: dict[str, object]) -> object:
        return job_detail_dto(self._job(_required_string(params, "job_id")))

    def _jobs_delete(self, params: dict[str, object]) -> object:
        job_id = _required_string(params, "job_id")
        self.queue.job_controller.delete_record(job_id)
        return {"job_id": job_id, "deleted": True, "artifacts_deleted": False}

    def _jobs_resume_draft(self, params: dict[str, object]) -> object:
        record = self._job(_required_string(params, "job_id"))
        draft = WorkflowDraft.from_recipe(deepcopy(record.recipe))
        draft.paths = list(record.input_paths)
        return asdict(draft)

    def _artifacts_resolve(self, params: dict[str, object]) -> object:
        record = self._job(_required_string(params, "job_id"))
        artifact_id = _required_string(params, "artifact_id")
        prefix, separator, raw_index = artifact_id.partition("-")
        if prefix != "artifact" or not separator or not raw_index.isdigit():
            raise ValueError("Invalid artifact ID.")
        index = int(raw_index)
        if index >= len(record.artifacts):
            raise KeyError(f"Unknown artifact ID: {artifact_id}")
        artifact = record.artifacts[index]
        path = artifact.get("path")
        if not isinstance(path, str):
            raise KeyError(f"Artifact has no resolvable path: {artifact_id}")
        return {"artifact_id": artifact_id, "path": str(Path(path).expanduser().resolve(strict=False))}

    def _job(self, job_id: str) -> JobRecord:
        record = next((item for item in self.queue.job_controller.records if item.job_id == job_id), None)
        if record is None:
            raise KeyError(f"Unknown job ID: {job_id}")
        return record

    def _queue_changed(self, snapshot: dict[str, object]) -> None:
        self.writer.send({"type": "event", "protocol": PROTOCOL_VERSION, "event": "queue.changed", "payload": snapshot})

    def _workflow_event(self, entry, event) -> None:
        self.writer.send(workflow_event_frame(entry, event), droppable=event.__class__.__name__ == "StageProgressEvent")


def serve(stdin: BinaryIO | None = None, stdout: TextIO | None = None) -> int:
    input_stream = stdin or sys.stdin.buffer
    wire_stdout = stdout or sys.stdout
    if stdout is None:
        sys.stdout = sys.stderr
    writer = JsonLineWriter(wire_stdout)
    writer.start()
    service = DesktopService(writer)
    service.start()
    writer.send({"type": "event", "protocol": PROTOCOL_VERSION, "event": "app.ready", "payload": hello_payload()})
    seen_ids: set[str] = set()
    try:
        try:
            frames = read_json_lines(input_stream)
            for payload in frames:
                request_id = payload.get("id", "invalid") if isinstance(payload, dict) else "invalid"
                request_id = request_id if isinstance(request_id, str) else "invalid"
                try:
                    request = parse_request(payload)
                    if request.request_id in seen_ids:
                        raise ProtocolError("DUPLICATE_REQUEST", f"Duplicate request id: {request.request_id}")
                    seen_ids.add(request.request_id)
                    if len(seen_ids) > 4096:
                        seen_ids = {request.request_id}
                    result = service.handle(request)
                    writer.send(success_response(request.request_id, json_value(result)))
                except Exception as exc:
                    writer.send(error_response(request_id, error_from_exception(exc)))
                if service.shutdown_requested:
                    break
        except Exception as exc:
            writer.send(error_response("invalid", error_from_exception(exc)))
    finally:
        service.close()
        writer.close()
    return 0


def _draft(params: dict[str, object]) -> WorkflowDraft:
    raw = params.get("draft")
    if not isinstance(raw, dict):
        raise ValueError("draft must be an object.")
    return WorkflowDraft.from_recipe(cast(dict[str, object], raw))


def _provider(params: dict[str, object]) -> str:
    provider = _required_string(params, "provider")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return provider


def _required_string(params: dict[str, object], key: str, *, max_length: int = 4096) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{key} must be a non-empty bounded string.")
    return value


def _deep_merge(base: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in patch.items():
        if key not in result:
            raise ValueError(f"Unknown settings field: {key}")
        current = result[key]
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(cast(dict[str, object], current), cast(dict[str, object], value))
        else:
            result[key] = deepcopy(value)
    return result
