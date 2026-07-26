"""Small, synchronous preflight for interactive Workflow frontends."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from openlrc.application.credentials import CredentialStore
from openlrc.application.drafts import WorkflowDraft, normalize_input_paths
from openlrc.application.settings import AppSettings
from openlrc.defaults import PREPROCESSED_DIR, PREPROCESSED_SUFFIX, TRANSCRIBED_SUFFIX
from openlrc.llama_resources import resolve_llama_model_path, resolve_llama_server
from openlrc.whisper_resources import resolve_vad_model_path, resolve_whisper_cli, resolve_whisper_model_path
from openlrc.workflow import RunRequest, TranscribeRequest, WorkflowKind, resolve_run_execution_strategy
from openlrc.workflow.planning import plan_request_paths


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    severity: str
    message: str
    field: str = ""


@dataclass(slots=True)
class PreflightReport:
    issues: list[PreflightIssue] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return any(issue.severity == "blocked" for issue in self.issues)

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def status(self) -> str:
        if self.blocked:
            return "Blocked"
        if self.warnings:
            return "Warning"
        return "Ready"


def preflight(draft: WorkflowDraft, settings: AppSettings, credentials: CredentialStore) -> PreflightReport:
    """Validate only the selected pipeline without network requests or model scans."""
    report = PreflightReport()
    paths, path_issues = normalize_input_paths(draft.paths, draft.workflow)
    for message in path_issues:
        severity = "warning" if message.startswith("Duplicate ignored:") else "blocked"
        report.issues.append(PreflightIssue(severity, message, "inputs"))
    if not paths:
        report.issues.append(PreflightIssue("blocked", "Select at least one input file.", "inputs"))

    if draft.workflow == WorkflowKind.TRANSLATE.value:
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_file() and path.suffix.lower() == ".json":
                error = _transcription_json_error(path)
                if error:
                    report.issues.append(PreflightIssue("blocked", error, "inputs"))

    request = None
    try:
        checked = WorkflowDraft.from_recipe(draft.to_recipe())
        checked.paths = paths
        request = checked.build_request(settings, credentials)
    except Exception as exc:
        report.issues.append(PreflightIssue("blocked", str(exc), "pipeline"))

    if request is not None:
        _check_resources(draft, request, settings, report)
        plans: dict[str, dict[str, tuple[Path, ...]]] = {}
        try:
            plans = plan_request_paths(request)
        except Exception as exc:
            report.issues.append(PreflightIssue("blocked", f"Cannot plan output paths: {exc}", "inputs"))
        else:
            _check_path_plans(plans, report)
        if isinstance(request, TranscribeRequest):
            strategy = "transcribe-only"
        elif isinstance(request, RunRequest):
            strategy = resolve_run_execution_strategy(request).value
        else:
            strategy = "translation-only"
        report.summary = {
            "pipeline": _pipeline_label(draft.workflow),
            "inputs": str(len(paths)),
            "translation": (
                "None" if draft.workflow == "transcribe" else f"{draft.translation_backend.title()} / {draft.mode}"
            ),
            "strategy": strategy,
            "target": draft.target_language if draft.workflow != "transcribe" else "source language",
            "cleanup": "owned temporary files only" if draft.clear_temp else "keep temporary files",
            "outputs": ", ".join(sorted({str(path.parent) for plan in plans.values() for path in plan["outputs"]})),
        }
    return report


def _check_resources(draft: WorkflowDraft, request, settings: AppSettings, report: PreflightReport) -> None:
    if draft.workflow in {WorkflowKind.TRANSCRIBE.value, WorkflowKind.RUN.value}:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            report.issues.append(
                PreflightIssue("blocked", "ffmpeg and ffprobe must be available on PATH.", "transcription")
            )
        _resolve_resource(
            lambda: resolve_whisper_cli(settings.local_models.whisper_cli), "whisper-cli", "transcription", report
        )
        _resolve_resource(
            lambda: resolve_whisper_model_path(draft.whisper_model or settings.local_models.whisper_model),
            "Whisper model",
            "transcription",
            report,
        )
        vad_model = draft.vad_model
        if vad_model:
            _resolve_resource(lambda: resolve_vad_model_path(vad_model), "Whisper VAD model", "transcription", report)

    if draft.workflow == WorkflowKind.TRANSCRIBE.value or draft.translation_backend not in {"local", "local-qwen"}:
        return
    _resolve_resource(
        lambda: resolve_llama_server(settings.local_models.llama_server), "llama-server", "translation", report
    )
    _resolve_resource(
        lambda: resolve_llama_model_path(request.translation.config.local_llm.model_path),
        "Local translation model",
        "translation",
        report,
    )
    if draft.context_provider == "local" and draft.context_model:
        _resolve_resource(
            lambda: resolve_llama_model_path(draft.context_model), "Context model", "context_model", report
        )


def _resolve_resource(resolver, label: str, field_name: str, report: PreflightReport) -> None:
    try:
        resolver()
    except Exception as exc:
        report.issues.append(PreflightIssue("blocked", f"{label}: {exc}", field_name))


def _check_path_plans(plans: dict[str, dict[str, tuple[Path, ...]]], report: PreflightReport) -> None:
    owners: dict[Path, str] = {}
    for item, plan in plans.items():
        for sidecar in plan["sidecars"]:
            if sidecar.exists():
                report.issues.append(
                    PreflightIssue("blocked", f"Existing sidecar WAV is not owned by this run: {sidecar}", "inputs")
                )
        for category in ("temporary", "recovery", "outputs"):
            for raw_path in plan[category]:
                path = raw_path.resolve(strict=False)
                previous = owners.get(path)
                if previous is not None and previous != item:
                    report.issues.append(
                        PreflightIssue(
                            "blocked", f"Inputs {previous} and {item} map to the same {category} path: {path}", "inputs"
                        )
                    )
                else:
                    owners[path] = item
                if not path.exists():
                    if category == "outputs" and (not path.parent.is_dir() or not os.access(path.parent, os.W_OK)):
                        report.issues.append(
                            PreflightIssue("blocked", f"Output directory is not writable: {path.parent}", "outputs")
                        )
                    continue
                if category == "outputs":
                    report.issues.append(
                        PreflightIssue("warning", f"Existing output will be replaced: {path}", "outputs")
                    )
                elif category == "temporary":
                    report.issues.append(
                        PreflightIssue("warning", f"Existing cache will be reused and preserved: {path}", "cleanup")
                    )
                else:
                    report.issues.append(
                        PreflightIssue("warning", f"Existing recovery file may be resumed: {path}", "recovery")
                    )


def _transcription_json_error(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"Invalid transcription JSON {path}: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        return f"Invalid transcription JSON {path}: expected an object with a segments array."
    if not isinstance(payload.get("language"), str) or not payload["language"].strip():
        return f"Invalid transcription JSON {path}: language must be a non-empty string."
    for index, segment in enumerate(payload["segments"]):
        if not isinstance(segment, dict) or not {"start", "end", "text"}.issubset(segment):
            return f"Invalid transcription JSON {path}: segment {index} needs start, end, and text."
        if not isinstance(segment["text"], str):
            return f"Invalid transcription JSON {path}: segment {index} text must be a string."
        if not isinstance(segment["start"], (int, float)) or not isinstance(segment["end"], (int, float)):
            return f"Invalid transcription JSON {path}: segment {index} timestamps must be numeric."
        if segment["start"] < 0 or segment["end"] < segment["start"]:
            return f"Invalid transcription JSON {path}: segment {index} timestamps are out of order."
    expected_suffix = f"{PREPROCESSED_SUFFIX}{TRANSCRIBED_SUFFIX}"
    if path.parent.name != PREPROCESSED_DIR or not path.stem.endswith(expected_suffix):
        return (
            f"Invalid transcription candidate {path}: choose an OpenLRC "
            f"{expected_suffix}.json file inside a {PREPROCESSED_DIR} directory."
        )
    return None


def _pipeline_label(value: str) -> str:
    return {"transcribe": "Transcribe", "translate": "Translation", "run": "Total Run"}.get(value, value)
