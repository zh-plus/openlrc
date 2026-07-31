#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Auditable subtitle edit models, transactions, reports, and restore helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openlrc.checkpoint import atomic_write_text, save_json_checkpoint


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class EditSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EditIssueStatus(StrEnum):
    DETECTED = "detected"
    PROPOSED = "proposed"
    APPLIED = "applied"
    VERIFIED = "verified"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    FAILED = "failed"


class EditAction(StrEnum):
    VERIFY = "verify"
    REVIEW = "review"
    RETRANSLATE = "retranslate"
    RESTORE = "restore"


class EditSessionStatus(StrEnum):
    CHECKING = "checking"
    EDITING = "editing"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class EditStopReason(StrEnum):
    NO_HIGH_RISK = "no-high-risk"
    NO_EFFECTIVE_PATCH = "no-effective-patch"
    UNCHANGED_HASH = "unchanged-hash"
    CHECKS_PASSED = "checks-passed"
    MAX_ROUNDS = "max-rounds"
    MODEL_FAILURE = "model-failure"
    VALIDATION_FAILURE = "validation-failure"


class EditRoundStatus(StrEnum):
    STARTED = "started"
    FAILED = "failed"
    COMMITTED = "committed"
    COMPLETED = "completed"


class EditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    segment_ids: list[int]
    category: str
    severity: EditSeverity
    source: str
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: EditIssueStatus = EditIssueStatus.DETECTED

    @classmethod
    def create(
        cls,
        *,
        segment_ids: Iterable[int],
        category: str,
        severity: EditSeverity | str,
        source: str,
        message: str,
        evidence: dict[str, Any] | None = None,
        status: EditIssueStatus = EditIssueStatus.DETECTED,
    ) -> EditIssue:
        ids = sorted(set(int(item) for item in segment_ids))
        evidence = evidence or {}
        payload = {
            "segment_ids": ids,
            "category": category,
            "severity": EditSeverity(severity).value,
            "source": source,
            "message": message,
            "evidence": evidence,
        }
        return cls(
            issue_id=_stable_hash(payload)[:20],
            segment_ids=ids,
            category=category,
            severity=EditSeverity(severity),
            source=source,
            message=message,
            evidence=evidence,
            status=status,
        )


class EditPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch_id: str
    issue_ids: list[str] = Field(default_factory=list)
    segment_id: int = Field(ge=1)
    before: str
    after: str
    reason: str
    action: EditAction
    rejection_reason: str | None = None

    @field_validator("before", "after", "reason")
    @classmethod
    def _require_string(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value

    @classmethod
    def create(
        cls,
        *,
        issue_ids: Iterable[str],
        segment_id: int,
        before: str,
        after: str,
        reason: str,
        action: EditAction | str,
    ) -> EditPatch:
        payload = {
            "issue_ids": sorted(set(issue_ids)),
            "segment_id": segment_id,
            "before": before,
            "after": after,
            "reason": reason,
            "action": EditAction(action).value,
        }
        return cls(patch_id=_stable_hash(payload)[:20], **payload)


class EditRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=0)
    status: EditRoundStatus = EditRoundStatus.STARTED
    scope_ids: list[int]
    issues_before: list[EditIssue] = Field(default_factory=list)
    proposed_patches: list[EditPatch] = Field(default_factory=list)
    applied_patches: list[EditPatch] = Field(default_factory=list)
    rejected_patches: list[EditPatch] = Field(default_factory=list)
    validation_after: list[EditIssue] = Field(default_factory=list)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    stop_reason: EditStopReason | None = None

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_status(cls, value):
        if not isinstance(value, dict) or "status" in value:
            return value
        payload = dict(value)
        stop_reason = payload.get("stop_reason")
        stop_value = stop_reason.value if isinstance(stop_reason, EditStopReason) else stop_reason
        if stop_value in {
            EditStopReason.MODEL_FAILURE.value,
            EditStopReason.VALIDATION_FAILURE.value,
            EditStopReason.NO_EFFECTIVE_PATCH.value,
            EditStopReason.UNCHANGED_HASH.value,
        }:
            payload["status"] = EditRoundStatus.FAILED.value
        elif payload.get("applied_patches"):
            payload["status"] = EditRoundStatus.COMMITTED.value
        elif stop_value is not None:
            payload["status"] = EditRoundStatus.COMPLETED.value
        return payload


class EditSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_fingerprint: str
    translation_fingerprint: str
    glossary_fingerprint: str = ""
    status: EditSessionStatus = EditSessionStatus.CHECKING
    max_rounds: int = Field(default=1, ge=0, le=3)
    restore_enabled: bool = False
    raw_translations: list[str] = Field(default_factory=list)
    current_translations: list[str] = Field(default_factory=list)
    source_timestamps: list[tuple[float, float | None]] = Field(default_factory=list)
    rounds: list[EditRound] = Field(default_factory=list)
    unresolved_issues: list[EditIssue] = Field(default_factory=list)
    glossary_state: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class PatchTransactionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    committed: bool
    translations: list[str]
    applied_patches: list[EditPatch] = Field(default_factory=list)
    rejected_patches: list[EditPatch] = Field(default_factory=list)
    validation_after: list[EditIssue] = Field(default_factory=list)
    reason: str | None = None


class EditResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    action: EditAction
    output_path: Path
    report_path: Path
    session_path: Path | None = None
    changed_ids: list[int] = Field(default_factory=list)
    rounds: list[EditRound] = Field(default_factory=list)
    unresolved_issues: list[EditIssue] = Field(default_factory=list)


def subtitle_fingerprint(
    texts: list[str], timestamps: list[tuple[float, float | None]], *, language: str | None = None
) -> str:
    return _stable_hash({"language": language, "texts": texts, "timestamps": timestamps})


def source_fingerprint(texts: list[str], timestamps: list[tuple[float, float | None]], *, language: str) -> str:
    return subtitle_fingerprint(texts, timestamps, language=language)


def issue_identity(issue: EditIssue) -> tuple[str, tuple[int, ...], str]:
    """Stable comparison key used to prevent new or worsened hard errors."""
    return issue.category, tuple(issue.segment_ids), issue.source


def _rejected(patch: EditPatch, reason: str) -> EditPatch:
    return patch.model_copy(update={"rejection_reason": reason})


def apply_patch_transaction(
    translations: list[str],
    patches: list[EditPatch],
    *,
    scope_ids: Iterable[int],
    baseline_issues: list[EditIssue],
    validate: Callable[[list[str]], list[EditIssue]],
) -> PatchTransactionResult:
    """Apply one all-or-nothing patch round and reject stale or unsafe output."""
    scope = set(int(item) for item in scope_ids)
    seen_ids: set[int] = set()

    def reject_all(reason: str, validation_after: list[EditIssue] | None = None) -> PatchTransactionResult:
        return PatchTransactionResult(
            committed=False,
            translations=list(translations),
            rejected_patches=[_rejected(item, reason) for item in patches],
            validation_after=validation_after or [],
            reason=reason,
        )

    for patch in patches:
        if patch.segment_id not in scope:
            return reject_all(f"Patch segment {patch.segment_id} is outside the authorized scope.")
        if patch.segment_id in seen_ids:
            return reject_all(f"Multiple patches target segment {patch.segment_id} in one round.")
        seen_ids.add(patch.segment_id)
        if patch.segment_id > len(translations):
            return reject_all(f"Patch segment {patch.segment_id} does not exist.")
        if translations[patch.segment_id - 1] != patch.before:
            return reject_all(f"Patch segment {patch.segment_id} has a stale before value.")
        try:
            from openlrc.hymt2_pipeline import ReviewRevisionValidator

            ReviewRevisionValidator.normalize(patch.after)
        except ValueError as exc:
            return reject_all(f"Patch segment {patch.segment_id} has an invalid revision: {exc}")

    candidate = list(translations)
    for patch in patches:
        candidate[patch.segment_id - 1] = patch.after
    if candidate == translations:
        return reject_all("Patch round did not change any translation.")

    validation_after = validate(candidate)
    baseline_errors = {issue.issue_id: issue for issue in baseline_issues if issue.severity is EditSeverity.ERROR}
    candidate_errors = [issue for issue in validation_after if issue.severity is EditSeverity.ERROR]
    new_errors = [issue for issue in candidate_errors if issue.issue_id not in baseline_errors]
    addressed = {issue_id for patch in patches for issue_id in patch.issue_ids}
    unresolved_addressed = [issue for issue in candidate_errors if issue.issue_id in addressed]
    if new_errors or unresolved_addressed:
        return reject_all("Patch round introduced or failed to resolve a hard validation issue.", validation_after)

    return PatchTransactionResult(
        committed=True, translations=candidate, applied_patches=patches, validation_after=validation_after
    )


def save_edit_session(path: Path, session: EditSession) -> None:
    save_json_checkpoint(path, session.model_dump(mode="json"))


def load_edit_session(path: Path) -> EditSession:
    if not path.exists():
        raise FileNotFoundError(f"Edit session not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EditSession.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid edit session at {path}: {exc}") from exc


def translations_at_round(session: EditSession, round_index: int) -> list[str]:
    """Rebuild a historical translation snapshot without trusting redundant copies."""
    successful = {
        item.round_index
        for item in session.rounds
        if item.round_index > 0 and item.status in {EditRoundStatus.COMMITTED, EditRoundStatus.COMPLETED}
    }
    available = {0, *successful}
    if round_index not in available:
        available_text = ", ".join(str(item) for item in sorted(available))
        raise ValueError(f"Restore round must be one of: {available_text}.")
    output = list(session.raw_translations)
    if round_index == 0:
        return output
    for edit_round in sorted(session.rounds, key=lambda item: item.round_index):
        if edit_round.round_index > round_index:
            break
        if edit_round.round_index > 0 and edit_round.status not in {
            EditRoundStatus.COMMITTED,
            EditRoundStatus.COMPLETED,
        }:
            continue
        for patch in edit_round.applied_patches:
            if patch.segment_id > len(output) or output[patch.segment_id - 1] != patch.before:
                raise ValueError("Edit session patch history is inconsistent and cannot be restored safely.")
            output[patch.segment_id - 1] = patch.after
    return output


def atomic_save_subtitle(
    path: Path, *, language: str, timestamps: list[tuple[float, float | None]], texts: list[str]
) -> None:
    if len(timestamps) != len(texts):
        raise ValueError("Subtitle timestamps and texts must have equal length.")
    payload = {
        "language": language,
        "segments": [{"start": start, "end": end, "text": text} for (start, end), text in zip(timestamps, texts)],
    }
    save_json_checkpoint(path, payload)


def write_edit_report(path: Path, session: EditSession, *, markdown: bool = False) -> None:
    """Write a stable JSON or concise Markdown audit report."""
    if not markdown:
        save_json_checkpoint(path, session.model_dump(mode="json"))
        return

    lines = [
        "# OpenLRC Mac Edit Report",
        "",
        f"- Status: `{session.status.value}`",
        f"- Rounds: {len(session.rounds)} / {session.max_rounds}",
        f"- Unresolved issues: {len(session.unresolved_issues)}",
        "",
    ]
    for edit_round in session.rounds:
        lines.extend(
            [
                f"## Round {edit_round.round_index}",
                "",
                f"- Scope: {', '.join(map(str, edit_round.scope_ids)) or 'none'}",
                f"- Applied patches: {len(edit_round.applied_patches)}",
                f"- Rejected patches: {len(edit_round.rejected_patches)}",
                f"- Stop reason: `{edit_round.stop_reason.value if edit_round.stop_reason else ''}`",
                "",
            ]
        )
        for patch in edit_round.applied_patches:
            lines.extend(
                [
                    f"### Segment {patch.segment_id}",
                    "",
                    f"- Before: {patch.before}",
                    f"- After: {patch.after}",
                    f"- Reason: {patch.reason}",
                    "",
                ]
            )
    atomic_write_text(path, "\n".join(lines))


def parse_segment_ids(value: str, *, maximum: int | None = None) -> list[int]:
    """Parse `12,18-24` style CLI ranges into sorted unique IDs."""
    ids: set[int] = set()
    if not value.strip():
        raise ValueError("At least one subtitle ID is required.")
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError("Subtitle ID list contains an empty item.")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid subtitle ID range: {token!r}.")
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise ValueError(f"Invalid subtitle ID range: {token!r}.")
            ids.update(range(start, end + 1))
        else:
            if not token.isdigit() or int(token) < 1:
                raise ValueError(f"Invalid subtitle ID: {token!r}.")
            ids.add(int(token))
    result = sorted(ids)
    if maximum is not None and result[-1] > maximum:
        raise ValueError(f"Subtitle ID {result[-1]} exceeds the available segment count {maximum}.")
    return result
