#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Crash-safe JSON checkpoint persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

HYM_T2_CHECKPOINT_KIND = "hymt2_pipeline"
HYM_T2_CHECKPOINT_SCHEMA_VERSION = 4
STANDALONE_EDIT_CHECKPOINT_KIND = "standalone_edit"
STANDALONE_EDIT_CHECKPOINT_SCHEMA_VERSION = 1


def save_json_checkpoint(path: Path, data: dict) -> None:
    """Atomically replace *path* with a fully written JSON checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    """Atomically replace a UTF-8 text artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temp_path = Path(file.name)
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def migrate_hymt2_checkpoint(
    payload: dict,
    *,
    canonical_mode: str,
    canonical_fingerprint: str,
    accepted_legacy_fingerprints: set[str] | None = None,
    glossary_fingerprint: str = "",
) -> dict:
    """Migrate contextual Hy-MT2 v2/v3 state to canonical checkpoint v4.

    The caller computes legacy fingerprints from the same current inputs. This
    prevents a renamed mode from invalidating compatible state without accepting
    checkpoints whose source/model inputs have actually changed.
    """
    if not payload:
        return {}
    accepted = {canonical_fingerprint, *(accepted_legacy_fingerprints or set())}
    existing_fingerprint = payload.get("source_fingerprint")
    if existing_fingerprint not in accepted:
        return {}

    schema_version = payload.get("schema_version")
    checkpoint_kind = payload.get("checkpoint_kind")
    if checkpoint_kind == HYM_T2_CHECKPOINT_KIND and schema_version == HYM_T2_CHECKPOINT_SCHEMA_VERSION:
        migrated = dict(payload)
        migrated.update(mode=canonical_mode, hymt2_mode=canonical_mode, source_fingerprint=canonical_fingerprint)
        return migrated
    if schema_version not in {2, 3}:
        return {}

    migrated = dict(payload)
    migrated.update(
        checkpoint_kind=HYM_T2_CHECKPOINT_KIND,
        schema_version=HYM_T2_CHECKPOINT_SCHEMA_VERSION,
        mode=canonical_mode,
        hymt2_mode=canonical_mode,
        source_fingerprint=canonical_fingerprint,
        glossary_fingerprint=glossary_fingerprint,
        edit_protocol_version=1,
    )
    if "edit_session" not in migrated and payload.get("review_results"):
        migrated["edit_session"] = _legacy_review_session(payload, glossary_fingerprint=glossary_fingerprint)
    return migrated


def _legacy_review_session(payload: dict, *, glossary_fingerprint: str) -> dict:
    """Adapt real legacy review facts without inventing unavailable evidence."""
    from openlrc.editing import (
        EditAction,
        EditIssue,
        EditIssueStatus,
        EditPatch,
        EditRound,
        EditSession,
        EditSessionStatus,
        EditSeverity,
    )

    raw = list(payload.get("raw_hymt2_translations") or [])
    final = list(payload.get("final_translations") or raw)
    issues = []
    patches = []
    scope_ids: set[int] = set()
    review_results = payload.get("review_results") or {}
    for chunk_items in review_results.values():
        if not isinstance(chunk_items, list):
            continue
        for item in chunk_items:
            if not isinstance(item, dict) or item.get("risk") != "high":
                continue
            try:
                segment_id = int(item["id"])
                before = raw[segment_id - 1]
                after = str(item["revised_translation"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            message = "; ".join(str(value) for value in item.get("issues", [])) or "Legacy high-risk review"
            issue = EditIssue.create(
                segment_ids=[segment_id],
                category="legacy-semantic",
                severity=EditSeverity.ERROR,
                source="review-protocol-v2",
                message=message,
                evidence={},
                status=EditIssueStatus.RESOLVED,
            )
            patch = EditPatch.create(
                issue_ids=[issue.issue_id],
                segment_id=segment_id,
                before=before,
                after=after,
                reason=message,
                action=EditAction.REVIEW,
            )
            issues.append(issue)
            patches.append(patch)
            scope_ids.add(segment_id)

    incomplete = bool(payload.get("review_incomplete"))
    edit_round = EditRound(
        round_index=1,
        scope_ids=sorted(scope_ids),
        issues_before=issues,
        proposed_patches=patches,
        applied_patches=patches,
        model_metadata={"legacy_review_protocol_version": payload.get("review_protocol_version")},
    )
    import hashlib

    translation_fingerprint = hashlib.sha256(
        json.dumps(final, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EditSession(
        source_fingerprint=str(payload.get("source_fingerprint", "")),
        translation_fingerprint=translation_fingerprint,
        glossary_fingerprint=glossary_fingerprint,
        status=EditSessionStatus.INCOMPLETE if incomplete else EditSessionStatus.COMPLETE,
        raw_translations=raw,
        current_translations=final,
        rounds=[edit_round],
    ).model_dump(mode="json")
