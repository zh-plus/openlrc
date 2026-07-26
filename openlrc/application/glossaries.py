"""Typed glossary inspection for Workflow, Settings, and Job detail views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openlrc.application.history import JobRecord
from openlrc.glossary import GlossaryCatalog, GlossaryConflict, GlossaryService, GlossaryState


@dataclass(frozen=True, slots=True)
class GlossaryInspection:
    path: Path | None
    catalog: GlossaryCatalog
    state: GlossaryState
    conflicts: tuple[GlossaryConflict, ...]

    @property
    def entry_count(self) -> int:
        return len(self.state.merged_entries)

    @property
    def violation_count(self) -> int:
        return sum(match.severity in {"warning", "error"} for match in self.state.matches)


class GlossaryApplicationService:
    """Expose glossary data without rendering CLI tables or parsing console output."""

    def inspect(
        self,
        source: str | Path,
        *,
        source_language: str | None = None,
        target_language: str | None = None,
        strict: bool = True,
        force: bool = False,
    ) -> GlossaryInspection:
        path = Path(source).expanduser().resolve(strict=False)
        service = GlossaryService(strict=strict, force=force)
        catalog, conflicts = service.load(
            path, source_language=source_language or None, target_language=target_language or None
        )
        state = service.merge(catalog, load_conflicts=conflicts)
        return GlossaryInspection(path, catalog, state, tuple(conflicts))

    def for_job(self, record: JobRecord) -> GlossaryInspection | None:
        for artifact in record.artifacts:
            raw_path = artifact.get("path")
            if not isinstance(raw_path, str):
                continue
            state = _read_glossary_state(Path(raw_path))
            if state is not None:
                return GlossaryInspection(None, GlossaryCatalog(entries=[]), state, tuple(state.conflicts))
        glossary_path = record.recipe.get("glossary_path")
        if isinstance(glossary_path, str) and glossary_path.strip():
            try:
                return self.inspect(
                    glossary_path,
                    source_language=str(record.recipe.get("source_language") or ""),
                    target_language=str(record.recipe.get("target_language") or ""),
                    strict=bool(record.recipe.get("glossary_strict", True)),
                    force=bool(record.recipe.get("force_glossary", False)),
                )
            except Exception:
                return None
        return None


def _read_glossary_state(path: Path) -> GlossaryState | None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidates = [payload.get("glossary_state")]
    edit_session = payload.get("edit_session")
    if isinstance(edit_session, dict):
        candidates.append(edit_session.get("glossary_state"))
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            continue
        try:
            return GlossaryState.model_validate(candidate)
        except Exception:
            continue
    return None
