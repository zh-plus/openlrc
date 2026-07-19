#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Versioned glossary loading, merging, matching, and compliance reporting."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GlossaryOrigin(str, Enum):
    """Supported glossary sources ordered from highest to lowest priority."""

    TASK = "task"
    BRIEF = "brief"


class GlossaryEntry(BaseModel):
    """One source-to-target terminology rule."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    aliases: list[str] = Field(default_factory=list)
    accepted_targets: list[str] = Field(default_factory=list)
    note: str = ""
    required: bool = False
    case_sensitive: bool = False
    enabled: bool = True

    @field_validator("source", "target")
    @classmethod
    def _nonempty_required_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("note")
    @classmethod
    def _normalize_note(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()

    @field_validator("aliases", "accepted_targets")
    @classmethod
    def _normalize_variants(cls, values: list[str]) -> list[str]:
        if not isinstance(values, list):
            raise ValueError("must be a list of strings")
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("must contain only non-empty strings")
            item = value.strip()
            if item not in normalized:
                normalized.append(item)
        return normalized


class GlossaryCatalog(BaseModel):
    """Portable JSON glossary schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = ""
    source_language: str | None = None
    target_language: str | None = None
    entries: list[GlossaryEntry] = Field(default_factory=list)


class ResolvedGlossaryEntry(GlossaryEntry):
    """Glossary entry after source priority and force rules are applied."""

    origin: GlossaryOrigin


class GlossaryConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    candidates: list[str]
    origins: list[GlossaryOrigin]
    selected_target: str
    reason: str


class GlossaryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_source: str
    occurrence_index: int | None = None
    matched_source: str | None = None
    segment_ids: list[int] = Field(default_factory=list)
    target_segment_ids: list[int] = Field(default_factory=list)
    target_found: bool | None = None
    origin: GlossaryOrigin
    severity: Literal["info", "warning", "error"]
    message: str


class GlossaryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_entries: int = 0
    enabled_entries: int = 0
    conflict_count: int = 0
    source_matched_entries: int = 0
    source_matched_segments: int = 0
    source_matched_occurrences: int = 0
    required_compliant: int = 0
    required_noncompliant: int = 0
    required_occurrences_compliant: int = 0
    required_occurrences_noncompliant: int = 0
    preferred_noncompliant: int = 0
    indeterminate: int = 0
    brief_entries: int = 0
    brief_entries_overridden: int = 0
    brief_characters_overridden: int = 0
    glossary_removed_retry_chunks: list[list[int]] = Field(default_factory=list)
    final_compliance_passed: bool = True


class GlossaryState(BaseModel):
    """Serializable effective glossary plus its audit information."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    fingerprint: str
    merged_entries: list[ResolvedGlossaryEntry] = Field(default_factory=list)
    conflicts: list[GlossaryConflict] = Field(default_factory=list)
    matches: list[GlossaryMatch] = Field(default_factory=list)
    metrics: GlossaryMetrics = Field(default_factory=GlossaryMetrics)

    def prompt_mapping(self) -> dict[str, str]:
        return {entry.source: entry.target for entry in self.merged_entries if entry.enabled}


@dataclass(frozen=True)
class _TextOccurrence:
    segment_ids: tuple[int, ...]
    start: int
    end: int
    variant: str


class GlossaryService:
    """Load task glossaries and merge them with Translation Brief terms."""

    def __init__(self, *, strict: bool = True, force: bool = False, report_matches: bool = True):
        self.strict = strict
        self.force = force
        self.report_matches = report_matches

    @staticmethod
    def _normalize_key(value: str, *, case_sensitive: bool = True) -> str:
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        return normalized if case_sensitive else normalized.casefold()

    @classmethod
    def _entries_overlap(cls, first: GlossaryEntry, second: GlossaryEntry) -> bool:
        case_sensitive = first.case_sensitive and second.case_sensitive
        first_variants = {
            cls._normalize_key(item, case_sensitive=case_sensitive) for item in [first.source, *first.aliases]
        }
        second_variants = {
            cls._normalize_key(item, case_sensitive=case_sensitive) for item in [second.source, *second.aliases]
        }
        return bool(first_variants & second_variants)

    @classmethod
    def _catalog_payload(cls, source: Any) -> tuple[dict[str, Any], str]:
        if isinstance(source, GlossaryCatalog):
            return source.model_dump(), "catalog"
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Glossary file not found: {path}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid glossary JSON at {path}: {exc}") from exc
            return payload, str(path)
        if isinstance(source, dict):
            return source, "mapping"
        raise TypeError("Glossary must be a mapping, GlossaryCatalog, or JSON path.")

    def load(
        self,
        source: dict | str | Path | GlossaryCatalog | None,
        *,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> tuple[GlossaryCatalog, list[GlossaryConflict]]:
        """Load legacy or versioned input and validate task-language declarations."""
        if source is None:
            return GlossaryCatalog(entries=[]), []

        payload, label = self._catalog_payload(source)
        if not isinstance(payload, dict):
            raise ValueError(f"Glossary {label} must contain a JSON object.")
        if "entries" not in payload and "schema_version" not in payload:
            entries: list[GlossaryEntry] = []
            for raw_source, raw_target in payload.items():
                if not isinstance(raw_source, str) or not isinstance(raw_target, str):
                    raise ValueError("Legacy glossary mappings must use string source and target values.")
                entries.append(GlossaryEntry(source=raw_source, target=raw_target))
            catalog = GlossaryCatalog(entries=entries)
        else:
            try:
                catalog = GlossaryCatalog.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"Invalid glossary schema in {label}: {exc}") from exc

        if source_language and catalog.source_language and catalog.source_language.lower() != source_language.lower():
            raise ValueError(
                f"Glossary source_language {catalog.source_language!r} does not match task language {source_language!r}."
            )
        if target_language and catalog.target_language and catalog.target_language.lower() != target_language.lower():
            raise ValueError(
                f"Glossary target_language {catalog.target_language!r} does not match task language {target_language!r}."
            )

        deduplicated: list[GlossaryEntry] = []
        conflicts: list[GlossaryConflict] = []
        for entry in catalog.entries:
            previous = next((item for item in deduplicated if self._entries_overlap(item, entry)), None)
            if previous is None:
                deduplicated.append(entry)
                continue
            if previous.target == entry.target:
                continue
            conflict = GlossaryConflict(
                source=entry.source,
                candidates=[previous.target, entry.target],
                origins=[GlossaryOrigin.TASK, GlossaryOrigin.TASK],
                selected_target=previous.target,
                reason="same-priority-conflict",
            )
            if self.strict:
                raise ValueError(
                    f"Conflicting task glossary entries for {entry.source!r}: {previous.target!r} vs {entry.target!r}."
                )
            conflicts.append(conflict)

        return catalog.model_copy(update={"entries": deduplicated}), conflicts

    @staticmethod
    def _brief_entries(brief: Any) -> list[GlossaryEntry]:
        if brief is None:
            return []
        raw_entries = getattr(brief, "glossary", brief)
        entries: list[GlossaryEntry] = []
        for raw in raw_entries or []:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            entries.append(
                GlossaryEntry(source=str(raw["source"]), target=str(raw["target"]), note=str(raw.get("note", "")))
            )
        return entries

    @classmethod
    def _entry_matches_source(cls, entry: GlossaryEntry, source: str) -> bool:
        """Return whether one glossary entry governs a Brief character name."""
        normalized_source = cls._normalize_key(source, case_sensitive=entry.case_sensitive)
        return normalized_source in {
            cls._normalize_key(variant, case_sensitive=entry.case_sensitive)
            for variant in [entry.source, *entry.aliases]
        }

    def merge(
        self, task_catalog: GlossaryCatalog, *, brief: Any = None, load_conflicts: Iterable[GlossaryConflict] = ()
    ) -> GlossaryState:
        """Resolve task-over-Brief priority and return a serializable state."""
        merged: list[ResolvedGlossaryEntry] = []
        task_entries: list[ResolvedGlossaryEntry] = []
        conflicts = list(load_conflicts)

        for entry in task_catalog.entries:
            payload = entry.model_dump()
            payload["required"] = bool(entry.required or self.force)
            resolved = ResolvedGlossaryEntry(**payload, origin=GlossaryOrigin.TASK)
            task_entries.append(resolved)
            merged.append(resolved)

        brief_count = 0
        overridden = 0
        for entry in self._brief_entries(brief):
            brief_count += 1
            selected = next((item for item in task_entries if self._entries_overlap(item, entry)), None)
            if selected is not None:
                overridden += 1
                if selected.target != entry.target:
                    conflicts.append(
                        GlossaryConflict(
                            source=entry.source,
                            candidates=[selected.target, entry.target],
                            origins=[GlossaryOrigin.TASK, GlossaryOrigin.BRIEF],
                            selected_target=selected.target,
                            reason="higher-priority-task-entry",
                        )
                    )
                continue
            merged.append(ResolvedGlossaryEntry(**entry.model_dump(), origin=GlossaryOrigin.BRIEF))

        # Character names live in a separate Brief section, but an explicit
        # task glossary is still authoritative. Record cross-section
        # disagreements here so the prompt renderer can use the same priority
        # rule without silently hiding what was overridden.
        character_overrides = 0
        for character in getattr(brief, "characters", []) if brief is not None else []:
            source_name = str(getattr(character, "source_name", ""))
            target_name = str(getattr(character, "target_name", ""))
            selected = next(
                (entry for entry in task_entries if entry.enabled and self._entry_matches_source(entry, source_name)),
                None,
            )
            if selected is None:
                continue
            character_overrides += 1
            if self._normalize_key(selected.target) != self._normalize_key(target_name):
                conflicts.append(
                    GlossaryConflict(
                        source=source_name,
                        candidates=[selected.target, target_name],
                        origins=[GlossaryOrigin.TASK, GlossaryOrigin.BRIEF],
                        selected_target=selected.target,
                        reason="higher-priority-task-entry-over-character",
                    )
                )

        fingerprint = self.fingerprint(merged)
        return GlossaryState(
            fingerprint=fingerprint,
            merged_entries=merged,
            conflicts=conflicts,
            metrics=GlossaryMetrics(
                total_entries=len(merged),
                enabled_entries=sum(entry.enabled for entry in merged),
                conflict_count=len(conflicts),
                brief_entries=brief_count,
                brief_entries_overridden=overridden,
                brief_characters_overridden=character_overrides,
            ),
        )

    @staticmethod
    def fingerprint(entries: Iterable[ResolvedGlossaryEntry]) -> str:
        payload = [entry.model_dump(mode="json") for entry in entries]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _normalized_text(value: str, *, case_sensitive: bool) -> str:
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        return normalized if case_sensitive else normalized.casefold()

    @classmethod
    def _contains(cls, text: str, phrase: str, *, case_sensitive: bool) -> bool:
        haystack = cls._normalized_text(text, case_sensitive=case_sensitive)
        needle = cls._normalized_text(phrase, case_sensitive=case_sensitive)
        if not needle:
            return False
        if any("\u4e00" <= char <= "\u9fff" for char in needle):
            return needle in haystack
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])", haystack) is not None

    @classmethod
    def _find_spans(cls, text: str, phrase: str, *, case_sensitive: bool) -> list[tuple[int, int]]:
        haystack = cls._normalized_text(text, case_sensitive=case_sensitive)
        needle = cls._normalized_text(phrase, case_sensitive=case_sensitive)
        if not needle:
            return []
        pattern = re.escape(needle)
        if not any("\u4e00" <= char <= "\u9fff" for char in needle):
            pattern = rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])"
        return [(match.start(), match.end()) for match in re.finditer(pattern, haystack)]

    @staticmethod
    def _overlaps(first: _TextOccurrence, second: _TextOccurrence) -> bool:
        return first.segment_ids == second.segment_ids and first.start < second.end and second.start < first.end

    @classmethod
    def _text_occurrences(cls, texts: list[str], variants: list[str], *, case_sensitive: bool) -> list[_TextOccurrence]:
        occurrences: list[_TextOccurrence] = []

        def add(candidate: _TextOccurrence) -> None:
            if not any(cls._overlaps(candidate, existing) for existing in occurrences):
                occurrences.append(candidate)

        for index, text in enumerate(texts, 1):
            for variant in variants:
                for start, end in cls._find_spans(text, variant, case_sensitive=case_sensitive):
                    add(_TextOccurrence((index,), start, end, variant))

        for index in range(len(texts) - 1):
            left = cls._normalized_text(texts[index], case_sensitive=case_sensitive)
            right = cls._normalized_text(texts[index + 1], case_sensitive=case_sensitive)
            for separator in ("", " "):
                joined = left + separator + right
                left_end = len(left)
                right_start = left_end + len(separator)
                for variant in variants:
                    for start, end in cls._find_spans(joined, variant, case_sensitive=case_sensitive):
                        if start < left_end and end > right_start:
                            add(_TextOccurrence((index + 1, index + 2), start, end, variant))

        return sorted(occurrences, key=lambda item: (item.segment_ids, item.start, item.end))

    def report_state(self, state: GlossaryState) -> GlossaryState:
        """Return the configured public/report representation without changing validation state."""
        return state if self.report_matches else state.model_copy(update={"matches": []})

    def check(
        self,
        state: GlossaryState,
        source_texts: list[str],
        target_texts: list[str] | None = None,
        *,
        glossary_removed_retry_chunks: Iterable[list[int]] = (),
    ) -> GlossaryState:
        """Match effective entries against source and optional target subtitles."""
        if target_texts is not None and len(source_texts) != len(target_texts):
            raise ValueError("Source and target subtitle counts must match for glossary checking.")

        matches: list[GlossaryMatch] = []
        matched_segment_ids: set[int] = set()
        matched_entries: set[str] = set()
        source_occurrence_count = 0
        required_compliant = 0
        required_noncompliant = 0
        preferred_noncompliant = 0
        indeterminate = 0

        for entry in state.merged_entries:
            if not entry.enabled:
                continue
            variants = [entry.source, *entry.aliases]
            source_occurrences = self._text_occurrences(source_texts, variants, case_sensitive=entry.case_sensitive)
            if not source_occurrences:
                matches.append(
                    GlossaryMatch(
                        entry_source=entry.source,
                        origin=entry.origin,
                        severity="info",
                        message="Source term does not occur in the subtitle.",
                    )
                )
                continue
            matched_entries.add(entry.source)
            source_occurrence_count += len(source_occurrences)
            matched_segment_ids.update(line_id for item in source_occurrences for line_id in item.segment_ids)
            target_occurrences = (
                self._text_occurrences(
                    target_texts, [entry.target, *entry.accepted_targets], case_sensitive=entry.case_sensitive
                )
                if target_texts is not None
                else []
            )
            used_targets: list[_TextOccurrence] = []

            for occurrence_index, occurrence in enumerate(source_occurrences, 1):
                if target_texts is None:
                    indeterminate += 1
                    matches.append(
                        GlossaryMatch(
                            entry_source=entry.source,
                            occurrence_index=occurrence_index,
                            matched_source=occurrence.variant,
                            segment_ids=list(occurrence.segment_ids),
                            target_found=None,
                            origin=entry.origin,
                            severity="info",
                            message="Source occurrence matched; target compliance was not requested.",
                        )
                    )
                    continue

                target_ids: set[int] = set()
                for line_id in occurrence.segment_ids:
                    target_ids.update(range(max(1, line_id - 1), min(len(target_texts), line_id + 1) + 1))
                matched_target = next(
                    (
                        candidate
                        for candidate in target_occurrences
                        if set(candidate.segment_ids).issubset(target_ids)
                        and not any(self._overlaps(candidate, used) for used in used_targets)
                    ),
                    None,
                )
                target_found = matched_target is not None
                if matched_target is not None:
                    used_targets.append(matched_target)
                if target_found:
                    if entry.required:
                        required_compliant += 1
                    severity: Literal["info", "warning", "error"] = "info"
                    message = "Target terminology is compliant for this source occurrence."
                elif entry.required:
                    required_noncompliant += 1
                    severity = "error"
                    message = f"Required target term {entry.target!r} was not found for this source occurrence."
                else:
                    preferred_noncompliant += 1
                    severity = "warning"
                    message = f"Preferred target term {entry.target!r} was not found for this source occurrence."
                matches.append(
                    GlossaryMatch(
                        entry_source=entry.source,
                        occurrence_index=occurrence_index,
                        matched_source=occurrence.variant,
                        segment_ids=list(occurrence.segment_ids),
                        target_segment_ids=list(matched_target.segment_ids) if matched_target else [],
                        target_found=target_found,
                        origin=entry.origin,
                        severity=severity,
                        message=message,
                    )
                )

        metrics = state.metrics.model_copy(
            update={
                "source_matched_entries": len(matched_entries),
                "source_matched_segments": len(matched_segment_ids),
                "source_matched_occurrences": source_occurrence_count,
                "required_compliant": required_compliant,
                "required_noncompliant": required_noncompliant,
                "required_occurrences_compliant": required_compliant,
                "required_occurrences_noncompliant": required_noncompliant,
                "preferred_noncompliant": preferred_noncompliant,
                "indeterminate": indeterminate,
                "glossary_removed_retry_chunks": [list(item) for item in glossary_removed_retry_chunks],
                "final_compliance_passed": required_noncompliant == 0,
            }
        )
        return state.model_copy(update={"matches": matches, "metrics": metrics})
