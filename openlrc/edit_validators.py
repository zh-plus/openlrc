#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Offline deterministic validators for auditable subtitle editing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from openlrc.context import TranslationBrief
from openlrc.editing import EditIssue, EditSeverity
from openlrc.glossary import GlossaryService, GlossaryState
from openlrc.subtitle import Subtitle

_PLACEHOLDER_RE = re.compile(
    r"\{\{[^{}]+}}|\{[^{}]+}|%\([^)]+\)[#0 +\-]?[0-9]*(?:\.[0-9]+)?[a-zA-Z]"
    r"|%[#0+\-]?[0-9]*(?:\.[0-9]+)?[a-zA-Z]"
)
_NUMBER_RE = re.compile(r"(?<!\d)[-+]?\d+(?:[,.]\d+)*(?!\d)")
_PERCENT_RE = re.compile(r"(?:%|％|٪|\bpercent\b|\bper\s+cent\b|百分之)", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"(?:US\$|A\$|C\$|[$€£¥￥]|\b(?:USD|EUR|GBP|CNY|RMB|JPY|AUD|CAD)\b|美元|欧元|英镑|人民币|日元|澳元|加元)",
    re.IGNORECASE,
)

_CURRENCY_CANONICAL = {
    "$": "usd",
    "us$": "usd",
    "usd": "usd",
    "美元": "usd",
    "€": "eur",
    "eur": "eur",
    "欧元": "eur",
    "£": "gbp",
    "gbp": "gbp",
    "英镑": "gbp",
    "¥": "cny",
    "￥": "cny",
    "cny": "cny",
    "rmb": "cny",
    "人民币": "cny",
    "jpy": "jpy",
    "日元": "jpy",
    "a$": "aud",
    "aud": "aud",
    "澳元": "aud",
    "c$": "cad",
    "cad": "cad",
    "加元": "cad",
}


def _normalize_number(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace(",", "")


def _tokens(text: str) -> dict[str, Counter[str]]:
    normalized = unicodedata.normalize("NFKC", text)
    placeholders = _PLACEHOLDER_RE.findall(normalized)
    percent_text = _PLACEHOLDER_RE.sub(" ", normalized)
    percent_count = len(_PERCENT_RE.findall(percent_text))
    return {
        "placeholder": Counter(placeholders),
        "number": Counter(_normalize_number(item) for item in _NUMBER_RE.findall(normalized)),
        "percent": Counter({"percent": percent_count}) if percent_count else Counter(),
        "currency": Counter(_CURRENCY_CANONICAL[item.casefold()] for item in _CURRENCY_RE.findall(normalized)),
    }


class StructureValidator:
    name = "structure"

    def validate(self, source: Subtitle, target_texts: list[str]) -> list[EditIssue]:
        if len(source) != len(target_texts):
            return [
                EditIssue.create(
                    segment_ids=[],
                    category="structure",
                    severity=EditSeverity.ERROR,
                    source=self.name,
                    message=f"Target contains {len(target_texts)} lines; expected {len(source)}.",
                    evidence={"expected": len(source), "actual": len(target_texts)},
                )
            ]
        return [
            EditIssue.create(
                segment_ids=[index],
                category="structure",
                severity=EditSeverity.ERROR,
                source=self.name,
                message="Target subtitle line is empty.",
                evidence={},
            )
            for index, text in enumerate(target_texts, 1)
            if not isinstance(text, str) or not text.strip()
        ]


class GlossaryComplianceValidator:
    name = "glossary"

    def __init__(self, service: GlossaryService, state: GlossaryState):
        self.service = service
        self.state = state

    def validate(self, source: Subtitle, target_texts: list[str]) -> list[EditIssue]:
        checked = self.service.check(self.state, source.texts, target_texts)
        self.state = checked
        issues: list[EditIssue] = []
        for match in checked.matches:
            if match.severity == "info":
                continue
            issues.append(
                EditIssue.create(
                    segment_ids=match.segment_ids,
                    category="glossary",
                    severity=EditSeverity(match.severity),
                    source=self.name,
                    message=match.message,
                    evidence=match.model_dump(mode="json"),
                )
            )
        return issues


class NumberPreservationValidator:
    name = "number-preservation"

    def validate(self, source: Subtitle, target_texts: list[str]) -> list[EditIssue]:
        if len(source) != len(target_texts):
            return []
        issues: list[EditIssue] = []
        for index, (source_text, target_text) in enumerate(zip(source.texts, target_texts), 1):
            source_tokens = _tokens(source_text)
            target_tokens = _tokens(target_text)
            for kind in ("placeholder", "number", "percent", "currency"):
                missing = source_tokens[kind] - target_tokens[kind]
                extra = target_tokens[kind] - source_tokens[kind]
                if missing:
                    issues.append(
                        EditIssue.create(
                            segment_ids=[index],
                            category="number",
                            severity=EditSeverity.ERROR,
                            source=self.name,
                            message=f"Target is missing source {kind} token(s).",
                            evidence={"kind": kind, "missing": dict(missing)},
                        )
                    )
                if extra:
                    issues.append(
                        EditIssue.create(
                            segment_ids=[index],
                            category="number",
                            severity=EditSeverity.WARNING,
                            source=self.name,
                            message=f"Target contains additional {kind} token(s).",
                            evidence={"kind": kind, "extra": dict(extra)},
                        )
                    )
        return issues


class EntityConsistencyValidator:
    name = "entity-consistency"

    def __init__(self, brief: TranslationBrief | None):
        self.brief = brief

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        haystack = unicodedata.normalize("NFKC", text).casefold()
        needle = unicodedata.normalize("NFKC", phrase).casefold()
        if any("\u4e00" <= char <= "\u9fff" for char in needle):
            return needle in haystack
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None

    def validate(self, source: Subtitle, target_texts: list[str]) -> list[EditIssue]:
        if self.brief is None or len(source) != len(target_texts):
            return []
        issues: list[EditIssue] = []
        for character in self.brief.characters:
            for index, (source_text, target_text) in enumerate(zip(source.texts, target_texts), 1):
                if self._contains(source_text, character.source_name) and not self._contains(
                    target_text, character.target_name
                ):
                    issues.append(
                        EditIssue.create(
                            segment_ids=[index],
                            category="entity",
                            severity=EditSeverity.ERROR,
                            source=self.name,
                            message=f"Expected character mapping {character.target_name!r} was not found.",
                            evidence={"source_name": character.source_name, "target_name": character.target_name},
                        )
                    )
        return issues


class ImmutableFieldValidator:
    name = "immutable-fields"

    def validate(self, source: Subtitle, target: Subtitle, *, scope_ids: set[int]) -> list[EditIssue]:
        issues: list[EditIssue] = []
        if len(source) != len(target):
            return [
                EditIssue.create(
                    segment_ids=[],
                    category="immutable",
                    severity=EditSeverity.ERROR,
                    source=self.name,
                    message="Edit changed the subtitle segment count.",
                    evidence={"source_count": len(source), "target_count": len(target)},
                )
            ]
        for index, (source_segment, target_segment) in enumerate(zip(source.segments, target.segments), 1):
            if source_segment.start != target_segment.start or source_segment.end != target_segment.end:
                issues.append(
                    EditIssue.create(
                        segment_ids=[index],
                        category="immutable",
                        severity=EditSeverity.ERROR,
                        source=self.name,
                        message="Edit changed a subtitle timestamp.",
                        evidence={
                            "source": [source_segment.start, source_segment.end],
                            "target": [target_segment.start, target_segment.end],
                        },
                    )
                )
        return issues


class DeterministicValidatorSuite:
    """Return a single EditIssue protocol for all offline checks."""

    def __init__(
        self,
        source: Subtitle,
        *,
        glossary_service: GlossaryService | None = None,
        glossary_state: GlossaryState | None = None,
        brief: TranslationBrief | None = None,
    ):
        self.source = source
        self.structure = StructureValidator()
        self.number = NumberPreservationValidator()
        self.entity = EntityConsistencyValidator(brief)
        self.glossary = (
            GlossaryComplianceValidator(glossary_service, glossary_state)
            if glossary_service is not None and glossary_state is not None
            else None
        )

    @property
    def effective_glossary_state(self) -> GlossaryState | None:
        return self.glossary.state if self.glossary is not None else None

    def validate(self, target_texts: list[str]) -> list[EditIssue]:
        issues = self.structure.validate(self.source, target_texts)
        if len(self.source) != len(target_texts):
            return issues
        issues.extend(self.number.validate(self.source, target_texts))
        issues.extend(self.entity.validate(self.source, target_texts))
        if self.glossary is not None:
            issues.extend(self.glossary.validate(self.source, target_texts))
        return issues


def validation_fingerprint(*, glossary_fingerprint: str, brief: TranslationBrief | None) -> str:
    payload: dict[str, Any] = {
        "validators": ["structure-v1", "glossary-v1", "number-v1", "entity-v1"],
        "glossary_fingerprint": glossary_fingerprint,
        "brief": brief.model_dump(mode="json") if brief is not None else None,
    }
    import hashlib
    import json

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
