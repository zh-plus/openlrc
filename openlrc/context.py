#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openlrc import ModelConfig


class TranslationContext(BaseModel):
    previous_summaries: list[str] | None = None
    summary: str | None = ""
    scene: str | None = ""
    model: str | ModelConfig | None = None
    guideline: str | None = None

    def update(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def non_glossary_guideline(self) -> str:
        if not self.guideline:
            return ""
        cleaned_text = re.sub(r"### Glossary.*?### Characters", "### Characters", self.guideline, flags=re.DOTALL)
        return cleaned_text


class TranslateInfo(BaseModel):
    title: str | None = ""
    audio_type: str = "Movie"
    glossary: dict | None = None
    forced_glossary: bool = False


class CharacterBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    target_name: str

    @field_validator("source_name", "target_name")
    @classmethod
    def _nonempty_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class GlossaryBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    note: str = ""


class TranslationBrief(BaseModel):
    """Structured context prepared by a general model for Hy-MT2."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    characters: list[CharacterBrief] = Field(default_factory=list)
    glossary: list[GlossaryBrief] = Field(default_factory=list)
    tone_style: str = ""

    def merge_user_glossary(self, glossary: dict | None) -> "TranslationBrief":
        if not glossary:
            return self

        merged = {item.source: item for item in self.glossary}
        for source, target in glossary.items():
            merged[str(source)] = GlossaryBrief(source=str(source), target=str(target), note="")
        self.glossary = list(merged.values())
        return self

    def source_semantic_context(self) -> dict:
        """Return target-agnostic semantic context for Timeline generation."""
        return {
            "summary": self.summary,
            "characters": [{"source_name": item.source_name} for item in self.characters],
            "glossary": [{"source": item.source, "note": item.note} for item in self.glossary],
            "tone_style": self.tone_style,
        }

    def characters_text(self) -> str:
        return "\n".join(f"- {item.source_name} -> {item.target_name}" for item in self.characters)

    def glossary_text(self) -> str:
        return "\n".join(f"- {item.source} -> {item.target}" for item in self.glossary)


class TranslationBriefInput(BaseModel):
    """User-authored fixed fields for a full or partial Translation Brief.

    Terminology intentionally remains outside this model: explicit user terms
    use the task glossary service, while an automatically completed Brief may
    still contribute lower-priority inferred terms.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    characters: list[CharacterBrief] | None = None
    tone_style: str | None = None

    @field_validator("summary")
    @classmethod
    def _nonempty_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string when provided")
        return value.strip()

    @field_validator("tone_style")
    @classmethod
    def _normalize_tone_style(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()

    @property
    def provided_fields(self) -> list[str]:
        return [
            field_name
            for field_name in ("summary", "characters", "tone_style")
            if getattr(self, field_name) is not None
        ]

    @property
    def is_complete(self) -> bool:
        return len(self.provided_fields) == 3

    @property
    def origin(self) -> Literal["partial", "manual"]:
        return "manual" if self.is_complete else "partial"

    def fixed_payload(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)

    def fingerprint(self) -> str:
        encoded = json.dumps(self.fixed_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def apply(self, generated: TranslationBrief) -> TranslationBrief:
        """Replace generated sections with every explicitly supplied section."""
        payload = generated.model_dump(mode="json")
        payload.update(self.fixed_payload())
        return TranslationBrief.model_validate(payload)

    def complete(self, *, glossary: dict[str, str] | None = None) -> TranslationBrief:
        """Build a complete Brief without a model when all fixed fields exist."""
        if not self.is_complete:
            raise ValueError("Partial TranslationBriefInput requires automatic completion.")
        terms = [
            GlossaryBrief(source=str(source), target=str(target), note="")
            for source, target in (glossary or {}).items()
        ]
        return TranslationBrief(
            summary=self.summary or "",
            characters=list(self.characters or []),
            glossary=terms,
            tone_style=self.tone_style or "",
        )


def normalize_translation_brief_input(value: TranslationBriefInput | dict | None) -> TranslationBriefInput | None:
    """Canonicalize an empty input to the unchanged fully automatic flow."""
    if value is None:
        return None
    brief_input = TranslationBriefInput.model_validate(value)
    return brief_input if brief_input.provided_fields else None


class ProChunkContext(BaseModel):
    """Bounded dynamic context prepared for one Pro translation chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=1)
    segment_ids: list[int]
    story_so_far: str = ""
    current_scene: str = ""


class ContextTimeline(BaseModel):
    """Ordered dynamic contexts aligned to one shared translation chunk plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    chunk_signature: str
    chunks: list[ProChunkContext] = Field(default_factory=list)

    def context_for(self, chunk_id: int) -> ProChunkContext:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"ContextTimeline has no context for chunk {chunk_id}.")
