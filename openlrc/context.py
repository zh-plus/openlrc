#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import re

from pydantic import BaseModel, Field

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
    source_name: str
    target_name: str
    description: str = ""


class GlossaryBrief(BaseModel):
    source: str
    target: str
    note: str = ""


class ASRAmbiguity(BaseModel):
    segment_ids: list[int]
    source_text: str
    interpretation: str


class TranslationBrief(BaseModel):
    """Structured context prepared by a general model for Hy-MT2."""

    summary: str
    characters: list[CharacterBrief] = Field(default_factory=list)
    glossary: list[GlossaryBrief] = Field(default_factory=list)
    tone_style: str = ""
    target_audience: str = ""
    asr_ambiguities: list[ASRAmbiguity] = Field(default_factory=list)

    def merge_user_glossary(self, glossary: dict | None) -> "TranslationBrief":
        if not glossary:
            return self

        merged = {item.source: item for item in self.glossary}
        for source, target in glossary.items():
            merged[str(source)] = GlossaryBrief(source=str(source), target=str(target), note="user-provided")
        self.glossary = list(merged.values())
        return self

    def characters_text(self) -> str:
        return "\n".join(f"- {item.source_name} -> {item.target_name}: {item.description}" for item in self.characters)

    def glossary_text(self) -> str:
        return "\n".join(f"- {item.source} -> {item.target}" for item in self.glossary)

    def ambiguities_text(self) -> str:
        return "\n".join(
            f"- segments {','.join(map(str, item.segment_ids))}: {item.source_text} => {item.interpretation}"
            for item in self.asr_ambiguities
        )
