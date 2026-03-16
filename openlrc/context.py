#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import re

from pydantic import BaseModel

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
