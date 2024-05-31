#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import re
from typing import Optional

from pydantic import BaseModel


class TranslationContext(BaseModel):
    summary: Optional[str] = ''
    scene: Optional[str] = ''
    model: Optional[str] = None
    guideline: Optional[str] = None

    def update(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def non_glossary_guideline(self) -> str:
        cleaned_text = re.sub(r'Glossary:\n(.*?\n)*?\nCharacters:', 'Characters:', self.guideline, flags=re.DOTALL)
        return cleaned_text


class TranslateInfo(BaseModel):
    title: Optional[str] = ''
    audio_type: str = 'Movie'
    glossary: Optional[dict] = None
