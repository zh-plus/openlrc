#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import re
from typing import Optional, Union

from pydantic import BaseModel

from openlrc import ModelConfig


class TranslationContext(BaseModel):
    summary: Optional[str] = ''
    scene: Optional[str] = ''
    model: Optional[Union[str, ModelConfig]] = None
    guideline: Optional[str] = None

    def update(self, **args):
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def non_glossary_guideline(self) -> str:
        cleaned_text = re.sub(r'### Glossary.*?### Characters', '### Characters', self.guideline, flags=re.DOTALL)
        return cleaned_text


class TranslateInfo(BaseModel):
    title: Optional[str] = ''
    audio_type: str = 'Movie'
    glossary: Optional[dict] = None
    forced_glossary: bool = False
