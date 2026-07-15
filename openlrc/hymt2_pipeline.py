#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Context preparation and high-risk review helpers for Hy-MT2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from json_repair import repair_json
from pydantic import BaseModel, Field

from openlrc.chatbot import ChatBot
from openlrc.checkpoint import save_json_checkpoint
from openlrc.context import TranslationBrief
from openlrc.exceptions import ChatBotException
from openlrc.utils import get_text_token_number


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def source_fingerprint(
    texts: list[str],
    *,
    src_lang: str,
    target_lang: str,
    glossary: dict | None,
    mode: str,
    context_model: str,
    translation_model: str,
) -> str:
    payload = {
        "texts": texts,
        "src_lang": src_lang,
        "target_lang": target_lang,
        "glossary": glossary or {},
        "mode": mode,
        "context_model": context_model,
        "translation_model": translation_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_checkpoint(path: Path, data: dict) -> None:
    save_json_checkpoint(path, data)


class TranslationBriefAgent:
    """Build a strict, reusable translation brief using a general-purpose model."""

    OUTPUT_RESERVE = 2048

    def __init__(self, *, chatbot: ChatBot, src_lang: str, target_lang: str):
        self.chatbot = chatbot
        self.src_lang = src_lang
        self.target_lang = target_lang

    @staticmethod
    def _schema_instruction() -> str:
        return """Return one JSON object with exactly these fields:
{
  "summary": "concise document-level summary",
  "characters": [{"source_name": "", "target_name": "", "description": ""}],
  "glossary": [{"source": "", "target": "", "note": ""}],
  "tone_style": "subtitle tone, register, and style guidance",
  "target_audience": "intended audience guidance",
  "asr_ambiguities": [{"segment_ids": [1], "source_text": "", "interpretation": ""}]
}
Do not translate the subtitle lines. Do not add Markdown or any text outside the JSON object.
Only report ASR ambiguities that can be resolved from context; otherwise use an empty list."""

    @staticmethod
    def _parse(content: str) -> TranslationBrief:
        repaired = repair_json(content)
        payload = json.loads(str(repaired))
        if not isinstance(payload, dict):
            raise ValueError("Translation brief must be a JSON object.")
        return TranslationBrief(**payload)

    @classmethod
    def _check(cls, _user_input: str, content: str) -> bool:
        try:
            cls._parse(content)
            return True
        except (ValueError, TypeError, json.JSONDecodeError):
            return False

    def _call(self, user_prompt: str) -> TranslationBrief:
        messages = [
            {
                "role": "system",
                "content": (
                    "You prepare context for a specialist machine-translation model. "
                    "Analyze the source subtitles, but do not translate them."
                ),
            },
            {"role": "user", "content": f"{self._schema_instruction()}\n\n{user_prompt}"},
        ]
        response = self.chatbot.message(messages, output_checker=self._check)[0]
        content = self.chatbot.get_content(response)
        if not content:
            raise ChatBotException("Context model returned an empty translation brief.")
        try:
            return self._parse(content)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ChatBotException(f"Failed to parse translation brief: {exc}") from exc

    def _split_texts(self, texts: list[str]) -> list[list[tuple[int, str]]]:
        model_config = getattr(self.chatbot, "model_config", None)
        context_window = getattr(model_config, "context_window", None) or 32768
        budget = max(512, int(context_window * 0.70) - self.OUTPUT_RESERVE)
        chunks: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        used = 0
        for line_id, text in enumerate(texts, 1):
            tokens = get_text_token_number(text) + 4
            if current and used + tokens > budget:
                chunks.append(current)
                current = []
                used = 0
            current.append((line_id, text))
            used += tokens
        if current:
            chunks.append(current)
        return chunks

    def build(self, texts: list[str], *, title: str = "", glossary: dict | None = None) -> TranslationBrief:
        partials: list[TranslationBrief] = []
        chunks = self._split_texts(texts)
        for chunk_index, chunk in enumerate(chunks, 1):
            numbered = "\n".join(f"[{line_id}] {text}" for line_id, text in chunk)
            partials.append(
                self._call(
                    f"Title: {title}\n"
                    f"Source language: {self.src_lang}\nTarget language: {self.target_lang}\n"
                    f"User glossary: {json.dumps(glossary or {}, ensure_ascii=False)}\n"
                    f"Subtitle section {chunk_index}/{len(chunks)}:\n{numbered}"
                )
            )

        if not partials:
            raise ChatBotException("Cannot build a translation brief from empty subtitles.")

        while len(partials) > 1:
            merged: list[TranslationBrief] = []
            for index in range(0, len(partials), 2):
                pair = partials[index : index + 2]
                if len(pair) == 1:
                    merged.append(pair[0])
                    continue
                merged.append(
                    self._call(
                        "Merge these partial briefs into one coherent document-level brief. "
                        "Union characters and glossary entries, preserve chronological meaning, and remove duplicates.\n"
                        + json.dumps([_model_dump(item) for item in pair], ensure_ascii=False)
                    )
                )
            partials = merged

        return partials[0].merge_user_glossary(glossary)


class RiskReviewItem(BaseModel):
    id: int
    risk: Literal["low", "medium", "high"]
    issues: list[str] = Field(default_factory=list)
    revised_translation: str | None = None


class RiskReviewResponse(BaseModel):
    items: list[RiskReviewItem]


class HyMT2RiskReviewAgent:
    """Scan every translated chunk and only revise genuinely high-risk lines."""

    def __init__(self, *, chatbot: ChatBot, src_lang: str, target_lang: str):
        self.chatbot = chatbot
        self.src_lang = src_lang
        self.target_lang = target_lang

    @staticmethod
    def _parse(content: str) -> RiskReviewResponse:
        repaired = repair_json(content)
        payload = json.loads(str(repaired))
        if not isinstance(payload, dict):
            raise ValueError("Risk review must be a JSON object.")
        return RiskReviewResponse(**payload)

    @classmethod
    def _checker(cls, expected_ids: list[int]):
        def check(_user_input: str, content: str) -> bool:
            try:
                result = cls._parse(content)
            except (ValueError, TypeError, json.JSONDecodeError):
                return False
            ids = [item.id for item in result.items]
            if ids != expected_ids:
                return False
            return all(item.risk != "high" or bool(item.revised_translation) for item in result.items)

        return check

    def review(
        self,
        chunk: list[tuple[int, str]],
        translations: dict[int, str],
        *,
        brief: TranslationBrief,
        neighboring_context: str = "",
        fallback_metadata: dict | None = None,
    ) -> RiskReviewResponse:
        expected_ids = [line_id for line_id, _ in chunk]
        pairs = [{"id": line_id, "source": source, "translation": translations[line_id]} for line_id, source in chunk]
        prompt = {
            "task": (
                "Review every item. High risk is limited to meaning errors, omissions/hallucinations, "
                "context disambiguation errors, name/terminology conflicts, or clearly unreadable output. "
                "Pure stylistic preference is not high risk. Only high-risk items may receive a revision."
            ),
            "source_language": self.src_lang,
            "target_language": self.target_lang,
            "translation_brief": _model_dump(brief),
            "neighboring_context": neighboring_context,
            "fallback_metadata": fallback_metadata or {},
            "items": pairs,
            "output_schema": {
                "items": [
                    {
                        "id": "same integer id",
                        "risk": "low|medium|high",
                        "issues": ["short reason"],
                        "revised_translation": "required only for high; otherwise null",
                    }
                ]
            },
        }
        messages = [
            {
                "role": "system",
                "content": "You are a conservative subtitle translation risk reviewer. Return JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        response = self.chatbot.message(messages, output_checker=self._checker(expected_ids))[0]
        content = self.chatbot.get_content(response)
        if not content:
            raise ChatBotException("Context model returned an empty risk review.")
        try:
            result = self._parse(content)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ChatBotException(f"Failed to parse risk review: {exc}") from exc
        if [item.id for item in result.items] != expected_ids:
            raise ChatBotException("Risk review IDs do not match the translated chunk.")
        return result
