#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Context preparation and high-risk review helpers for Hy-MT2."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from json_repair import repair_json
from langcodes import Language as LangcodeLanguage
from lingua import Language as LinguaLanguage
from lingua import LanguageDetectorBuilder
from pydantic import BaseModel, ConfigDict, Field

from openlrc.chatbot import ChatBot
from openlrc.checkpoint import save_json_checkpoint
from openlrc.chunking import TranslationChunkPlan
from openlrc.context import (
    ContextTimeline,
    ProChunkContext,
    TranslationBrief,
    TranslationBriefInput,
    normalize_translation_brief_input,
)
from openlrc.defaults import supported_languages_lingua
from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.utils import get_text_token_number


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class _SourceLanguageValidator:
    """Reject clear language-contract violations without guessing on short text."""

    MIN_CHARACTERS = 24
    MIN_CONFIDENCE = 0.75
    MIN_EXPECTED_MARGIN = 0.20

    def __init__(self, src_lang: str):
        self.src_lang = src_lang
        self.expected_code = LangcodeLanguage.get(src_lang).language
        languages = [getattr(LinguaLanguage, name) for name in supported_languages_lingua]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        self.supported_codes = {self._language_code(language) for language in languages}

    @staticmethod
    def _character_count(text: str) -> int:
        return sum(character.isalpha() for character in text)

    @staticmethod
    def _language_code(language: LinguaLanguage) -> str | None:
        return language.iso_code_639_1.name.lower()

    def validate_text(self, field_name: str, text: str) -> bool:
        if self.expected_code not in self.supported_codes or self._character_count(text) < self.MIN_CHARACTERS:
            return True

        confidence_values = self.detector.compute_language_confidence_values(text)
        if not confidence_values:
            return True
        top = confidence_values[0]
        top_code = self._language_code(top.language)
        expected_confidence = next(
            (item.value for item in confidence_values if self._language_code(item.language) == self.expected_code), 0.0
        )
        invalid = (
            top_code != self.expected_code
            and top.value >= self.MIN_CONFIDENCE
            and top.value - expected_confidence >= self.MIN_EXPECTED_MARGIN
        )
        if invalid:
            logger.warning(
                f"{field_name} is clearly {top_code}, not the required source language {self.expected_code}."
            )
        return not invalid

    def validate_brief(self, brief: TranslationBrief) -> bool:
        if not brief.summary.strip():
            return False
        for character in brief.characters:
            if not character.source_name.strip() or not character.target_name.strip():
                return False
        for term in brief.glossary:
            if not term.source.strip() or not term.target.strip():
                return False

        semantic_fields = [("summary", brief.summary)]
        semantic_fields.extend(
            (f"glossary[{index}].note", item.note) for index, item in enumerate(brief.glossary) if item.note
        )
        semantic_fields.append(("tone_style", brief.tone_style))

        nonempty_fields = [(name, text) for name, text in semantic_fields if text.strip()]
        if not all(self.validate_text(name, text) for name, text in nonempty_fields):
            return False
        aggregate = "\n".join(text for _, text in nonempty_fields)
        return self.validate_text("translation_brief.source_semantics", aggregate)


def source_fingerprint(
    texts: list[str],
    *,
    src_lang: str,
    target_lang: str,
    glossary: dict | None,
    mode: str,
    context_model: str,
    translation_model: str,
    translation_brief: TranslationBriefInput | dict | None = None,
    normalize_mode: bool = True,
) -> str:
    if normalize_mode:
        mode = {"context": "normal", "context-plus": "normal-plus"}.get(mode, mode)
    payload = {
        "texts": texts,
        "src_lang": src_lang,
        "target_lang": target_lang,
        "glossary": glossary or {},
        "mode": mode,
        "context_model": context_model,
        "translation_model": translation_model,
    }
    brief_input = normalize_translation_brief_input(translation_brief)
    if brief_input is not None:
        payload["translation_brief_input"] = brief_input.fixed_payload()
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

    PROMPT_VERSION = 3
    OUTPUT_RESERVE = 2048

    def __init__(self, *, chatbot: ChatBot | None, src_lang: str, target_lang: str):
        self.chatbot = chatbot
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.language_validator = _SourceLanguageValidator(src_lang)

    def _schema_instruction(self) -> str:
        return f"""Return one JSON object with exactly these fields:
{{
  "summary": "source-language document summary",
  "characters": [{{"source_name": "source name", "target_name": "target-language name"}}],
  "glossary": [{{"source": "source term", "target": "target-language translation", "note": "source-language note"}}],
  "tone_style": "source-language tone, register, and style guidance"
}}
The source language is {self.src_lang}. Write summary, glossary notes, and tone_style in the source language.
The target language is {self.target_lang}. Only character target_name and glossary target are
target-language translation mappings. Preserve source names and source terms in their source form.
Every character and glossary item must contain non-empty source and target values. Omit uncertain items.
Do not translate the subtitle lines. Do not add Markdown, unlisted fields, or any text outside the JSON object."""

    @staticmethod
    def _parse(content: str) -> TranslationBrief:
        repaired = repair_json(content)
        payload = json.loads(str(repaired))
        if not isinstance(payload, dict):
            raise ValueError("Translation brief must be a JSON object.")
        return TranslationBrief(**payload)

    def _check(self, _user_input: str, content: str) -> bool:
        try:
            brief = self._parse(content)
            return self.language_validator.validate_brief(brief)
        except (ValueError, TypeError, json.JSONDecodeError):
            return False

    def _call(self, user_prompt: str) -> TranslationBrief:
        if self.chatbot is None:
            raise ChatBotException("A context model is required to complete a Partial Translation Brief.")
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
            brief = self._parse(content)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ChatBotException(f"Failed to parse translation brief: {exc}") from exc
        if not self.language_validator.validate_brief(brief):
            raise ChatBotException("Translation brief violates the source-language field contract.")
        return brief

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

    def build(
        self,
        texts: list[str],
        *,
        title: str = "",
        glossary: dict | None = None,
        translation_brief: TranslationBriefInput | dict | None = None,
    ) -> TranslationBrief:
        fixed = normalize_translation_brief_input(translation_brief)
        if fixed is not None and fixed.is_complete:
            brief = fixed.complete(glossary=glossary)
            if not self.language_validator.validate_brief(brief):
                raise ChatBotException("Manual translation brief violates the source-language field contract.")
            return brief

        fixed_instruction = ""
        if fixed is not None:
            fixed_instruction = (
                "\nUser-fixed Brief fields follow. Treat every supplied field as authoritative, copy it "
                "unchanged into the full JSON response, and infer only missing fields.\n"
                f"Fixed fields: {json.dumps(fixed.fixed_payload(), ensure_ascii=False)}\n"
            )
        partials: list[TranslationBrief] = []
        chunks = self._split_texts(texts)
        for chunk_index, chunk in enumerate(chunks, 1):
            numbered = "\n".join(f"[{line_id}] {text}" for line_id, text in chunk)
            partials.append(
                self._call(
                    f"Title: {title}\n"
                    f"Source language: {self.src_lang}\nTarget language: {self.target_lang}\n"
                    f"User glossary: {json.dumps(glossary or {}, ensure_ascii=False)}\n"
                    f"{fixed_instruction}"
                    f"Subtitle section {chunk_index}/{len(chunks)}:\n{numbered}"
                )
            )
            if fixed is not None:
                partials[-1] = fixed.apply(partials[-1])

        if not partials:
            raise ChatBotException("Cannot build a translation brief from empty subtitles.")

        while len(partials) > 1:
            merged: list[TranslationBrief] = []
            for index in range(0, len(partials), 2):
                pair = partials[index : index + 2]
                if len(pair) == 1:
                    merged.append(pair[0])
                    continue
                candidate = self._call(
                    "Merge these partial briefs into one coherent document-level brief. "
                    "Union characters and glossary entries, preserve chronological meaning, remove duplicates, "
                    "and keep the same source-semantic and bilingual-mapping language contract.\n"
                    + fixed_instruction
                    + json.dumps([_model_dump(item) for item in pair], ensure_ascii=False)
                )
                merged.append(fixed.apply(candidate) if fixed is not None else candidate)
            partials = merged

        brief = partials[0].merge_user_glossary(glossary)
        if fixed is not None:
            brief = fixed.apply(brief)
        if not self.language_validator.validate_brief(brief):
            raise ChatBotException("Merged translation brief violates the source-language field contract.")
        return brief


class _TimelineChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=1)
    segment_ids: list[int]
    story_so_far: str
    current_scene: str


class ContextTimelineAgent:
    """Generate bounded sequential context for every planned Pro chunk."""

    SCHEMA_VERSION = 1
    PROMPT_VERSION = 3
    STORY_TOKEN_LIMIT = 256
    SCENE_TOKEN_LIMIT = 160
    TOTAL_TOKEN_LIMIT = 416

    def __init__(self, *, chatbot: ChatBot, src_lang: str):
        self.chatbot = chatbot
        self.src_lang = src_lang
        self.language_validator = _SourceLanguageValidator(src_lang)

    @classmethod
    def _parse(cls, content: str) -> _TimelineChunkResponse:
        repaired = repair_json(content)
        payload = json.loads(str(repaired))
        if not isinstance(payload, dict):
            raise ValueError("Timeline chunk must be a JSON object.")
        result = _TimelineChunkResponse(**payload)
        story_tokens = get_text_token_number(result.story_so_far)
        scene_tokens = get_text_token_number(result.current_scene)
        if story_tokens > cls.STORY_TOKEN_LIMIT:
            raise ValueError(f"story_so_far exceeds {cls.STORY_TOKEN_LIMIT} tokens.")
        if scene_tokens > cls.SCENE_TOKEN_LIMIT:
            raise ValueError(f"current_scene exceeds {cls.SCENE_TOKEN_LIMIT} tokens.")
        if story_tokens + scene_tokens > cls.TOTAL_TOKEN_LIMIT:
            raise ValueError(f"Timeline chunk exceeds {cls.TOTAL_TOKEN_LIMIT} total tokens.")
        return result

    @classmethod
    def _check(cls, _user_input: str, content: str) -> bool:
        try:
            cls._parse(content)
            return True
        except (ValueError, TypeError, json.JSONDecodeError):
            return False

    @classmethod
    def _checker(
        cls,
        expected_chunk_id: int,
        expected_segment_ids: list[int],
        language_validator: _SourceLanguageValidator | None = None,
    ):
        def check(_user_input: str, content: str) -> bool:
            try:
                result = cls._parse(content)
            except (ValueError, TypeError, json.JSONDecodeError):
                return False
            if result.chunk_id != expected_chunk_id or result.segment_ids != expected_segment_ids:
                return False
            if language_validator is None:
                return True
            return language_validator.validate_text("story_so_far", result.story_so_far) and (
                language_validator.validate_text("current_scene", result.current_scene)
            )

        return check

    @staticmethod
    def _neighboring_source(texts: list[str], plan: TranslationChunkPlan, radius: int = 2) -> list[dict]:
        first = plan.segment_ids[0] - 1
        last = plan.segment_ids[-1] - 1
        start = max(0, first - radius)
        end = min(len(texts), last + radius + 1)
        return [{"id": index + 1, "text": texts[index]} for index in range(start, end) if index < first or index > last]

    def _generate_chunk(
        self, plan: TranslationChunkPlan, *, texts: list[str], brief: TranslationBrief, previous: ProChunkContext | None
    ) -> ProChunkContext:
        prompt = {
            "task": (
                "Update bounded story and current-scene context after reading this subtitle chunk. "
                "Use only evidence in the source subtitles. Do not translate individual lines, invent events, "
                "or create glossary entries. If hard_scene_break_before is true, reset current_scene while "
                "preserving a compressed story_so_far. Write story_so_far and current_scene in source_language. "
                "Return exactly the requested JSON fields."
            ),
            "source_language": self.src_lang,
            "source_semantic_brief": brief.source_semantic_context(),
            "previous_story_so_far": previous.story_so_far if previous else "",
            "previous_current_scene": (
                previous.current_scene if previous is not None and not plan.hard_scene_break_before else ""
            ),
            "chunk_id": plan.chunk_id,
            "segment_ids": plan.segment_ids,
            "hard_scene_break_before": plan.hard_scene_break_before,
            "neighboring_source_subtitles": self._neighboring_source(texts, plan),
            "source_subtitles": [{"id": line_id, "text": text} for line_id, text in zip(plan.segment_ids, plan.texts)],
            "limits": {
                "story_so_far_tokens": self.STORY_TOKEN_LIMIT,
                "current_scene_tokens": self.SCENE_TOKEN_LIMIT,
                "combined_tokens": self.TOTAL_TOKEN_LIMIT,
            },
            "output_schema": {
                "chunk_id": plan.chunk_id,
                "segment_ids": plan.segment_ids,
                "story_so_far": "string",
                "current_scene": "string",
            },
        }
        messages = [
            {
                "role": "system",
                "content": "You prepare compact sequential context for a specialist subtitle translator. Return JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        response = self.chatbot.message(
            messages, output_checker=self._checker(plan.chunk_id, plan.segment_ids, self.language_validator)
        )[0]
        content = self.chatbot.get_content(response)
        if not content:
            raise ChatBotException(f"Context model returned an empty Timeline chunk for chunk {plan.chunk_id}.")
        try:
            result = self._parse(content)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ChatBotException(f"Failed to parse Timeline chunk {plan.chunk_id}: {exc}") from exc
        if result.chunk_id != plan.chunk_id or result.segment_ids != plan.segment_ids:
            raise ChatBotException(f"Timeline chunk {plan.chunk_id} returned mismatched chunk or segment IDs.")
        if not self.language_validator.validate_text("story_so_far", result.story_so_far) or not (
            self.language_validator.validate_text("current_scene", result.current_scene)
        ):
            raise ChatBotException(f"Timeline chunk {plan.chunk_id} violates the source-language contract.")
        return ProChunkContext(
            chunk_id=result.chunk_id,
            segment_ids=result.segment_ids,
            story_so_far=result.story_so_far,
            current_scene=result.current_scene,
        )

    def build(
        self,
        plans: list[TranslationChunkPlan],
        *,
        texts: list[str],
        brief: TranslationBrief,
        chunk_signature: str,
        checkpoint_path: Path,
        checkpoint: dict,
    ) -> ContextTimeline:
        """Build or resume a Timeline, saving each completed chunk atomically."""
        restored: list[ProChunkContext] = []
        timeline_data = checkpoint.get("context_timeline")
        if isinstance(timeline_data, dict) and timeline_data.get("chunk_signature") == chunk_signature:
            try:
                saved = ContextTimeline(**timeline_data)
                restored = saved.chunks
            except (ValueError, TypeError):
                restored = []

        contexts: list[ProChunkContext] = []
        can_restore = True
        for index, plan in enumerate(plans):
            if can_restore and index < len(restored):
                saved_context = restored[index]
                if saved_context.chunk_id == plan.chunk_id and saved_context.segment_ids == plan.segment_ids:
                    contexts.append(saved_context)
                    continue
                can_restore = False
            else:
                can_restore = False

            context = self._generate_chunk(plan, texts=texts, brief=brief, previous=contexts[-1] if contexts else None)
            contexts.append(context)
            timeline = ContextTimeline(
                schema_version=self.SCHEMA_VERSION, chunk_signature=chunk_signature, chunks=contexts
            )
            checkpoint.update(
                context_timeline=_model_dump(timeline),
                timeline_completed_chunks=[item.chunk_id for item in contexts],
                timeline_prompt_version=self.PROMPT_VERSION,
                pipeline_stage="context_plan",
            )
            save_checkpoint(checkpoint_path, checkpoint)

        return ContextTimeline(schema_version=self.SCHEMA_VERSION, chunk_signature=chunk_signature, chunks=contexts)


class RiskReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    risk: Literal["low", "medium", "high"]
    issues: list[str] = Field(default_factory=list)
    revised_translation: str | None = None


class RiskReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RiskReviewItem]


class ReviewRevisionValidator:
    """Reject empty, multiline, or protocol-shaped review replacements."""

    MAX_CHARACTERS = 1000
    _TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
    _ANCHOR_RE = re.compile(r"^#(?:<\d+>|\d+)\s*$")
    _PROTOCOL_RE = re.compile(
        r"(?:\[\s*(?:background information|source text|translation guidance)\s*\]"
        r"|<--\s*END-[^>]*-->"
        r"|\b(?:story_so_far|current_scene|revised_translation|review_results)\b\s*[:=])",
        re.IGNORECASE,
    )

    @classmethod
    def normalize(cls, value: str | None) -> str:
        if not isinstance(value, str):
            raise ValueError("High-risk revision must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("High-risk revision is blank.")
        if "\n" in normalized or "\r" in normalized or "\t" in normalized:
            raise ValueError("High-risk revision must contain exactly one subtitle line.")
        if len(normalized) > cls.MAX_CHARACTERS:
            raise ValueError(f"High-risk revision exceeds {cls.MAX_CHARACTERS} characters.")
        if "```" in normalized or "~~~" in normalized:
            raise ValueError("High-risk revision contains a Markdown code fence.")
        if cls._TAG_RE.search(normalized) or cls._ANCHOR_RE.fullmatch(normalized):
            raise ValueError("High-risk revision contains a subtitle protocol tag or anchor.")
        if cls._PROTOCOL_RE.search(normalized):
            raise ValueError("High-risk revision contains internal context protocol text.")
        if normalized[0] in "[{":
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(payload, (dict, list)):
                    raise ValueError("High-risk revision contains a JSON object or array.")
        return normalized


class HyMT2RiskReviewAgent:
    """Scan every translated chunk and only revise genuinely high-risk lines."""

    REVIEW_PROTOCOL_VERSION = 2

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
                cls._validate_result(result, expected_ids)
            except (ValueError, TypeError, json.JSONDecodeError):
                return False
            return True

        return check

    @staticmethod
    def _validate_result(result: RiskReviewResponse, expected_ids: list[int]) -> RiskReviewResponse:
        if [item.id for item in result.items] != expected_ids:
            raise ValueError("Risk review IDs do not match the translated chunk.")
        for item in result.items:
            if item.risk == "high":
                item.revised_translation = ReviewRevisionValidator.normalize(item.revised_translation)
            elif item.revised_translation is not None:
                raise ValueError("Only high-risk review items may contain a revised translation.")
        return result

    @classmethod
    def validate_saved_items(cls, items: object, expected_ids: list[int]) -> list[dict]:
        if not isinstance(items, list):
            raise ValueError("Saved risk review items must be a list.")
        result = cls._validate_result(RiskReviewResponse(items=items), expected_ids)
        return [_model_dump(item) for item in result.items]

    def review(
        self,
        chunk: list[tuple[int, str]],
        translations: dict[int, str],
        *,
        brief: TranslationBrief,
        neighboring_context: str = "",
        fallback_metadata: dict | None = None,
        chunk_context: ProChunkContext | None = None,
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
            "context_timeline_chunk": _model_dump(chunk_context) if chunk_context is not None else None,
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
        try:
            return self._validate_result(result, expected_ids)
        except ValueError as exc:
            raise ChatBotException(f"Invalid risk review: {exc}") from exc

    def review_patches(
        self,
        chunk: list[tuple[int, str]],
        translations: dict[int, str],
        *,
        brief: TranslationBrief,
        neighboring_context: str = "",
        fallback_metadata: dict | None = None,
        chunk_context: ProChunkContext | None = None,
    ):
        """Convert the compatible v2 response into structured edit issues and patches."""
        from openlrc.editing import EditAction, EditIssue, EditPatch, EditSeverity

        response = self.review(
            chunk,
            translations,
            brief=brief,
            neighboring_context=neighboring_context,
            fallback_metadata=fallback_metadata,
            chunk_context=chunk_context,
        )
        issues = []
        patches = []
        severity_map = {"low": EditSeverity.INFO, "medium": EditSeverity.WARNING, "high": EditSeverity.ERROR}
        for item in response.items:
            if not item.issues and item.risk != "high":
                continue
            message = "; ".join(item.issues) or "High-risk semantic translation issue"
            issue = EditIssue.create(
                segment_ids=[item.id],
                category="semantic",
                severity=severity_map[item.risk],
                source="semantic-review",
                message=message,
                evidence={"risk": item.risk},
            )
            issues.append(issue)
            if item.risk == "high":
                assert item.revised_translation is not None
                patches.append(
                    EditPatch.create(
                        issue_ids=[issue.issue_id],
                        segment_id=item.id,
                        before=translations[item.id],
                        after=item.revised_translation,
                        reason=message,
                        action=EditAction.REVIEW,
                    )
                )
        metadata = {
            "review_protocol_version": self.REVIEW_PROTOCOL_VERSION,
            "model": getattr(self.chatbot, "model_name", None),
        }
        return issues, patches, metadata
