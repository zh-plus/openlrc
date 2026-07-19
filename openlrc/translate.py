#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import json
import os
import re
import unicodedata
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from itertools import zip_longest
from pathlib import Path

import requests

from openlrc.agents import ChunkedTranslatorAgent, ContextReviewerAgent
from openlrc.chatbot import ChatBot
from openlrc.checkpoint import save_json_checkpoint
from openlrc.chunking import CHUNK_PLANNER_VERSION, TranslationChunkPlan, chunk_plan_signature, plan_translation_chunks
from openlrc.context import ContextTimeline, TranslateInfo, TranslationBrief, TranslationContext
from openlrc.exceptions import ChatBotException, LengthExceedException
from openlrc.glossary import GlossaryOrigin, GlossaryService, ResolvedGlossaryEntry
from openlrc.logger import logger
from openlrc.prompter import (
    LeanContextReviewPrompter,
    LeanTranslatePrompter,
    create_atomic_translate_prompter,
    create_lean_translate_prompter,
)
from openlrc.utils import get_text_token_number


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: str | list[str], src_lang: str, target_lang: str, info: TranslateInfo) -> list[str]:
        pass


class BaseLLMTranslator(Translator):
    """Base class for LLM-based translators.

    Provides shared infrastructure used by all LLM translator variants:
    chunking (by line count, token budget, and scene boundaries), atomic
    (per-line) translation fallback, checkpoint save/load for resumption,
    and compare-list generation for debugging.

    Subclasses implement :meth:`translate` and their own chunk-level
    translation strategy.
    """

    CHUNK_SIZE = 30
    MAX_CHUNK_TOKENS = 1000  # Token budget per chunk (text content only, excludes prompt overhead)
    SCENE_THRESHOLD = 30.0  # Seconds of silence that indicates a scene boundary

    OUTPUT_RATIO: float = 1.0  # Expected output tokens per input token (subclass overrides)

    def __init__(
        self,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        cr_chatbot: ChatBot | None = None,
        chunk_size: int = CHUNK_SIZE,
        timestamps: list[tuple[float, float | None]] | None = None,
        chunked_guideline: bool = False,
        prompt_profile: str = "default",
    ):
        self.chatbot = chatbot
        self.retry_chatbot = retry_chatbot
        self.cr_chatbot = cr_chatbot
        self.chunk_size = chunk_size
        self.timestamps = timestamps
        self.chunked_guideline = chunked_guideline
        self.prompt_profile = prompt_profile
        self.api_fee = 0.0
        self.metrics = {
            "mode": prompt_profile,
            "retries": 0,
            "splits": 0,
            "max_split_depth": 0,
            "atomic_ids": [],
            "validator_issues": [],
            "glossary_removed_retry_chunks": [],
        }

    @staticmethod
    def make_chunks(texts: list[str], chunk_size: int = 30) -> list[list[tuple[int, str]]]:
        """
        Split the text into chunks of specified size for efficient processing.

        Args:
            texts (List[str]): List of texts to be chunked.
            chunk_size (int): Maximum size of each chunk.

        Returns:
            List[List[Tuple[int, str]]]: List of chunks, each chunk is a list of (line_number, text) tuples.
        """
        chunks = []
        start = 1
        for i in range(0, len(texts), chunk_size):
            chunk = [(start + j, text) for j, text in enumerate(texts[i : i + chunk_size])]
            start += len(chunk)
            chunks.append(chunk)

        # Merge the last chunk if it's too small
        if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def make_chunks_by_tokens(self, texts: list[str]) -> list[list[tuple[int, str]]]:
        """Split texts into chunks using token budget, scene boundaries, and line-count limits.

        Splitting strategy (applied in priority order):
        1. **Scene boundary (hard split):** A time gap > ``SCENE_THRESHOLD`` between
           adjacent lines forces a chunk break regardless of token budget.
           Only active when ``self.timestamps`` is set.
        2. **Token budget:** Accumulated text tokens must not exceed ``MAX_CHUNK_TOKENS``.
           When the budget is exceeded, the chunk is split at the largest time gap
           within the current chunk (if timestamps are available), otherwise at the
           current position.
        3. **Line-count cap:** A chunk never exceeds ``chunk_size`` lines.

        A small trailing chunk may be merged only when no scene boundary separates it
        and the combined chunk still satisfies both line and token caps.

        Args:
            texts: List of subtitle texts.

        Returns:
            List of chunks, each chunk a list of ``(line_number, text)`` tuples.
        """
        return [plan.pairs() for plan in self.plan_chunks(texts)]

    def plan_chunks(self, texts: list[str]) -> list[TranslationChunkPlan]:
        """Return the shared translation plan used by context, translation, and review."""
        return plan_translation_chunks(
            texts,
            timestamps=self.timestamps,
            chunk_size=self.chunk_size,
            token_budget=self.MAX_CHUNK_TOKENS,
            scene_threshold=self.SCENE_THRESHOLD,
        )

    def _estimate_output_tokens(self, chunk: list[tuple[int, str]]) -> int:
        """Estimate how many output tokens this chunk is likely to require."""
        content_tokens = sum(get_text_token_number(text) for _, text in chunk)
        return max(1024, int(content_tokens * self.OUTPUT_RATIO))

    def atomic_translate(
        self,
        chatbot: ChatBot,
        texts: list[str],
        src_lang: str,
        target_lang: str,
        *,
        contexts: list[str] | None = None,
        guideline: str = "",
    ) -> list[str]:
        """
        Perform atomic translation for each text individually.

        This method is used as a fallback when chunk translation fails. It translates
        each text separately, which can be slower but more reliable for problematic texts.

        Args:
            chatbot (ChatBot): ChatBot instance to use for translation.
            texts (List[str]): List of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.

        Returns:
            List[str]: List of translated texts.

        Raises:
            ChatBotException: If the number of translated texts doesn't match the input.
        """
        from openlrc.agents import ChunkedTranslatorAgent as _CTA

        prompter = create_atomic_translate_prompter(src_lang, target_lang, self.prompt_profile)
        contexts = contexts or [""] * len(texts)
        message_lists = [
            [{"role": "user", "content": prompter.user(text, context=context, guideline=guideline)}]
            for text, context in zip(texts, contexts)
        ]

        responses = chatbot.message(
            message_lists, output_checker=prompter.check_format, temperature=chatbot.agent_temperature(_CTA.TEMPERATURE)
        )
        self.api_fee += sum(chatbot.api_fees[-(len(texts)) :])
        translated = list(map(chatbot.get_content, responses))

        if len(translated) != len(texts):
            raise ChatBotException(
                f"Atomic translation failed: expected {len(texts)} translations, got {len(translated)}"
            )

        return translated

    def _save_checkpoint(self, compare_path: Path, compare_list: list[dict], context: dict) -> None:
        """Save translation checkpoint for potential resumption.

        Args:
            compare_path: Path to save the checkpoint JSON file.
            compare_list: List of per-line comparison records.
            context: Subclass-specific context data (e.g. summaries, guideline,
                sliding window).  Stored alongside ``compare_list`` in the JSON.
        """
        data = {"compare": compare_list, **context}
        save_json_checkpoint(compare_path, data)

    def _load_checkpoint(self, compare_path: Path) -> tuple[list[str], list[dict], int, dict]:
        """Load translation checkpoint for resumption.

        Args:
            compare_path: Path to the checkpoint JSON file.

        Returns:
            A 4-tuple of ``(translations, compare_list, start_chunk, context)``.
            *translations* are the already-translated texts extracted from
            *compare_list*.  *start_chunk* is the chunk number to resume from.
            *context* is the remaining data dict (subclass-specific).
            If the file does not exist, returns empty defaults.
        """
        if not compare_path.exists():
            return [], [], 0, {}

        logger.info(f"Resuming translation from {compare_path}")
        with open(compare_path, encoding="utf-8") as f:
            data = json.load(f)

        compare_list = data.pop("compare")
        translations = [item["output"] for item in compare_list]
        start_chunk = compare_list[-1]["chunk"] if compare_list else 0
        return translations, compare_list, start_chunk, data

    @staticmethod
    def _generate_compare_list(
        chunk: list[tuple[int, str]], translated: list[str], chunk_id: int, atomic: bool, context: TranslationContext
    ) -> list[dict]:
        """
        Generate a comparison list for the translated chunk.

        This method creates a detailed record of each translation, including the original text,
        translated text, and metadata about the translation process.

        Args:
            chunk (List[Tuple[int, str]]): Original chunk of text.
            translated (List[str]): Translated texts.
            chunk_id (int): ID of the current chunk.
            atomic (bool): Whether atomic translation was used.
            context (TranslationContext): Current translation context.

        Returns:
            List[dict]: List of dictionaries containing comparison information.
        """
        return [
            {
                "chunk": chunk_id,
                "idx": item[0] if item else "N/A",
                "method": "atomic" if atomic else "chunked",
                "model": str(context.model),
                "input": item[1] if item else "N/A",
                "output": trans if trans else "N/A",
            }
            for (item, trans) in zip_longest(chunk, translated)
        ]


class LeanTranslator(BaseLLMTranslator):
    """Lightweight LLM translator optimised for token efficiency.

    Compared to :class:`LLMTranslator`, this class:

    * Uses a simplified single-task prompt (~150 tokens vs ~839).
    * Replaces accumulated historical summaries with a fixed-budget
      sliding window of recent translation pairs.
    * Aligns parsed output by subtitle IDs instead of strict line-count
      matching, tolerating minor omissions and ID offsets.
    * Supports a separate *cr_chatbot* for Context Review so that a
      cheap/fast MT model can handle translation while a larger LLM
      handles CR.
    """

    SLIDING_WINDOW_BUDGET = 150  # Max tokens for the recent-translations window
    ANCHOR_OFFSET_TOLERANCE = 3  # Max ID offset for fuzzy anchor matching
    ATOMIC_FILL_THRESHOLD = 0.2  # Missing ≤20% → atomic fill; >20% → full retry
    MAX_CHUNK_RETRIES = 2  # Full-chunk retries before falling back
    MIN_SPLIT_SIZE = 3  # Minimum chunk size for binary-split retry; below this, use atomic
    OUTPUT_RATIO = 1.5  # Lean: target-only output with safety margin

    def __init__(
        self,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        cr_chatbot: ChatBot | None = None,
        chunk_size: int = BaseLLMTranslator.CHUNK_SIZE,
        timestamps: list[tuple[float, float | None]] | None = None,
        enable_cr: bool = True,
        chunked_guideline: bool = False,
        prompt_profile: str = "default",
    ):
        """
        Args:
            chatbot: Primary ChatBot for translation.
            retry_chatbot: Optional fallback ChatBot for retries.
            cr_chatbot: Optional separate ChatBot for Context Review.
                When *enable_cr* is True and *cr_chatbot* is None, *chatbot*
                is used for CR.
            chunk_size: Maximum lines per chunk.
            timestamps: Optional per-line (start, end) for scene-aware chunking.
            enable_cr: Whether to run Context Review before translation.
            chunked_guideline: Enable chunked guideline generation for long texts.
        """
        super().__init__(
            chatbot=chatbot,
            retry_chatbot=retry_chatbot,
            cr_chatbot=cr_chatbot,
            chunk_size=chunk_size,
            timestamps=timestamps,
            chunked_guideline=chunked_guideline,
            prompt_profile=prompt_profile,
        )
        self.enable_cr = enable_cr

    @staticmethod
    def _prompt_messages(prompter: LeanTranslatePrompter, user_msg: str, extra_user: str | None = None) -> list[dict]:
        messages: list[dict] = []
        system_msg = prompter.system()
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        if extra_user:
            messages.append({"role": "user", "content": extra_user})
        return messages

    def _align_translations(
        self, expected_ids: list[int], parsed: dict[int, str], *, exact_only: bool = False
    ) -> tuple[list[str | None], list[int]]:
        """Align parsed translations to the expected line IDs.

        Returns ``(aligned, missing_ids)`` where *aligned* has one entry
        per *expected_ids* (``None`` for unmatched lines) and *missing_ids*
        lists the IDs that could not be matched.
        """
        aligned: list[str | None] = []
        missing: list[int] = []

        for eid in expected_ids:
            # Exact match
            if eid in parsed:
                aligned.append(parsed.pop(eid))
                continue

            if exact_only:
                aligned.append(None)
                missing.append(eid)
                continue

            # Fuzzy match: try offsets ±1 … ±ANCHOR_OFFSET_TOLERANCE
            found = False
            for offset in range(1, self.ANCHOR_OFFSET_TOLERANCE + 1):
                for candidate in (eid + offset, eid - offset):
                    if candidate in parsed:
                        aligned.append(parsed.pop(candidate))
                        found = True
                        break
                if found:
                    break

            if not found:
                aligned.append(None)
                missing.append(eid)

        return aligned, missing

    @staticmethod
    def _neighbor_context(texts: list[str], line_id: int, radius: int = 2) -> str:
        index = line_id - 1
        start = max(0, index - radius)
        end = min(len(texts), index + radius + 1)
        return "\n".join(
            f"[{item_index + 1}] {texts[item_index]}" for item_index in range(start, end) if item_index != index
        )

    @staticmethod
    def _build_sliding_window(recent_pairs: Sequence[tuple[int, str, str] | list], budget: int) -> str:
        """Build a sliding-window context string from recent translation pairs.

        Args:
            recent_pairs: List of ``(line_id, source, translation)`` triples
                (or lists after JSON roundtrip), ordered chronologically
                (oldest first).
            budget: Maximum token budget for the window.

        Returns:
            Formatted string of recent pairs that fits within *budget*.
        """
        lines: list[str] = []
        used = 0
        for line_id, src, tgt in reversed(recent_pairs):
            entry = f"#{line_id} {src} | {tgt}"
            tokens = get_text_token_number(entry)
            if used + tokens > budget:
                break
            lines.append(entry)
            used += tokens
        return "\n".join(reversed(lines))

    @staticmethod
    def _build_preceding_window(texts: list[str], translations: list[str], *, before_id: int, budget: int) -> str:
        """Build bounded history directly from the nearest preceding lines."""
        lines: list[str] = []
        used = 0
        for line_id in range(before_id - 1, 0, -1):
            entry = f"#{line_id} {texts[line_id - 1]} | {translations[line_id - 1]}"
            tokens = get_text_token_number(entry)
            if used + tokens > budget:
                break
            lines.append(entry)
            used += tokens
        return "\n".join(reversed(lines))

    @staticmethod
    def _terminology_text(
        translation_brief: TranslationBrief | None,
        glossary: dict[str, str] | None,
        resolved_glossary: Sequence[ResolvedGlossaryEntry] | None,
        *,
        base_terminology: str = "",
        excluded_task_sources: set[str] | None = None,
    ) -> str:
        """Render one normalized terminology source for every translation prompt."""
        excluded_task_sources = excluded_task_sources or set()
        if resolved_glossary is not None:
            entries = [
                (item.source, item.target)
                for item in resolved_glossary
                if item.enabled
                and not (
                    item.origin == GlossaryOrigin.TASK
                    and " ".join(unicodedata.normalize("NFKC", item.source).split()).casefold() in excluded_task_sources
                )
            ]
        else:
            ordered: dict[str, tuple[str, str]] = {}
            unparsed: list[str] = []

            # Older Context Review responses expose a pre-rendered glossary.
            # Parse the common YAML/Markdown bullets so later, higher-priority
            # sources can replace them instead of being appended a second time.
            for raw_line in base_terminology.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                value = line[1:].strip() if line.startswith("-") else line
                delimiter = " -> " if " -> " in value else ": " if ": " in value else None
                if delimiter is None:
                    unparsed.append(raw_line)
                    continue
                source, target = (item.strip() for item in value.split(delimiter, 1))
                if not source or not target:
                    unparsed.append(raw_line)
                    continue
                key = " ".join(unicodedata.normalize("NFKC", source).split()).casefold()
                ordered[key] = (source, target)
            if translation_brief is not None:
                for item in translation_brief.glossary:
                    key = " ".join(unicodedata.normalize("NFKC", item.source).split()).casefold()
                    ordered[key] = (item.source, item.target)
            for source, target in (glossary or {}).items():
                key = " ".join(unicodedata.normalize("NFKC", source).split()).casefold()
                ordered[key] = (source, target)
            entries = list(ordered.values())
            return "\n".join([*unparsed, *(f"- {source}: {target}" for source, target in entries)])
        return "\n".join(f"- {source}: {target}" for source, target in entries)

    @staticmethod
    def _characters_with_task_glossary(
        translation_brief: TranslationBrief, resolved_glossary: Sequence[ResolvedGlossaryEntry] | None
    ) -> tuple[str, set[str]]:
        """Apply explicit task terms to matching Brief characters once in the prompt."""
        if resolved_glossary is None:
            return translation_brief.characters_text(), set()

        task_entries = [item for item in resolved_glossary if item.enabled and item.origin == GlossaryOrigin.TASK]
        lines: list[str] = []
        consumed_sources: set[str] = set()
        for character in translation_brief.characters:
            selected = next(
                (
                    entry
                    for entry in task_entries
                    if GlossaryService._entry_matches_source(entry, character.source_name)
                ),
                None,
            )
            target = selected.target if selected is not None else character.target_name
            lines.append(f"- {character.source_name} -> {target}")
            if selected is not None:
                consumed_sources.add(" ".join(unicodedata.normalize("NFKC", selected.source).split()).casefold())
        return "\n".join(lines), consumed_sources

    def translate(
        self,
        texts: str | list[str],
        src_lang: str,
        target_lang: str,
        info: TranslateInfo | None = None,
        compare_path: Path = Path("translate_intermediate.json"),
        translation_brief: TranslationBrief | None = None,
        checkpoint_metadata: dict | None = None,
        chunk_plans: list[TranslationChunkPlan] | None = None,
        context_timeline: ContextTimeline | None = None,
        resolved_glossary: Sequence[ResolvedGlossaryEntry] | None = None,
    ) -> list[str]:
        """Translate *texts* using the lean single-task prompt strategy."""
        if info is None:
            info = TranslateInfo()
        if not isinstance(texts, list):
            texts = [texts]
        if not texts:
            return []

        prompter = create_lean_translate_prompter(src_lang, target_lang, self.prompt_profile)

        # --- Context Review (optional) --------------------------------
        summary = ""
        characters = ""
        terminology = ""
        guideline = ""
        style = ""

        cr_fee_start = len(self.cr_chatbot.api_fees) if self.cr_chatbot else 0
        retry_fee_start = len(self.retry_chatbot.api_fees) if self.retry_chatbot else 0
        fee_start = len(self.chatbot.api_fees)

        plans = chunk_plans or self.plan_chunks(texts)
        planner_signature = chunk_plan_signature(plans, timestamps=self.timestamps)
        if checkpoint_metadata and checkpoint_metadata.get("chunk_signature") not in {None, planner_signature}:
            raise ValueError("Checkpoint chunk signature does not match the active translation plan.")
        checkpoint_metadata = {
            **(checkpoint_metadata or {}),
            "chunk_planner_version": CHUNK_PLANNER_VERSION,
            "chunk_signature": planner_signature,
        }

        # Load checkpoint
        translations, compare_list, start_chunk, ctx = self._load_checkpoint(compare_path)
        if compare_list and (
            ctx.get("chunk_planner_version") != CHUNK_PLANNER_VERSION or ctx.get("chunk_signature") != planner_signature
        ):
            logger.warning("Discarding translation progress created by an incompatible chunk planner.")
            translations = []
            compare_list = []
            start_chunk = 0
            for key in (
                "recent_pairs",
                "raw_hymt2_translations",
                "final_translations",
                "review_chunks",
                "reviewed_chunks",
                "review_failed_chunks",
                "review_results",
                "review_incomplete",
                "review_protocol_version",
            ):
                ctx.pop(key, None)
        recent_pairs: list[tuple[int, str, str] | list] = ctx.get("recent_pairs", [])
        guideline = ctx.get("guideline", "")

        if self.enable_cr and not guideline:
            cr_bot = self.cr_chatbot or self.chatbot
            logger.info("Building translation guideline via Context Review.")
            context_reviewer = ContextReviewerAgent(
                src_lang,
                target_lang,
                info,
                chatbot=cr_bot,
                retry_chatbot=self.retry_chatbot,
                chunked_guideline=self.chunked_guideline,
                prompter=LeanContextReviewPrompter(src_lang, target_lang),
            )
            guideline = context_reviewer.build_context(
                texts, title=info.title or "", glossary=info.glossary, forced_glossary=info.forced_glossary
            )
            logger.debug(f"Translation Guideline:\n{guideline}")

        if guideline:
            summary, characters, terminology = self._extract_cr_context(guideline)

        if translation_brief is not None:
            summary = translation_brief.summary
            characters, character_term_sources = self._characters_with_task_glossary(
                translation_brief, resolved_glossary
            )
            style = translation_brief.tone_style
        else:
            character_term_sources = set()

        terminology = self._terminology_text(
            translation_brief,
            info.glossary,
            resolved_glossary,
            base_terminology=terminology,
            excluded_task_sources=character_term_sources,
        )

        # --- Chunk and translate --------------------------------------
        chunks = [plan.pairs() for plan in plans]
        if context_timeline is not None:
            if len(context_timeline.chunks) != len(plans):
                raise ValueError("ContextTimeline chunk count does not match the translation plan.")
            for plan, dynamic_context in zip(plans, context_timeline.chunks):
                if dynamic_context.chunk_id != plan.chunk_id or dynamic_context.segment_ids != plan.segment_ids:
                    raise ValueError(f"ContextTimeline does not align with translation chunk {plan.chunk_id}.")
        logger.info(f"Translating {info.title}: {len(chunks)} chunks, {len(texts)} lines in total.")

        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            expected_ids = [line_id for line_id, _ in chunk]
            source_texts = {line_id: text for line_id, text in chunk}
            dynamic_context = context_timeline.context_for(i) if context_timeline is not None else None
            story_so_far = dynamic_context.story_so_far if dynamic_context else ""
            current_scene = dynamic_context.current_scene if dynamic_context else ""

            # Build prompts (with and without terminology for glossary-removal retry)
            window_str = self._build_sliding_window(recent_pairs, self.SLIDING_WINDOW_BUDGET)
            formatted = prompter.format_texts(chunk)
            user_msg = prompter.user(
                formatted,
                summary=summary,
                characters=characters,
                terminology=terminology,
                sliding_window=window_str,
                **({"style": style} if getattr(prompter, "exact_id_alignment", False) else {}),
                **(
                    {"story_so_far": story_so_far, "current_scene": current_scene}
                    if getattr(prompter, "exact_id_alignment", False)
                    else {}
                ),
            )
            user_msg_no_glossary = (
                prompter.user(
                    formatted,
                    summary=summary,
                    characters=characters,
                    terminology="",
                    sliding_window=window_str,
                    **({"style": style} if getattr(prompter, "exact_id_alignment", False) else {}),
                    **(
                        {"story_so_far": story_so_far, "current_scene": current_scene}
                        if getattr(prompter, "exact_id_alignment", False)
                        else {}
                    ),
                )
                if terminology
                else None
            )

            # Update validator for this chunk's expected IDs
            prompter.update_expected_ids(expected_ids)

            # Translate with retries
            translated, used_atomic = self._translate_lean_chunk(
                prompter,
                user_msg,
                expected_ids,
                source_texts,
                src_lang,
                target_lang,
                user_msg_no_glossary=user_msg_no_glossary,
                prompt_context={
                    "summary": summary,
                    "characters": characters,
                    "terminology": terminology,
                    "sliding_window": window_str,
                    "style": style,
                    "story_so_far": story_so_far,
                    "current_scene": current_scene,
                },
                all_texts=texts,
            )

            translations.extend(translated)

            # Update sliding window with new pairs
            for line_id, text in chunk:
                idx = expected_ids.index(line_id)
                recent_pairs.append((line_id, text, translated[idx]))

            # Build compare list and save checkpoint
            context_obj = TranslationContext(guideline=guideline)
            compare_records = self._generate_compare_list(chunk, translated, i, used_atomic, context_obj)
            mode = (checkpoint_metadata or {}).get("hymt2_mode")
            if mode:
                for record in compare_records:
                    record["mode"] = mode
                    if dynamic_context is not None:
                        record["timeline_chunk_id"] = dynamic_context.chunk_id
            compare_list.extend(compare_records)
            self._save_checkpoint(
                compare_path,
                compare_list,
                {
                    **(checkpoint_metadata or {}),
                    "guideline": guideline,
                    "recent_pairs": recent_pairs,
                    "hymt2_metrics": self.metrics,
                    **(
                        {"raw_hymt2_translations": translations, "pipeline_stage": "translation"}
                        if mode == "pro"
                        else {}
                    ),
                },
            )

            logger.info(f"Translated {info.title}: {i}/{len(chunks)}")

        self.api_fee += sum(self.chatbot.api_fees[fee_start:])
        if self.cr_chatbot:
            self.api_fee += sum(self.cr_chatbot.api_fees[cr_fee_start:])
        if self.retry_chatbot:
            self.api_fee += sum(self.retry_chatbot.api_fees[retry_fee_start:])

        logger.info(f"Translation complete for {info.title}. Fee: {self.api_fee:.4f} USD")
        return translations

    def translate_targeted(
        self,
        texts: list[str],
        current_translations: list[str],
        segment_ids: list[int],
        *,
        src_lang: str,
        target_lang: str,
        info: TranslateInfo | None = None,
        translation_brief: TranslationBrief | None = None,
        chunk_plans: list[TranslationChunkPlan] | None = None,
        context_timeline: ContextTimeline | None = None,
        resolved_glossary: Sequence[ResolvedGlossaryEntry] | None = None,
        completed_results: dict[int, str] | None = None,
        checkpoint_hook: Callable[[dict[int, str], int], None] | None = None,
    ) -> dict[int, str]:
        """Retranslate selected IDs while preserving their original parent chunks."""
        if len(texts) != len(current_translations):
            raise ValueError("Source and current translation counts must match for targeted translation.")
        selected = sorted(set(segment_ids))
        if not selected or selected[0] < 1 or selected[-1] > len(texts):
            raise ValueError("Targeted translation IDs must be within the source subtitle range.")
        info = info or TranslateInfo()
        plans = chunk_plans or self.plan_chunks(texts)
        if context_timeline is not None:
            if len(context_timeline.chunks) != len(plans):
                raise ValueError("ContextTimeline chunk count does not match targeted translation plans.")
            for plan, chunk_context in zip(plans, context_timeline.chunks):
                if plan.chunk_id != chunk_context.chunk_id or plan.segment_ids != chunk_context.segment_ids:
                    raise ValueError(f"ContextTimeline does not align with translation chunk {plan.chunk_id}.")

        summary = translation_brief.summary if translation_brief is not None else ""
        if translation_brief is not None:
            characters, character_term_sources = self._characters_with_task_glossary(
                translation_brief, resolved_glossary
            )
        else:
            characters, character_term_sources = "", set()
        terminology = self._terminology_text(
            translation_brief, info.glossary, resolved_glossary, excluded_task_sources=character_term_sources
        )
        style = translation_brief.tone_style if translation_brief is not None else ""

        prompter = create_lean_translate_prompter(src_lang, target_lang, self.prompt_profile)
        selected_set = set(selected)
        results: dict[int, str] = {
            int(line_id): value for line_id, value in (completed_results or {}).items() if int(line_id) in selected_set
        }
        fee_start = len(self.chatbot.api_fees)
        for plan in plans:
            active_ids = [line_id for line_id in plan.segment_ids if line_id in selected_set and line_id not in results]
            if not active_ids:
                continue
            chunk = [(line_id, texts[line_id - 1]) for line_id in active_ids]
            source_map = dict(chunk)
            sliding_window = self._build_preceding_window(
                texts, current_translations, before_id=active_ids[0], budget=self.SLIDING_WINDOW_BUDGET
            )
            neighboring_context = "\n".join(self._neighbor_context(texts, line_id) for line_id in active_ids)
            dynamic_context = context_timeline.context_for(plan.chunk_id) if context_timeline is not None else None
            prompt_context = {
                "summary": summary,
                "characters": characters,
                "terminology": terminology,
                "sliding_window": sliding_window,
                "style": style,
                "neighboring_context": neighboring_context,
                "story_so_far": dynamic_context.story_so_far if dynamic_context else "",
                "current_scene": dynamic_context.current_scene if dynamic_context else "",
            }
            user_context = prompt_context
            if not getattr(prompter, "exact_id_alignment", False):
                user_context = {
                    key: value
                    for key, value in prompt_context.items()
                    if key in {"summary", "characters", "terminology", "sliding_window"}
                }
            formatted = prompter.format_texts(chunk)
            user_msg = prompter.user(formatted, **user_context)
            user_msg_no_glossary = None
            if terminology:
                without_glossary = dict(user_context)
                without_glossary["terminology"] = ""
                user_msg_no_glossary = prompter.user(formatted, **without_glossary)
            prompter.update_expected_ids(active_ids)
            translated, _ = self._translate_lean_chunk(
                prompter,
                user_msg,
                active_ids,
                source_map,
                src_lang,
                target_lang,
                user_msg_no_glossary=user_msg_no_glossary,
                prompt_context=prompt_context,
                all_texts=texts,
            )
            results.update(zip(active_ids, translated))
            if checkpoint_hook is not None:
                checkpoint_hook(dict(results), plan.chunk_id)

        self.api_fee += sum(self.chatbot.api_fees[fee_start:])
        if sorted(results) != selected:
            raise ValueError("Targeted translation did not return every requested subtitle ID.")
        return results

    def _translate_lean_chunk(
        self,
        prompter: LeanTranslatePrompter,
        user_msg: str,
        expected_ids: list[int],
        source_texts: dict[int, str],
        src_lang: str,
        target_lang: str,
        user_msg_no_glossary: str | None = None,
        prompt_context: dict[str, str] | None = None,
        all_texts: list[str] | None = None,
    ) -> tuple[list[str], bool]:
        """Translate one chunk with the full retry chain.

        Retry chain:
        1. Primary chatbot (up to ``MAX_CHUNK_RETRIES`` attempts).
           - Attempt 0: original prompt.
           - Attempt 1+: retry instruction appended; glossary-removal
             variant used if available.
        2. ``retry_chatbot`` (1 attempt, if available).
        3. Binary split: halve the chunk and recurse on each half.
        4. Atomic fallback for chunks ≤ ``MIN_SPLIT_SIZE``.

        At any point, if ≤ ``ATOMIC_FILL_THRESHOLD`` lines are missing,
        only the missing lines are filled with atomic translation.

        Returns ``(translations, used_atomic)`` where *used_atomic* is
        ``True`` if any atomic translation was used (fill or full fallback).
        """
        result = self._try_chatbot_attempts(
            self.chatbot,
            prompter,
            user_msg,
            expected_ids,
            source_texts,
            src_lang,
            target_lang,
            user_msg_no_glossary,
            prompt_context=prompt_context,
            all_texts=all_texts,
        )
        if result is not None:
            return result

        # Compute min_tokens for retry and fallback paths.
        chunk_data = [(eid, source_texts[eid]) for eid in expected_ids]
        min_tokens = self._estimate_output_tokens(chunk_data)

        # Step 2: retry_chatbot
        if self.retry_chatbot:
            logger.info("Primary chatbot exhausted, trying retry chatbot.")
            result = self._try_single_attempt(
                self.retry_chatbot,
                prompter,
                user_msg,
                expected_ids,
                source_texts,
                src_lang,
                target_lang,
                min_tokens=min_tokens,
                prompt_context=prompt_context,
                all_texts=all_texts,
            )
            if result is not None:
                return result

        # Step 3: binary split (always involves atomic at the leaves)
        logger.warning("All chatbot retries exhausted, attempting binary-split retry.")
        translations = self._split_and_translate_lean(
            prompter,
            expected_ids,
            source_texts,
            src_lang,
            target_lang,
            user_msg_no_glossary=user_msg_no_glossary,
            prompt_context=prompt_context,
            all_texts=all_texts,
        )
        return translations, True

    def _try_chatbot_attempts(
        self,
        bot: ChatBot,
        prompter: LeanTranslatePrompter,
        user_msg: str,
        expected_ids: list[int],
        source_texts: dict[int, str],
        src_lang: str,
        target_lang: str,
        user_msg_no_glossary: str | None = None,
        prompt_context: dict[str, str] | None = None,
        all_texts: list[str] | None = None,
    ) -> tuple[list[str], bool] | None:
        """Run up to ``MAX_CHUNK_RETRIES`` attempts on *bot*.

        Returns ``(translations, used_atomic)`` or ``None`` if all attempts
        had >``ATOMIC_FILL_THRESHOLD`` missing lines.
        """
        base_messages = self._prompt_messages(prompter, user_msg)

        # Compute min_tokens for this chunk.
        chunk_data = [(eid, source_texts[eid]) for eid in expected_ids]
        min_tokens = self._estimate_output_tokens(chunk_data)

        for attempt in range(self.MAX_CHUNK_RETRIES):
            # Build messages for this attempt.
            if attempt == 0:
                messages = base_messages
            else:
                # Retry: append retry instruction; use glossary-removal variant if available.
                retry_user = user_msg_no_glossary if user_msg_no_glossary else user_msg
                if user_msg_no_glossary and list(expected_ids) not in self.metrics["glossary_removed_retry_chunks"]:
                    self.metrics["glossary_removed_retry_chunks"].append(list(expected_ids))
                messages = self._prompt_messages(prompter, retry_user, extra_user=prompter.retry_instruction())

            try:
                result = self._try_single_attempt(
                    bot,
                    prompter,
                    messages,
                    expected_ids,
                    source_texts,
                    src_lang,
                    target_lang,
                    min_tokens=min_tokens,
                    prompt_context=prompt_context,
                    all_texts=all_texts,
                )
            except LengthExceedException:
                logger.warning(
                    f"Lean chunk attempt {attempt + 1}/{self.MAX_CHUNK_RETRIES} failed "
                    f"due to output length limit — skipping remaining primary retries."
                )
                return None  # Same config will also fail on retry; go to fallback.

            if result is not None:
                return result

            logger.warning(f"Lean chunk attempt {attempt + 1}/{self.MAX_CHUNK_RETRIES} failed.")
            self.metrics["retries"] += 1

        return None

    def _try_single_attempt(
        self,
        bot: ChatBot,
        prompter: LeanTranslatePrompter,
        messages: str | list[dict],
        expected_ids: list[int],
        source_texts: dict[int, str],
        src_lang: str,
        target_lang: str,
        min_tokens: int | None = None,
        prompt_context: dict[str, str] | None = None,
        all_texts: list[str] | None = None,
    ) -> tuple[list[str], bool] | None:
        """Execute one LLM call and attempt ID-based alignment.

        *messages* is either a pre-built message list or a plain user-msg
        string (in which case a system+user pair is constructed).

        Returns ``(translations, used_atomic)`` or ``None`` if
        >``ATOMIC_FILL_THRESHOLD`` lines are missing.
        """
        if isinstance(messages, str):
            messages = self._prompt_messages(prompter, messages)

        validator = getattr(prompter, "validator", None)
        issue_count = len(getattr(validator, "issues", []))
        try:
            responses = bot.message(messages, output_checker=prompter.check_format, min_tokens=min_tokens)
            raw = bot.get_content(responses[0])
        except LengthExceedException:
            raise
        except ChatBotException:
            logger.error("ChatBot failed for lean chunk.")
            return None
        finally:
            new_issues = getattr(validator, "issues", [])[issue_count:]
            self.metrics["validator_issues"].extend({"ids": list(expected_ids), "issue": issue} for issue in new_issues)

        if not raw:
            return None

        parsed = prompter.parse_translations(raw)
        aligned, missing = self._align_translations(
            expected_ids, parsed, exact_only=bool(getattr(prompter, "exact_id_alignment", False))
        )
        missing_ratio = len(missing) / len(expected_ids) if expected_ids else 0.0

        if missing_ratio == 0:
            return [t for t in aligned if t is not None], False

        if missing_ratio <= self.ATOMIC_FILL_THRESHOLD:
            logger.info(
                f"Translation alignment: {len(missing)}/{len(expected_ids)} lines missing, "
                "filling with atomic translation."
            )
            missing_texts = [source_texts[mid] for mid in missing]
            contexts = [self._neighbor_context(all_texts or [], line_id) for line_id in missing]
            guideline = "\n".join(value for value in (prompt_context or {}).values() if value)
            if any(contexts) or guideline:
                atomic_results = self.atomic_translate(
                    bot, missing_texts, src_lang, target_lang, contexts=contexts, guideline=guideline
                )
            else:
                atomic_results = self.atomic_translate(bot, missing_texts, src_lang, target_lang)
            self.metrics["atomic_ids"].extend(missing)
            atomic_map = dict(zip(missing, atomic_results))
            return [atomic_map[eid] if t is None else t for eid, t in zip(expected_ids, aligned)], True

        logger.warning(
            f"Translation alignment: {len(missing)}/{len(expected_ids)} lines missing ({missing_ratio:.0%})."
        )
        return None

    def _split_and_translate_lean(
        self,
        prompter: LeanTranslatePrompter,
        expected_ids: list[int],
        source_texts: dict[int, str],
        src_lang: str,
        target_lang: str,
        user_msg_no_glossary: str | None = None,
        prompt_context: dict[str, str] | None = None,
        all_texts: list[str] | None = None,
        split_depth: int = 0,
    ) -> list[str]:
        """Binary-split a chunk and translate each half recursively.

        Falls back to atomic translation when a half is ≤ ``MIN_SPLIT_SIZE``.
        """
        if len(expected_ids) <= self.MIN_SPLIT_SIZE:
            if not expected_ids:
                return []
            logger.info(f"Chunk below MIN_SPLIT_SIZE ({self.MIN_SPLIT_SIZE}), using atomic translation.")
            contexts = [self._neighbor_context(all_texts or [], line_id) for line_id in expected_ids]
            guideline = "\n".join(value for value in (prompt_context or {}).values() if value)
            self.metrics["atomic_ids"].extend(expected_ids)
            if any(contexts) or guideline:
                return self.atomic_translate(
                    self.chatbot,
                    [source_texts[eid] for eid in expected_ids],
                    src_lang,
                    target_lang,
                    contexts=contexts,
                    guideline=guideline,
                )
            return self.atomic_translate(
                self.chatbot, [source_texts[eid] for eid in expected_ids], src_lang, target_lang
            )

        mid = len(expected_ids) // 2
        left_ids, right_ids = expected_ids[:mid], expected_ids[mid:]
        logger.info(f"Splitting chunk into {len(left_ids)}+{len(right_ids)} lines for retry.")
        self.metrics["splits"] += 1
        next_depth = split_depth + 1
        self.metrics["max_split_depth"] = max(self.metrics["max_split_depth"], next_depth)

        results: list[str] = []
        for half_ids in (left_ids, right_ids):
            half_chunk = [(eid, source_texts[eid]) for eid in half_ids]
            formatted = prompter.format_texts(half_chunk)
            context_kwargs = dict(prompt_context or {})
            if getattr(prompter, "exact_id_alignment", False):
                context_kwargs["neighboring_context"] = "\n".join(
                    self._neighbor_context(all_texts or [], line_id) for line_id in half_ids
                )
            else:
                context_kwargs = {
                    key: value
                    for key, value in context_kwargs.items()
                    if key in {"summary", "characters", "terminology", "sliding_window"}
                }
            half_user_msg = prompter.user(formatted, **context_kwargs)
            prompter.update_expected_ids(half_ids)

            half_source = {eid: source_texts[eid] for eid in half_ids}
            min_tokens = self._estimate_output_tokens(half_chunk)

            def _try_bot(bot: ChatBot) -> tuple[list[str], bool] | None:
                try:
                    return self._try_single_attempt(
                        bot,
                        prompter,
                        half_user_msg,
                        half_ids,
                        half_source,
                        src_lang,
                        target_lang,
                        min_tokens=min_tokens,
                        prompt_context=prompt_context,
                        all_texts=all_texts,
                    )
                except LengthExceedException:
                    return None

            # Try primary chatbot
            result = _try_bot(self.chatbot)
            if result is not None:
                results.extend(result[0])
                continue

            # Try retry chatbot
            if self.retry_chatbot:
                result = _try_bot(self.retry_chatbot)
                if result is not None:
                    results.extend(result[0])
                    continue

            # Recurse
            results.extend(
                self._split_and_translate_lean(
                    prompter,
                    half_ids,
                    half_source,
                    src_lang,
                    target_lang,
                    user_msg_no_glossary=None,
                    prompt_context=prompt_context,
                    all_texts=all_texts,
                    split_depth=next_depth,
                )
            )

        return results

    @staticmethod
    def _extract_cr_context(guideline: str) -> tuple[str, str, str]:
        """Extract summary, characters, and terminology from a CR guideline.

        Supports both ``### Section:`` (markdown) and ``section:`` (YAML-like)
        formats produced by :class:`ContextReviewPrompter` and
        :class:`LeanContextReviewPrompter` respectively.

        Returns ``(summary, characters, terminology)``.  Empty string for
        any section not found.
        """

        # Pattern matches both "### Glossary:" (markdown) and "glossary:" (YAML)
        # at the start of a line.  Captures everything after the header
        # (including same-line content like "summary: text here") until the
        # next section header or end of string.
        def _extract(section_name: str) -> str:
            pattern = rf"^(?:###\s*)?{section_name}[:\s]*(.*?)(?=\n(?:###\s*)?\w+[:\s]*(?:\n|$)|\Z)"
            match = re.search(pattern, guideline, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else ""

        summary = _extract("summary")
        characters = _extract("characters")
        terminology = _extract("glossary")

        return summary, characters, terminology


class LLMTranslator(BaseLLMTranslator):
    """Translator using Large Language Models for translation.

    This class implements a sophisticated translation process using chunking,
    context-aware translation, and fallback mechanisms.
    """

    RETRY_STREAK = 10  # Number of consecutive chunks to use retry model after a failure
    LINE_MISMATCH_RETRIES = 2  # Max attempts per agent for line-count mismatch recovery
    MIN_SPLIT_SIZE = 3  # Minimum chunk size for binary-split retry; below this, use atomic
    OUTPUT_RATIO = 2.5  # Standard: source + target output (Original>/Translation> + summary/scene)

    def __init__(
        self,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        cr_chatbot: ChatBot | None = None,
        chunk_size: int = BaseLLMTranslator.CHUNK_SIZE,
        intercept_line: int | None = None,
        chunked_guideline: bool = False,
        timestamps: list[tuple[float, float | None]] | None = None,
    ):
        """
        Initialize the LLMTranslator with given parameters.

        Args:
            chatbot: Primary ChatBot instance for translation.
            retry_chatbot: Optional ChatBot instance for retry attempts.
            cr_chatbot: Optional separate ChatBot for Context Review.
                When None, *chatbot* is used for CR.
            chunk_size: Maximum lines per chunk for processing.
            intercept_line: Line number to intercept translation, useful for debugging.
            chunked_guideline: Enable chunked guideline generation for long texts. Default: False.
            timestamps: Optional list of (start, end) per line for scene-aware chunking.
                When provided, chunk splitting respects scene boundaries (time gaps > SCENE_THRESHOLD).
                When None, only token budget and line-count limits apply.
        """
        super().__init__(
            chatbot=chatbot,
            retry_chatbot=retry_chatbot,
            cr_chatbot=cr_chatbot,
            chunk_size=chunk_size,
            timestamps=timestamps,
            chunked_guideline=chunked_guideline,
        )
        self.intercept_line = intercept_line
        self.use_retry_cnt = 0

    @staticmethod
    def _is_valid_translation(translated: list[str] | None, expected_len: int) -> bool:
        """Check whether a translation result is usable (non-empty and correct line count)."""
        return translated is not None and len(translated) == expected_len

    def _translate_chunk(
        self,
        translator_agent: ChunkedTranslatorAgent,
        chunk: list[tuple[int, str]],
        context: TranslationContext,
        chunk_id: int,
        retry_agent: ChunkedTranslatorAgent | None = None,
    ) -> tuple[list[str], TranslationContext]:
        """
        Translate a single chunk of text, with retry mechanism.

        Tries the primary agent first, then optionally falls back to a retry agent.
        Each agent attempt includes a glossary-removal retry when the line count
        is inconsistent.

        Returns the best available result. The caller should check whether
        ``len(translated) == len(chunk)`` to decide if further fallback
        (e.g. split or atomic translation) is needed.
        """
        expected = len(chunk)
        max_attempts = 1 if retry_agent else self.LINE_MISMATCH_RETRIES

        def _try_agent(agent: ChunkedTranslatorAgent) -> tuple[list[str] | None, TranslationContext | None]:
            """Try an agent up to max_attempts times for line-count mismatch recovery.

            Each attempt includes a glossary-removal sub-retry when applicable.
            Returns the best available result (may still have mismatched line count).
            """
            trans: list[str] | None = None
            ctx: TranslationContext | None = None

            for attempt in range(max_attempts):
                try:
                    trans, ctx = agent.translate_chunk(chunk_id, chunk, context)
                except ChatBotException:
                    logger.error(f"Failed to translate chunk {chunk_id}.")
                    return None, None

                if self._is_valid_translation(trans, expected):
                    return trans, ctx

                # Line count mismatch — retry without glossary if applicable.
                if trans is not None and agent.info.glossary:
                    logger.warning(
                        f"Agent {agent}: Removing glossary for chunk {chunk_id} due to inconsistent translation."
                    )
                    try:
                        trans, ctx = agent.translate_chunk(chunk_id, chunk, context, use_glossary=False)
                    except ChatBotException:
                        logger.error(f"Failed to translate chunk {chunk_id}.")

                if self._is_valid_translation(trans, expected):
                    return trans, ctx

                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Chunk {chunk_id} line count mismatch ({len(trans) if trans else 0} vs {expected}),"
                        f" retrying ({attempt + 1}/{max_attempts})."
                    )

            return trans, ctx

        translated: list[str] | None = None
        updated_ctx: TranslationContext | None = None

        # Step 1: Try primary or retry agent based on retry streak.
        if self.use_retry_cnt == 0 or not retry_agent:
            translated, updated_ctx = _try_agent(translator_agent)
        else:
            logger.info(f"Using retry agent for chunk {chunk_id}, remaining retries: {self.use_retry_cnt}")
            translated, updated_ctx = _try_agent(retry_agent)
            self.use_retry_cnt -= 1

        # Step 2: If primary failed and retry agent is available, switch to it.
        if not self._is_valid_translation(translated, expected) and retry_agent and self.use_retry_cnt == 0:
            self.use_retry_cnt = self.RETRY_STREAK
            logger.warning(
                f"Using retry agent {retry_agent} for chunk {chunk_id}, and next {self.use_retry_cnt} chunks."
            )
            translated, updated_ctx = _try_agent(retry_agent)

            # Retry agent also failed — reset streak so next chunk tries primary first.
            if not self._is_valid_translation(translated, expected):
                logger.warning(f"Retry agent also failed for chunk {chunk_id}, resetting retry streak.")
                self.use_retry_cnt = 0

        if not translated:
            raise ChatBotException(f"Failed to translate chunk {chunk_id}.")

        return translated, updated_ctx or context

    def _split_and_translate(
        self,
        translator_agent: ChunkedTranslatorAgent,
        chunk: list[tuple[int, str]],
        context: TranslationContext,
        chunk_id: int,
        src_lang: str,
        target_lang: str,
        retry_agent: ChunkedTranslatorAgent | None = None,
    ) -> list[str]:
        """Recursively split a chunk in half and translate each half separately.

        Used as a fallback when ``_translate_chunk`` returns a line-count mismatch.
        Splits at the midpoint, translates each half via ``_translate_chunk``, and
        merges the results.  Halves that still mismatch are split again recursively
        until ``MIN_SPLIT_SIZE`` is reached, at which point ``atomic_translate`` is
        used as the final fallback.

        Returns:
            List of translated strings with ``len == len(chunk)``.
        """
        if len(chunk) <= self.MIN_SPLIT_SIZE:
            if not chunk:
                return []
            logger.info(f"Chunk {chunk_id} below MIN_SPLIT_SIZE ({self.MIN_SPLIT_SIZE}), using atomic translation.")
            return self.atomic_translate(self.chatbot, [c[1] for c in chunk], src_lang, target_lang)

        mid = len(chunk) // 2
        left, right = chunk[:mid], chunk[mid:]
        logger.info(f"Splitting chunk {chunk_id} into {len(left)}+{len(right)} lines for retry.")

        results: list[str] = []
        for half in (left, right):
            try:
                translated, _ = self._translate_chunk(
                    translator_agent, half, context, chunk_id, retry_agent=retry_agent
                )
            except ChatBotException:
                translated = None

            if translated is not None and self._is_valid_translation(translated, len(half)):
                results.extend(translated)
            else:
                # Recurse on the failing half.
                results.extend(
                    self._split_and_translate(
                        translator_agent, half, context, chunk_id, src_lang, target_lang, retry_agent
                    )
                )

        return results

    def translate(
        self,
        texts: str | list[str],
        src_lang: str,
        target_lang: str,
        info: TranslateInfo | None = None,
        compare_path: Path = Path("translate_intermediate.json"),
    ) -> list[str]:
        """
        Translate a list of texts from source language to target language.

        This method implements the main translation process:
        1. Initialize translation agents and chunk the input texts.
        2. Build or load a translation guideline.
        3. Translate each chunk, maintaining context between chunks.
        4. Handle translation failures with retry mechanisms and atomic translation.
        5. Save intermediate results for potential resumption.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.
            info (TranslateInfo): Additional translation information like title and glossary.
            compare_path (Path): Path to save intermediate results for potential resumption.

        Returns:
            List[str]: List of translated texts.
        """
        if info is None:
            info = TranslateInfo()

        if not isinstance(texts, list):
            texts = [texts]

        if not texts:
            return []

        translator_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.chatbot)

        retry_agent = (
            ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.retry_chatbot)
            if self.retry_chatbot
            else None
        )

        chunks = self.make_chunks_by_tokens(texts)
        logger.info(f"Translating {info.title}: {len(chunks)} chunks, {len(texts)} lines in total.")

        # Load checkpoint — unpack subclass-specific context from the dict.
        translations, compare_list, start_chunk, ctx = self._load_checkpoint(compare_path)
        summaries: list[str] = ctx.get("summaries", [])
        guideline: str = ctx.get("guideline", "")

        # Record fee baselines before CR and translation so we capture all costs.
        fee_start = len(self.chatbot.api_fees)
        cr_fee_start = len(self.cr_chatbot.api_fees) if self.cr_chatbot else 0
        retry_fee_start = len(self.retry_chatbot.api_fees) if self.retry_chatbot else 0

        if not guideline:
            logger.info("Building translation guideline.")
            cr_bot = self.cr_chatbot or self.chatbot
            context_reviewer = ContextReviewerAgent(
                src_lang,
                target_lang,
                info,
                chatbot=cr_bot,
                retry_chatbot=self.retry_chatbot,
                chunked_guideline=self.chunked_guideline,
            )
            guideline = context_reviewer.build_context(
                texts, title=info.title or "", glossary=info.glossary, forced_glossary=info.forced_glossary
            )
            logger.debug(f"Translation Guideline:\n{guideline}")

        context = TranslationContext(guideline=guideline, previous_summaries=summaries)
        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            atomic = False
            translated, context = self._translate_chunk(translator_agent, chunk, context, i, retry_agent=retry_agent)

            if not self._is_valid_translation(translated, len(chunk)):
                logger.warning(
                    f"Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},"
                    f" attempting binary-split retry."
                )
                translated = self._split_and_translate(
                    translator_agent, chunk, context, i, src_lang, target_lang, retry_agent=retry_agent
                )
                atomic = True

            translations.extend(translated)
            summaries.append(context.summary or "")
            logger.info(f"Translated {info.title}: {i}/{len(chunks)}")
            logger.debug(f"Summary: {context.summary}")
            logger.debug(f"Scene: {context.scene}")

            compare_list.extend(self._generate_compare_list(chunk, translated, i, atomic, context))
            self._save_checkpoint(
                compare_path,
                compare_list,
                {"summaries": summaries, "scene": context.scene or "", "guideline": guideline},
            )
            context.previous_summaries = summaries

        self.api_fee += sum(self.chatbot.api_fees[fee_start:])
        if self.cr_chatbot:
            self.api_fee += sum(self.cr_chatbot.api_fees[cr_fee_start:])
        if self.retry_chatbot:
            self.api_fee += sum(self.retry_chatbot.api_fees[retry_fee_start:])

        logger.info(f"Translation complete for {info.title}. Fee: {self.api_fee:.4f} USD")

        return translations


class MSTranslator(Translator):
    """
    Translator using Microsoft Translator API.
    This class provides an alternative translation method using Microsoft's services.
    """

    def __init__(self):
        """
        Initialize the Microsoft Translator with API key and endpoint.
        The API key is expected to be set in the environment variables.
        """
        self.key = os.environ["MS_TRANSLATOR_KEY"]
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        self.location = "eastasia"
        self.path = "/translate"
        self.constructed_url = self.endpoint + self.path

        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }

    def translate(self, texts: str | list[str], src_lang, target_lang, info=None):  # type: ignore[override]
        params = {"api-version": "3.0", "from": src_lang, "to": target_lang}

        body = [{"text": text} for text in texts]

        try:
            request = requests.post(self.constructed_url, params=params, headers=self.headers, json=body, timeout=20)
        except TimeoutError:
            raise RuntimeError("Failed to connect to Microsoft Translator API.") from None
        response = request.json()

        return json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(",", ": "))
