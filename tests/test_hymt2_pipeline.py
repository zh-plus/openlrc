#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from openlrc.chunking import chunk_plan_signature, plan_translation_chunks
from openlrc.context import CharacterBrief, ContextTimeline, ProChunkContext, TranslationBrief, TranslationBriefInput
from openlrc.exceptions import ChatBotException
from openlrc.hymt2_pipeline import (
    ContextTimelineAgent,
    HyMT2RiskReviewAgent,
    TranslationBriefAgent,
    load_checkpoint,
    save_checkpoint,
    source_fingerprint,
)
from openlrc.models import ModelConfig, ModelProvider


def _mock_bot(outputs: list[str]):
    bot = MagicMock()
    bot.model_name = "general-model"
    bot.model_config = ModelConfig(provider=ModelProvider.OPENAI, name="general-model", context_window=32768)
    bot.api_fees = []
    bot.message.side_effect = [[MagicMock()] for _ in outputs]
    bot.get_content.side_effect = outputs
    return bot


class TestTranslationBriefAgent(unittest.TestCase):
    def test_complete_manual_brief_skips_model_and_uses_only_task_glossary(self):
        manual = TranslationBriefInput(
            summary="A user-authored detective story summary.",
            characters=[CharacterBrief(source_name="John", target_name="强尼")],
            tone_style="restrained noir",
        )
        bot = _mock_bot([])

        brief = TranslationBriefAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").build(
            ["John opened the case."], glossary={"case": "案件"}, translation_brief=manual
        )

        bot.message.assert_not_called()
        self.assertEqual(brief.summary, manual.summary)
        self.assertEqual(brief.characters, manual.characters)
        self.assertEqual(brief.tone_style, manual.tone_style)
        self.assertEqual([(item.source, item.target) for item in brief.glossary], [("case", "案件")])

    def test_partial_manual_brief_locks_supplied_sections(self):
        generated = {
            "summary": "An unstable generated summary.",
            "characters": [{"source_name": "John", "target_name": "约翰"}],
            "glossary": [{"source": "case", "target": "案件", "note": "investigation"}],
            "tone_style": "generated style",
        }
        bot = _mock_bot([json.dumps(generated, ensure_ascii=False)])
        manual = TranslationBriefInput(summary="The user's fixed document summary.", characters=[], tone_style=None)

        brief = TranslationBriefAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").build(
            ["John opened the case."], translation_brief=manual
        )

        self.assertEqual(brief.summary, manual.summary)
        self.assertEqual(brief.characters, [])
        self.assertEqual(brief.tone_style, "generated style")
        self.assertEqual(brief.glossary[0].source, "case")
        prompt = bot.message.call_args.args[0][1]["content"]
        self.assertIn("User-fixed Brief fields", prompt)
        self.assertIn("The user's fixed document summary.", prompt)

    def test_partial_manual_fields_remain_locked_after_hierarchical_merge(self):
        partial_one = {
            "summary": "drifted summary one",
            "characters": [{"source_name": "John", "target_name": "约翰"}],
            "glossary": [],
            "tone_style": "generated style one",
        }
        partial_two = {
            "summary": "drifted summary two",
            "characters": [{"source_name": "Jane", "target_name": "简"}],
            "glossary": [],
            "tone_style": "generated style two",
        }
        merged = {
            "summary": "drifted merged summary",
            "characters": [
                {"source_name": "John", "target_name": "约翰"},
                {"source_name": "Jane", "target_name": "简"},
            ],
            "glossary": [],
            "tone_style": "merged generated style",
        }
        bot = _mock_bot([json.dumps(item, ensure_ascii=False) for item in (partial_one, partial_two, merged)])
        manual = TranslationBriefInput(summary="The user's immutable document summary.", characters=[], tone_style=None)
        agent = TranslationBriefAgent(chatbot=bot, src_lang="en", target_lang="zh-cn")
        agent._split_texts = MagicMock(return_value=[[(1, "First section.")], [(2, "Second section.")]])

        brief = agent.build(["First section.", "Second section."], translation_brief=manual)

        self.assertEqual(bot.message.call_count, 3)
        self.assertEqual(brief.summary, manual.summary)
        self.assertEqual(brief.characters, [])
        self.assertEqual(brief.tone_style, "merged generated style")

    def test_empty_manual_input_preserves_automatic_source_fingerprint(self):
        kwargs = {
            "src_lang": "en",
            "target_lang": "zh-cn",
            "glossary": None,
            "mode": "normal",
            "context_model": "general-model",
            "translation_model": "hy-mt2",
        }

        self.assertEqual(
            source_fingerprint(["Hello"], **kwargs), source_fingerprint(["Hello"], translation_brief={}, **kwargs)
        )

    def test_manual_brief_fingerprint_is_content_stable(self):
        first = TranslationBriefInput(
            summary="Summary", characters=[{"source_name": "John", "target_name": "强尼"}], tone_style=""
        )
        same = TranslationBriefInput.model_validate(first.model_dump())
        changed = first.model_copy(update={"tone_style": "formal"})

        self.assertEqual(first.fingerprint(), same.fingerprint())
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())

    def test_builds_structured_brief_and_user_glossary_wins(self):
        payload = {
            "summary": "A detective investigates a case.",
            "characters": [{"source_name": "John", "target_name": "约翰"}],
            "glossary": [{"source": "case", "target": "案子", "note": ""}],
            "tone_style": "serious",
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])

        brief = TranslationBriefAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").build(
            ["John opened the case."], glossary={"case": "案件"}
        )

        self.assertEqual(brief.summary, payload["summary"])
        self.assertEqual(brief.glossary[0].target, "案件")
        self.assertEqual(brief.glossary[0].note, "")

    def test_brief_prompt_and_fields_use_source_semantics_with_bilingual_mappings(self):
        payload = {
            "summary": "侦探正在调查一宗复杂案件，并逐步确认嫌疑人的行动路线。",
            "characters": [{"source_name": "李明", "target_name": "Li Ming"}],
            "glossary": [{"source": "案发现场", "target": "crime scene", "note": "警方进行勘查的地点。"}],
            "tone_style": "严肃、简洁的犯罪悬疑语气。",
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])

        brief = TranslationBriefAgent(chatbot=bot, src_lang="zh-cn", target_lang="en").build(
            ["李明来到案发现场，开始调查这宗复杂案件。"]
        )

        self.assertEqual(brief.characters[0].target_name, "Li Ming")
        prompt = bot.message.call_args.args[0]
        self.assertIn("Write summary", prompt[1]["content"])
        self.assertIn("Only character target_name and glossary target", prompt[1]["content"])
        self.assertNotIn('"description"', prompt[1]["content"])
        self.assertNotIn('"target_audience"', prompt[1]["content"])
        self.assertNotIn('"asr_ambiguities"', prompt[1]["content"])
        self.assertNotIn("ASR", prompt[1]["content"])

    def test_rejects_clear_target_language_semantic_fields(self):
        payload = {
            "summary": "The detective investigates a complicated crime and follows the suspect through the city.",
            "characters": [],
            "glossary": [],
            "tone_style": "A serious and restrained crime thriller style for adult viewers.",
        }
        agent = TranslationBriefAgent(chatbot=_mock_bot([]), src_lang="zh-cn", target_lang="en")

        self.assertFalse(agent._check("", json.dumps(payload)))

    def test_short_names_and_unchanged_terms_are_not_language_rejected(self):
        payload = {
            "summary": "会议将讨论新的人工智能系统，以及后续的测试和部署安排。",
            "characters": [{"source_name": "AI", "target_name": "AI"}],
            "glossary": [{"source": "OpenAI", "target": "OpenAI", "note": "文中提到的机构名称。"}],
            "tone_style": "正式、清晰。",
        }
        agent = TranslationBriefAgent(chatbot=_mock_bot([]), src_lang="zh-cn", target_lang="en")

        self.assertTrue(agent._check("", json.dumps(payload, ensure_ascii=False)))

    def test_rejects_extra_fields_and_empty_mappings(self):
        base = {
            "summary": "A sufficiently detailed English summary of the meeting and its decisions.",
            "characters": [],
            "glossary": [],
            "tone_style": "formal",
        }
        agent = TranslationBriefAgent(chatbot=_mock_bot([]), src_lang="en", target_lang="zh-cn")
        extra = {**base, "context_language": "en"}
        removed_fields = {**base, "target_audience": "general", "asr_ambiguities": []}
        described_character = {
            **base,
            "characters": [{"source_name": "John", "target_name": "约翰", "description": "detective"}],
        }
        empty_mapping = {**base, "glossary": [{"source": "meeting", "target": "", "note": ""}]}

        self.assertFalse(agent._check("", json.dumps(extra)))
        self.assertFalse(agent._check("", json.dumps(removed_fields)))
        self.assertFalse(agent._check("", json.dumps(described_character)))
        self.assertFalse(agent._check("", json.dumps(empty_mapping)))

    def test_rejects_empty_user_glossary_target_after_merge(self):
        payload = {
            "summary": "A sufficiently detailed English summary of the meeting and its decisions.",
            "characters": [],
            "glossary": [],
            "tone_style": "formal",
        }
        agent = TranslationBriefAgent(chatbot=_mock_bot([json.dumps(payload)]), src_lang="en", target_lang="zh-cn")

        with self.assertRaisesRegex(ChatBotException, "field contract"):
            agent.build(["The meeting begins."], glossary={"meeting": ""})

    def test_source_fingerprint_changes_with_mode(self):
        common = dict(
            texts=["Hello"],
            src_lang="en",
            target_lang="zh-cn",
            glossary=None,
            context_model="openai:gpt",
            translation_model="local_llama:hy-mt2",
        )
        fast = source_fingerprint(mode="fast", **common)
        contextual = source_fingerprint(mode="context", **common)
        self.assertNotEqual(fast, contextual)

    def test_source_fingerprint_changes_with_manual_brief_content(self):
        common = dict(
            texts=["Hello"],
            src_lang="en",
            target_lang="zh-cn",
            glossary=None,
            mode="normal",
            context_model="",
            translation_model="local_llama:hy-mt2",
        )
        first = source_fingerprint(
            translation_brief=TranslationBriefInput(summary="First summary", characters=[], tone_style=""), **common
        )
        second = source_fingerprint(
            translation_brief=TranslationBriefInput(summary="Second summary", characters=[], tone_style=""), **common
        )

        self.assertNotEqual(first, second)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            save_checkpoint(path, {"compare": [], "schema_version": 2})
            self.assertEqual(load_checkpoint(path)["schema_version"], 2)


class TestContextTimelineAgent(unittest.TestCase):
    def test_builds_bounded_timeline_and_checkpoints_each_chunk(self):
        plans = plan_translation_chunks(["one", "two", "three", "four"], chunk_size=2, token_budget=10000)
        signature = chunk_plan_signature(plans)
        bot = _mock_bot(
            [
                json.dumps(
                    {"chunk_id": 1, "segment_ids": [1, 2], "story_so_far": "Story one", "current_scene": "Scene one"}
                ),
                json.dumps(
                    {"chunk_id": 2, "segment_ids": [3, 4], "story_so_far": "Story two", "current_scene": "Scene two"}
                ),
            ]
        )
        agent = ContextTimelineAgent(chatbot=bot, src_lang="en")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            checkpoint = {"compare": [], "schema_version": 3}
            timeline = agent.build(
                plans,
                texts=["one", "two", "three", "four"],
                brief=TranslationBrief(
                    summary="A story",
                    characters=[{"source_name": "John", "target_name": "约翰"}],
                    glossary=[{"source": "case", "target": "案件", "note": "An investigation"}],
                ),
                chunk_signature=signature,
                checkpoint_path=path,
                checkpoint=checkpoint,
            )
            saved = load_checkpoint(path)

        self.assertEqual([item.chunk_id for item in timeline.chunks], [1, 2])
        self.assertEqual(saved["timeline_completed_chunks"], [1, 2])
        self.assertEqual(saved["context_timeline"]["chunk_signature"], signature)
        self.assertEqual(saved["timeline_prompt_version"], 3)
        second_prompt = json.loads(bot.message.call_args_list[1].args[0][1]["content"])
        self.assertEqual(second_prompt["previous_story_so_far"], "Story one")
        self.assertNotIn("target_language", second_prompt)
        self.assertNotIn("output_language", second_prompt)
        self.assertIn("source_semantic_brief", second_prompt)
        self.assertNotIn("target_name", second_prompt["source_semantic_brief"]["characters"][0])
        self.assertNotIn("description", second_prompt["source_semantic_brief"]["characters"][0])
        self.assertNotIn("target", second_prompt["source_semantic_brief"]["glossary"][0])
        self.assertNotIn("target_audience", second_prompt["source_semantic_brief"])
        self.assertNotIn("asr_ambiguities", second_prompt["source_semantic_brief"])

    def test_resume_reuses_prefix_and_only_generates_missing_chunks(self):
        texts = ["one", "two", "three", "four"]
        plans = plan_translation_chunks(texts, chunk_size=2, token_budget=10000)
        signature = chunk_plan_signature(plans)
        checkpoint = {
            "compare": [],
            "schema_version": 3,
            "context_timeline": {
                "schema_version": 1,
                "chunk_signature": signature,
                "chunks": [
                    {
                        "chunk_id": 1,
                        "segment_ids": [1, 2],
                        "story_so_far": "Restored story",
                        "current_scene": "Restored scene",
                    }
                ],
            },
        }
        bot = _mock_bot(
            [
                json.dumps(
                    {
                        "chunk_id": 2,
                        "segment_ids": [3, 4],
                        "story_so_far": "Finished story",
                        "current_scene": "Finished scene",
                    }
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            timeline = ContextTimelineAgent(chatbot=bot, src_lang="en").build(
                plans,
                texts=texts,
                brief=TranslationBrief(summary="A story"),
                chunk_signature=signature,
                checkpoint_path=Path(tmpdir) / "checkpoint.json",
                checkpoint=checkpoint,
            )

        self.assertEqual(bot.message.call_count, 1)
        self.assertEqual(timeline.chunks[0].story_so_far, "Restored story")
        prompt = json.loads(bot.message.call_args.args[0][1]["content"])
        self.assertEqual(prompt["previous_story_so_far"], "Restored story")

    def test_rejects_extra_fields(self):
        content = json.dumps(
            {"chunk_id": 1, "segment_ids": [1], "story_so_far": "story", "current_scene": "scene", "confidence": 0.9}
        )

        self.assertFalse(ContextTimelineAgent._check("", content))

    def test_context_timeline_rejects_wrong_payload_schema_version(self):
        with self.assertRaises(ValueError):
            ContextTimeline(schema_version=999, chunk_signature="signature", chunks=[])

    def test_rejects_mismatched_chunk_or_segment_ids(self):
        checker = ContextTimelineAgent._checker(2, [3, 4])

        self.assertFalse(
            checker(
                "",
                json.dumps({"chunk_id": 2, "segment_ids": [4, 3], "story_so_far": "story", "current_scene": "scene"}),
            )
        )

    def test_rejects_context_over_token_limit(self):
        content = json.dumps(
            {"chunk_id": 1, "segment_ids": [1], "story_so_far": "word " * 1000, "current_scene": "scene"}
        )

        self.assertFalse(ContextTimelineAgent._check("", content))

    def test_rejects_clear_non_source_language_context(self):
        agent = ContextTimelineAgent(chatbot=_mock_bot([]), src_lang="zh-cn")
        checker = agent._checker(1, [1], agent.language_validator)
        content = json.dumps(
            {
                "chunk_id": 1,
                "segment_ids": [1],
                "story_so_far": "The detective investigates the crime and follows the suspect across the city center.",
                "current_scene": "The detective is questioning a witness inside the police station.",
            }
        )

        self.assertFalse(checker("", content))

    def test_hard_scene_break_drops_previous_scene_from_prompt(self):
        texts = ["first scene", "second scene"]
        plans = plan_translation_chunks(texts, timestamps=[(0.0, 1.0), (40.0, 41.0)], chunk_size=30, token_budget=10000)
        signature = chunk_plan_signature(plans, timestamps=[(0.0, 1.0), (40.0, 41.0)])
        bot = _mock_bot(
            [
                json.dumps({"chunk_id": 1, "segment_ids": [1], "story_so_far": "story", "current_scene": "old scene"}),
                json.dumps(
                    {"chunk_id": 2, "segment_ids": [2], "story_so_far": "story 2", "current_scene": "new scene"}
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ContextTimelineAgent(chatbot=bot, src_lang="en").build(
                plans,
                texts=texts,
                brief=TranslationBrief(summary="Two scenes"),
                chunk_signature=signature,
                checkpoint_path=Path(tmpdir) / "checkpoint.json",
                checkpoint={"compare": [], "schema_version": 3},
            )

        second_prompt = json.loads(bot.message.call_args_list[1].args[0][1]["content"])
        self.assertTrue(second_prompt["hard_scene_break_before"])
        self.assertEqual(second_prompt["previous_story_so_far"], "story")
        self.assertEqual(second_prompt["previous_current_scene"], "")


class TestHyMT2RiskReviewAgent(unittest.TestCase):
    def test_scans_every_id_and_returns_high_risk_revision(self):
        payload = {
            "items": [
                {"id": 1, "risk": "low", "issues": [], "revised_translation": None},
                {"id": 2, "risk": "high", "issues": ["meaning error"], "revised_translation": "修订译文"},
            ]
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])
        brief = TranslationBrief(summary="A greeting")

        result = HyMT2RiskReviewAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").review(
            [(1, "Hello"), (2, "Goodbye")], {1: "你好", 2: "你好"}, brief=brief
        )

        self.assertEqual([item.id for item in result.items], [1, 2])
        self.assertEqual(result.items[1].risk, "high")
        self.assertEqual(result.items[1].revised_translation, "修订译文")
        prompt = json.loads(bot.message.call_args.args[0][1]["content"])
        self.assertEqual(set(prompt["translation_brief"]), {"summary", "characters", "glossary", "tone_style"})

    def test_rejects_blank_multiline_and_protocol_revisions(self):
        checker = HyMT2RiskReviewAgent._checker([1])

        for revision in (
            "   ",
            "first line\nsecond line",
            '<seg id="1">污染</seg>',
            '```json\n{"items": []}\n```',
            '{"items": []}',
            "story_so_far: leaked context",
        ):
            with self.subTest(revision=revision):
                payload = {
                    "items": [{"id": 1, "risk": "high", "issues": ["meaning error"], "revised_translation": revision}]
                }
                self.assertFalse(checker("", json.dumps(payload, ensure_ascii=False)))

    def test_review_strips_valid_revision(self):
        payload = {
            "items": [{"id": 1, "risk": "high", "issues": ["meaning error"], "revised_translation": "  修订译文  "}]
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])

        result = HyMT2RiskReviewAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").review(
            [(1, "Hello")], {1: "你好"}, brief=TranslationBrief(summary="A greeting")
        )

        self.assertEqual(result.items[0].revised_translation, "修订译文")

    def test_pro_review_receives_matching_timeline_context(self):
        payload = {"items": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]}
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])
        chunk_context = ProChunkContext(
            chunk_id=1,
            segment_ids=[1],
            story_so_far="A suspect escaped.",
            current_scene="The detective is at a station.",
        )

        HyMT2RiskReviewAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").review(
            [(1, "Where did she go?")],
            {1: "她去哪儿了？"},
            brief=TranslationBrief(summary="A chase"),
            chunk_context=chunk_context,
        )

        prompt = json.loads(bot.message.call_args.args[0][1]["content"])
        self.assertEqual(prompt["context_timeline_chunk"]["chunk_id"], 1)
        self.assertEqual(prompt["context_timeline_chunk"]["current_scene"], "The detective is at a station.")

    def test_review_patches_converts_only_high_risk_revisions(self):
        payload = {
            "items": [
                {"id": 1, "risk": "medium", "issues": ["awkward"], "revised_translation": None},
                {"id": 2, "risk": "high", "issues": ["meaning"], "revised_translation": "修订"},
            ]
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])

        issues, patches, metadata = HyMT2RiskReviewAgent(
            chatbot=bot, src_lang="en", target_lang="zh-cn"
        ).review_patches(
            [(1, "Hello"), (2, "Goodbye")], {1: "你好", 2: "你好"}, brief=TranslationBrief(summary="summary")
        )

        self.assertEqual([issue.severity.value for issue in issues], ["warning", "error"])
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].segment_id, 2)
        self.assertEqual(patches[0].before, "你好")
        self.assertEqual(metadata["review_protocol_version"], 2)
