#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from openlrc.context import TranslationBrief
from openlrc.hymt2_pipeline import (
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
    def test_builds_structured_brief_and_user_glossary_wins(self):
        payload = {
            "summary": "A detective investigates a case.",
            "characters": [{"source_name": "John", "target_name": "约翰", "description": "detective"}],
            "glossary": [{"source": "case", "target": "案子", "note": ""}],
            "tone_style": "serious",
            "target_audience": "adult viewers",
            "asr_ambiguities": [],
        }
        bot = _mock_bot([json.dumps(payload, ensure_ascii=False)])

        brief = TranslationBriefAgent(chatbot=bot, src_lang="en", target_lang="zh-cn").build(
            ["John opened the case."], glossary={"case": "案件"}
        )

        self.assertEqual(brief.summary, payload["summary"])
        self.assertEqual(brief.glossary[0].target, "案件")

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

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            save_checkpoint(path, {"compare": [], "schema_version": 2})
            self.assertEqual(load_checkpoint(path)["schema_version"], 2)


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
