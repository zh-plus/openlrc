#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

from openlrc.agents import create_chatbot
from openlrc.translate import LLMTranslator
from tests.conftest import LIVE_API, TEST_LLM_API_KEY, TEST_MODELS
from tests.live_api_contract import assert_translation_contract


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestLLMTranslator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")

    def tearDown(self) -> None:
        compare_path = Path("translate_intermediate.json")
        compare_path.unlink(missing_ok=True)

    def test_single_chunk_translation(self):
        for chatbot_model in TEST_MODELS.values():
            text = "John paid 12 dollars on July 5."
            chatbot = create_chatbot(chatbot_model)
            try:
                translator = LLMTranslator(chatbot=chatbot)
                translation = translator.translate(text, "en", "es")[0]
            finally:
                chatbot.close()

            assert_translation_contract(text, translation, "es", preserved_terms=("John", "12", "5"))

    def test_multiple_chunk_translation(self):
        for chatbot_model in TEST_MODELS.values():
            texts = ["Hello, how are you?", "I am fine, thank you."]
            chatbot = create_chatbot(chatbot_model)
            try:
                translator = LLMTranslator(chatbot=chatbot)
                translations = translator.translate(texts, "en", "es")
            finally:
                chatbot.close()
            self.assertEqual(len(translations), len(texts))
            for source, translation in zip(texts, translations, strict=True):
                assert_translation_contract(source, translation, "es")

    def test_different_language_translation(self):
        for chatbot_model in TEST_MODELS.values():
            text = "Hello, how are you?"
            chatbot = create_chatbot(chatbot_model)
            try:
                translator = LLMTranslator(chatbot=chatbot)
                translation = translator.translate(text, "en", "ja")[0]
            finally:
                chatbot.close()
            assert_translation_contract(text, translation, "ja")

    def test_empty_text_list_translation(self):
        for chatbot_model in TEST_MODELS.values():
            texts = []
            chatbot = create_chatbot(chatbot_model)
            try:
                translator = LLMTranslator(chatbot=chatbot)
                translations = translator.translate(texts, "en", "es")
            finally:
                chatbot.close()
            self.assertEqual(translations, [])

    def test_atomic_translate(self):
        for chatbot_model in TEST_MODELS.values():
            texts = ["Hello, how are you?", "I am fine, thank you."]
            chatbot = create_chatbot(chatbot_model)
            try:
                translator = LLMTranslator(chatbot=chatbot)
                translations = translator.atomic_translate(chatbot, texts, "en", "zh")
            finally:
                chatbot.close()
            self.assertEqual(len(translations), len(texts))
            for source, translation in zip(texts, translations, strict=True):
                assert_translation_contract(source, translation, "zh")
