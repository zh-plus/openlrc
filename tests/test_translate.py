#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import os
import unittest
from pathlib import Path

import anthropic
import openai

from openlrc.models import ModelConfig, ModelProvider
from openlrc.translate import LLMTranslator
from openlrc.utils import get_similarity

LIVE_API = os.environ.get('OPENLRC_TEST_LIVE_API', '').lower() in ('1', 'true', 'yes')

test_model_config = ModelConfig(provider=ModelProvider.OPENAI, name='gpt-4.1-nano')
test_models = ['claude-3-5-haiku-latest', test_model_config]


@unittest.skipUnless(LIVE_API, 'Requires OPENLRC_TEST_LIVE_API=1 and valid API keys')
class TestLLMTranslator(unittest.TestCase):

    def tearDown(self) -> None:
        compare_path = Path('translate_intermediate.json')
        compare_path.unlink(missing_ok=True)

    def test_single_chunk_translation(self):
        for chatbot_model in test_models:
            text = 'Hello, how are you?'
            translator = LLMTranslator(chatbot_model)
            translation = translator.translate(text, 'en', 'es')[0]

            self.assertGreater(get_similarity(translation, 'Hola, ¿cómo estás?'), 0.5)

    def test_multiple_chunk_translation(self):
        for chatbot_model in test_models:
            texts = ['Hello, how are you?', 'I am fine, thank you.']
            translator = LLMTranslator(chatbot_model)
            translations = translator.translate(texts, 'en', 'es')
            self.assertGreater(get_similarity(translations[0], 'Hola, ¿cómo estás?'), 0.5)
            self.assertGreater(get_similarity(translations[1], 'Estoy bien, gracias.'), 0.5)

    def test_different_language_translation(self):
        for chatbot_model in test_models:
            text = 'Hello, how are you?'
            translator = LLMTranslator(chatbot_model)
            try:
                translation = translator.translate(text, 'en', 'ja')[0]
                self.assertTrue(
                    get_similarity(translation, 'こんにちは、お元気ですか？') > 0.5 or
                    get_similarity(translation, 'こんにちは、調子はどうですか?') > 0.5
                )
            except (openai.OpenAIError, anthropic.APIError):
                pass

    def test_empty_text_list_translation(self):
        for chatbot_model in test_models:
            texts = []
            translator = LLMTranslator(chatbot_model)
            translations = translator.translate(texts, 'en', 'es')
            self.assertEqual(translations, [])

    def test_atomic_translate(self):
        for chatbot_model in test_models:
            texts = ['Hello, how are you?', 'I am fine, thank you.']
            translator = LLMTranslator(chatbot_model)
            translations = translator.atomic_translate(chatbot_model, texts, 'en', 'zh')
            self.assertGreater(get_similarity(translations[0], '你好，你好吗？'), 0.5)
            self.assertGreater(get_similarity(translations[1], '我很好，谢谢。'), 0.5)

# Not integrated by the openlrc main function because of performance
#
# class TestDeepLTranslator(unittest.TestCase):
#     def test_single_chunk_translation(self):
#         text = 'Hello, how are you?'
#         translator = DeepLTranslator()
#         translation = translator.translate(text, 'en', 'es')[0]
#
#         assert get_similarity(translation, 'Hola, ¿cómo estás?') > 0.5
#
#     def test_multiple_chunk_translation(self):
#         texts = ['Hello, how are you?', 'I am fine, thank you.']
#         translator = DeepLTranslator()
#         translations = translator.translate(texts, 'en', 'es')
#         assert get_similarity(translations[0], 'Hola, ¿cómo estás?') > 0.5
#         assert get_similarity(translations[1], 'Estoy bien, gracias.') > 0.5
