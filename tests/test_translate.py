#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

import anthropic
import openai

from openlrc.translate import LLMTranslator
from openlrc.utils import get_similarity

test_models = ['gpt-4o-mini', 'claude-3-haiku-20240307']


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
