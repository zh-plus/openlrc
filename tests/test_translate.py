#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

import openai

from openlrc.translate import GPTTranslator
from openlrc.utils import get_similarity


class TestGPTTranslator(unittest.TestCase):
    def tearDown(self) -> None:
        compare_path = Path('translate_intermediate.json')
        compare_path.unlink(missing_ok=True)

    def test_single_chunk_translation(self):
        text = 'Hello, how are you?'
        translator = GPTTranslator()
        translation = translator.translate(text, 'en', 'es')[0]

        assert get_similarity(translation, 'Hola, ¿cómo estás?') > 0.618

    def test_multiple_chunk_translation(self):
        texts = ['Hello, how are you?', 'I am fine, thank you.']
        translator = GPTTranslator()
        translations = translator.translate(texts, 'en', 'es')
        assert get_similarity(translations[0], 'Hola, ¿cómo estás?') > 0.618
        assert get_similarity(translations[1], 'Estoy bien, gracias.') > 0.618

    def test_different_language_translation(self):
        text = 'Hello, how are you?'
        translator = GPTTranslator()
        try:
            translation = translator.translate(text, 'en', 'fr')[0]
            assert get_similarity(translation, 'Bonjour, comment ça va?') > 0.618
        except openai.OpenAIError:
            pass

    def test_empty_text_list_translation(self):
        texts = []
        translator = GPTTranslator()
        translations = translator.translate(texts, 'en', 'es')
        assert translations == []

# Not integrated by the openlrc main function because of performance
#
# class TestDeepLTranslator(unittest.TestCase):
#     def test_single_chunk_translation(self):
#         text = 'Hello, how are you?'
#         translator = DeepLTranslator()
#         translation = translator.translate(text, 'en', 'es')[0]
#
#         assert get_similarity(translation, 'Hola, ¿cómo estás?') > 0.618
#
#     def test_multiple_chunk_translation(self):
#         texts = ['Hello, how are you?', 'I am fine, thank you.']
#         translator = DeepLTranslator()
#         translations = translator.translate(texts, 'en', 'es')
#         assert get_similarity(translations[0], 'Hola, ¿cómo estás?') > 0.618
#         assert get_similarity(translations[1], 'Estoy bien, gracias.') > 0.618
