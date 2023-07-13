#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest

import openai

from openlrc.translate import GPTTranslator
from openlrc.utils import get_similarity


# TODO: Need more check
class TestGPTTranslator(unittest.TestCase):
    #  Tests that a single chunk of text is translated correctly
    def test_single_chunk_translation(self):
        text = 'Hello, how are you?'
        translator = GPTTranslator()
        translation = translator.translate(text, 'en', 'es')[0]

        assert get_similarity(translation, 'Hola, ¿cómo estás?') > 0.618

    #  Tests that multiple chunks of text are translated correctly
    def test_multiple_chunk_translation(self):
        texts = ['Hello, how are you?', 'I am fine, thank you.']
        translator = GPTTranslator()
        translations = translator.translate(texts, 'en', 'es')
        assert get_similarity(translations[0], 'Hola, ¿cómo estás?') > 0.618
        assert get_similarity(translations[1], 'Estoy bien, gracias.') > 0.618

    #  Tests that text with different source and target languages is translated correctly
    def test_different_language_translation(self):
        text = 'Hello, how are you?'
        translator = GPTTranslator()
        try:
            translation = translator.translate(text, 'en', 'fr')[0]
            assert get_similarity(translation, 'Bonjour, comment ça va?') > 0.618
        except openai.error.OpenAIError:
            pass

    #  Tests that an empty list of texts is translated correctly
    def test_empty_text_list_translation(self):
        texts = []
        translator = GPTTranslator()
        translations = translator.translate(texts, 'en', 'es')
        assert translations == []
