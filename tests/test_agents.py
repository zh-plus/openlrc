#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import os
import unittest
from typing import List
from unittest.mock import patch, MagicMock

from pydantic import BaseModel

from openlrc.agents import ChunkedTranslatorAgent, TranslationContext, ContextReviewerAgent
from openlrc.context import TranslateInfo
from openlrc.prompter import ChunkedTranslatePrompter

LIVE_API = os.environ.get('OPENLRC_TEST_LIVE_API', '').lower() in ('1', 'true', 'yes')


class DummyMessage(BaseModel):
    content: str


class DummyChoice(BaseModel):
    message: DummyMessage


class DummyResponse(BaseModel):
    choices: List[DummyChoice]


class TestTranslatorAgent(unittest.TestCase):

    @patch('openlrc.chatbot.GPTBot.message',
           MagicMock(return_value=[
               DummyResponse(
                   choices=[
                       DummyChoice(
                           message=DummyMessage(
                               content='<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n')
                       )]
               )
           ]))
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-dummy'})
    def test_translate_chunk_success(self):
        agent = ChunkedTranslatorAgent(
            src_lang='en', target_lang='fr', info=TranslateInfo(
                title='Example Title', audio_type='Book',
                glossary={'hello': 'bonjour'}
            )
        )
        agent.chatbot.api_fees = [0.00035]
        translations, context = agent.translate_chunk(
            chunk_id=1, chunk=[(1, 'Hello, how are you?'), (2, 'I am fine, thank you.')],
            context=TranslationContext(
                summary='Example Summary',
                previous_summaries=['s1', 's2'],
                scene='Example Scene'
            )
        )

        self.assertListEqual(translations, ['Bonjour, comment ça va?', 'Je vais bien, merci.'])
        self.assertEqual(context.summary, 'Example Summary')
        self.assertEqual(context.scene, 'Example Scene')

    #  Handle invalid chatbot model names gracefully
    def test_invalid_chatbot_model(self):
        with self.assertRaises(ValueError):
            ChunkedTranslatorAgent(src_lang='en', target_lang='fr', info=TranslateInfo(), chatbot_model='invalid-model')

    @patch('openlrc.chatbot.GPTBot.get_content',
           MagicMock(
               return_value='<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n'))
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-dummy'})
    def test_parse_response_success(self):
        agent = ChunkedTranslatorAgent(src_lang='en', target_lang='fr')
        translations, summary, scene = agent._parse_responses('dummy_response')

        self.assertListEqual(translations, ['Bonjour, comment ça va?', 'Je vais bien, merci.'])
        self.assertEqual(summary, 'Example Summary')
        self.assertEqual(scene, 'Example Scene')

    #  Properly format texts for translation
    def test_format_texts_success(self):
        texts = [(1, 'Hello, how are you?'), (2, 'I am fine, thank you.')]
        formatted_text = ChunkedTranslatePrompter.format_texts(texts)

        expected_output = '#1\nOriginal>\nHello, how are you?\nTranslation>\n\n#2\nOriginal>\nI am fine, thank you.\nTranslation>\n'
        self.assertEqual(formatted_text, expected_output)

    #  Use glossary terms in translations when provided
    def test_use_glossary_terms_success(self):
        glossary = {'hello': 'bonjour', 'how are you': 'comment ça va'}
        prompter = ChunkedTranslatePrompter(src_lang='en', target_lang='fr', context=TranslateInfo(glossary=glossary))

        formatted_glossary = prompter.formatted_glossary

        expected_output = '\n# Glossary\nUse the following glossary to ensure consistency in your translations:\n<preferred-translation>\nhello: bonjour\nhow are you: comment ça va\n</preferred-translation>\n'
        self.assertEqual(formatted_glossary, expected_output)


@unittest.skipUnless(LIVE_API, 'Requires OPENLRC_TEST_LIVE_API=1 and valid API keys')
class TestContextReviewerAgent(unittest.TestCase):
    def test_generates_valid_context(self):
        texts = ["John and Sarah discuss their plan to locate a suspect",
                 "John: 'As a 10 years experienced detector, my advice is we should start our search in the uptown area.'",
                 "Sarah: 'Agreed. Let's gather more information before we move.'",
                 "Then, they prepare to start their investigation."]
        title = "The Detectors"
        glossary = {"suspect": "嫌疑人", "uptown": "市中心"}

        agent = ContextReviewerAgent('en', 'zh')
        context = agent.build_context(texts, title, glossary)

        self.assertIsNotNone(context)
        self.assertIsInstance(context, str)
        self.assertIn("Glossary", context)
        self.assertIn("Characters", context)
        self.assertIn("Summary", context)
        self.assertIn("Tone and Style", context)
        self.assertIn("Target Audience", context)
