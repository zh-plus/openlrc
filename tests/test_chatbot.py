#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import unittest
import os
from typing import Union

from pydantic import BaseModel

from openlrc.chatbot import GPTBot, ClaudeBot, route_chatbot

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_MODELS = {
    'gpt': 'openai/gpt-5-nano',
    'claude': 'anthropic/claude-haiku-4.5',
    'gemini': 'google/gemini-2.5-flash-lite',
}


class Usage(BaseModel):
    pass


class OpenAIUsage(Usage):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AnthropicUsage(Usage):
    input_tokens: int
    output_tokens: int


class OpenAIResponse(BaseModel):
    usage: Union[Usage]


class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.gpt_bot = GPTBot(
            model_name=OPENROUTER_MODELS['gpt'],
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            base_url_config={'openai': OPENROUTER_BASE_URL},
            api_key=OPENROUTER_API_KEY or 'test-key'
        )
        self.claude_bot = ClaudeBot(
            model_name='claude-3-5-sonnet-20241022',
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            api_key='test-key'
        )

    def _make_openrouter_bot(self, model_name: str):
        return GPTBot(
            model_name=model_name,
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            base_url_config={'openai': OPENROUTER_BASE_URL},
            api_key=OPENROUTER_API_KEY
        )

    def test_estimate_fee(self):
        bot = self.gpt_bot
        messages = [
            {'role': 'system', 'content': 'You are gpt.'},
            {'role': 'user', 'content': 'Hello'},
        ]
        fee = bot.estimate_fee(messages)
        self.assertIsNotNone(fee)

    def test_gpt_update_fee(self):
        bot = self.gpt_bot
        bot.api_fees += [0]
        response1 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300))
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=200, completion_tokens=400, total_tokens=600))
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=300, completion_tokens=600, total_tokens=900))
        bot.update_fee(response3)
        self.assertIsNotNone(bot.api_fees)

    def test_claude_update_fee(self):
        bot = self.claude_bot
        bot.api_fees += [0]
        response1 = OpenAIResponse(usage=AnthropicUsage(input_tokens=100, output_tokens=200))
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = OpenAIResponse(usage=AnthropicUsage(input_tokens=200, output_tokens=400))
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = OpenAIResponse(usage=AnthropicUsage(input_tokens=300, output_tokens=600))
        bot.update_fee(response3)

        self.assertIsNotNone(bot.api_fees)

    def test_gpt_message_async(self):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest('OPENROUTER_API_KEY is required for LLM integration tests.')
        bot = self._make_openrouter_bot(OPENROUTER_MODELS['gpt'])
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
        ]
        results = bot.message(messages_list)

        self.assertTrue(all(['hello' in bot.get_content(r).lower() for r in results]))

    def test_claude_message_async(self):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest('OPENROUTER_API_KEY is required for LLM integration tests.')
        bot = self._make_openrouter_bot(OPENROUTER_MODELS['claude'])
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
        ]
        results = bot.message(messages_list)

        self.assertTrue(all(['hello' in bot.get_content(r).lower() for r in results]))

    def test_gpt_message_seq(self):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest('OPENROUTER_API_KEY is required for LLM integration tests.')
        bot = self._make_openrouter_bot(OPENROUTER_MODELS['gpt'])
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ]
        ]
        results = bot.message(messages_list)

        self.assertIn('hello', bot.get_content(results[0]).lower())

    def test_claude_message_seq(self):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest('OPENROUTER_API_KEY is required for LLM integration tests.')
        bot = self._make_openrouter_bot(OPENROUTER_MODELS['claude'])
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ]
        ]
        results = bot.message(messages_list)
        assert 'hello' in bot.get_content(results[0]).lower()

        self.assertIn('hello', bot.get_content(results[0]).lower())

    def test_route_chatbot(self):
        chatbot_model1 = 'openai: claude-3-5-haiku-20241022'
        chabot_cls1, model_name1 = route_chatbot(chatbot_model1)
        self.assertEqual(chabot_cls1, GPTBot)
        try:
            _ = chabot_cls1(
                model_name=model_name1,
                temperature=1,
                top_p=1,
                retry=8,
                max_async=16,
                api_key='test-key'
            )
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

        chatbot_model2 = 'anthropic: gpt-3.5-turbo'
        chabot_cls2, model_name2 = route_chatbot(chatbot_model2)
        self.assertEqual(chabot_cls2, ClaudeBot)
        try:
            _ = chabot_cls2(
                model_name=model_name2,
                temperature=1,
                top_p=1,
                retry=8,
                max_async=16,
                api_key='test-key'
            )
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

    def test_route_chatbot_undefined(self):
        chatbot_model = 'openai: invalid_model_name'
        model_cls, model_name = route_chatbot(chatbot_model)
        self.assertEqual(model_cls, GPTBot)
        self.assertEqual(model_name, chatbot_model.split(':')[-1].strip())

    def test_temperature_clamp(self):
        chatbot1 = GPTBot(temperature=10, top_p=1, retry=8, max_async=16, api_key='test-key')
        chatbot2 = GPTBot(temperature=-1, top_p=1, retry=8, max_async=16, api_key='test-key')
        chatbot3 = ClaudeBot(temperature=2, top_p=1, retry=8, max_async=16, api_key='test-key')
        chatbot4 = ClaudeBot(temperature=-1, top_p=1, retry=8, max_async=16, api_key='test-key')

        self.assertEqual(chatbot1.temperature, 2)
        self.assertEqual(chatbot2.temperature, 0)
        self.assertEqual(chatbot3.temperature, 1)
        self.assertEqual(chatbot4.temperature, 0)


class TestThirdPartyBot(unittest.TestCase):
    def test_beta_base_url(self):
        bot = GPTBot(model_name='deepseek-chat', temperature=1, top_p=1, retry=8, max_async=16,
                     base_url_config={'openai': 'https://api.deepseek.com/beta'},
                     api_key='test-key')
        self.assertTrue(bot.model_info.beta)

    def test_non_beta_base_url(self):
        bot = GPTBot(model_name='deepseek-chat', temperature=1, top_p=1, retry=8, max_async=16,
                     base_url_config={'openai': 'https://api.deepseek.com'},
                     api_key='test-key')
        self.assertFalse(bot.model_info.beta)


# TODO: Retry_bot testing

class TestGeminiBot(unittest.TestCase):
    # def setUp(self):
    #     import os
    #     os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    #     os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    #
    # def tearDown(self):
    #     import os
    #     os.environ.pop('HTTP_PROXY')
    #     os.environ.pop('HTTPS_PROXY')

    def test_multi_turn(self):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest('OPENROUTER_API_KEY is required for LLM integration tests.')
        bot = GPTBot(
            model_name=OPENROUTER_MODELS['gemini'],
            base_url_config={'openai': OPENROUTER_BASE_URL},
            api_key=OPENROUTER_API_KEY
        )
        result = bot.message([
            {'role': 'system', 'content': 'You are a echo machine, echo each word from input.'},
            {'role': 'user', 'content': 'How are you?'},
            {'role': 'assistant', 'content': 'How are you?'},
            {'role': 'user', 'content': 'THen?'}
        ])[0]
        self.assertIsNotNone(bot.get_content(result))
