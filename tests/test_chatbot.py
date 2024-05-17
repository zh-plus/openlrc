#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import unittest
from math import isclose
from typing import Union

from pydantic import BaseModel

from openlrc.chatbot import GPTBot, ClaudeBot, route_chatbot


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


class TestGPTBot(unittest.TestCase):
    def setUp(self):
        self.gpt_bot = GPTBot(temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.05)
        self.claude_bot = ClaudeBot(temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.05)

    def test_estimate_fee(self):
        bot = self.gpt_bot
        messages = [
            {'role': 'system', 'content': 'You are gpt.'},
            {'role': 'user', 'content': 'Hello'},
        ]
        fee = bot.estimate_fee(messages)
        self.assertTrue(isclose(fee, 6e-06))

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
        self.assertListEqual(bot.api_fees, [0.00035, 0.0007, 0.00105])

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

        self.assertListEqual(bot.api_fees, [0.0033, 0.0066, 0.0099])

    def test_gpt_message_async(self):
        bot = self.gpt_bot
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
        bot = self.claude_bot
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
        bot = self.gpt_bot
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ]
        ]
        results = bot.message(messages_list)

        self.assertIn('hello', bot.get_content(results[0]).lower())

    def test_claude_message_seq(self):
        bot = self.claude_bot
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ]
        ]
        results = bot.message(messages_list)
        assert 'hello' in bot.get_content(results[0]).lower()

        self.assertIn('hello', bot.get_content(results[0]).lower())

    def test_route_chatbot(self):
        chatbot_model1 = 'openai: claude-3-haiku-20240307'
        chabot_cls1, model_name1 = route_chatbot(chatbot_model1)
        self.assertEqual(chabot_cls1, GPTBot)
        try:
            _ = chabot_cls1(model=model_name1, temperature=1, top_p=1, retry=8, max_async=16)
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

        chatbot_model2 = 'anthropic: gpt-3.5-turbo'
        chabot_cls2, model_name2 = route_chatbot(chatbot_model2)
        self.assertEqual(chabot_cls2, ClaudeBot)
        try:
            _ = chabot_cls2(model=model_name2, temperature=1, top_p=1, retry=8, max_async=16)
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

    def test_route_chatbot_error(self):
        chatbot_model = 'openai: invalid_model_name'
        with self.assertRaises(ValueError):
            route_chatbot(chatbot_model + 'error')
