#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import unittest
from math import isclose

from pydantic import BaseModel

from openlrc.chatbot import GPTBot


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(BaseModel):
    usage: CompletionUsage


class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.bot = GPTBot(temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.05)

    def test_estimate_fee(self):
        bot = self.bot
        messages = [
            {'role': 'system', 'content': 'You are gpt.'},
            {'role': 'user', 'content': 'Hello'},
        ]
        fee = bot.estimate_fee(messages)
        assert isclose(fee, 6e-06)

    def test_update_fee(self):
        bot = self.bot
        bot.api_fees += [0]
        response1 = OpenAIResponse(usage=CompletionUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300))
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = OpenAIResponse(usage=CompletionUsage(prompt_tokens=200, completion_tokens=400, total_tokens=600))
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = OpenAIResponse(usage=CompletionUsage(prompt_tokens=300, completion_tokens=600, total_tokens=900))
        bot.update_fee(response3)

        assert bot.api_fees == [0.00035, 0.0007, 0.0010500000000000002]

    def test_message_async(self):
        bot = self.bot
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ],
        ]
        results = bot.message(messages_list)
        assert all(['hello' in r.choices[0].message.content.lower() for r in results])

    def test_message_seq(self):
        bot = self.bot
        messages_list = [
            [
                {'role': 'user', 'content': 'Echo hello:'}
            ]
        ]
        results = bot.message(messages_list)
        assert 'hello' in results[0].choices[0].message.content.lower()
