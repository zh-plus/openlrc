#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from collections import namedtuple
from math import isclose

from openlrc.chatbot import GPTBot


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
        assert isclose(fee, 2.6e-05)

    def test_update_fee(self):
        bot = self.bot
        Response = namedtuple('response', 'usage')
        bot.api_fees += [0]
        response1 = Response({'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300})
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = Response({'prompt_tokens': 200, 'completion_tokens': 400, 'total_tokens': 600})
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = Response({'prompt_tokens': 300, 'completion_tokens': 600, 'total_tokens': 900})
        bot.update_fee(response3)

        assert bot.api_fees == [0.0011, 0.0022, 0.0033]

    def test_message(self):
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
