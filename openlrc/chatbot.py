import asyncio
import os
import time
from typing import List, Union, Dict

import openai
from aiohttp import ClientSession

from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.utils import get_messages_token_number, get_text_token_number


class GPTBot:
    def __init__(self, model='gpt-3.5-turbo', retry=5, max_async=16, fee_limit=0.1):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.model = model
        self.retry = retry
        self.max_async = max_async
        self.fee_limit = fee_limit

    def get_fee(self, messages: List[Dict]):
        """
        Calculate the total fee for the given messages.
        Pricing info from https://openai.com/pricing
        """
        # Pricing for 1k token
        pricing = {
            'gpt-3.5': (0.002, 0.002),
            'gpt-4': (0.03, 0.06),
            'gpt-4-32k': (0.06, 0.12)
        }

        token_map = {'system': 0, 'user': 0, 'assistant': 0}
        for message in messages:
            token_map[message['role']] += get_text_token_number(message['content'])

        if self.model.startswith('gpt-3.5'):
            prompt_price, completion_price = pricing['gpt-3.5']
        elif self.model.startswith('gpt-4-32k'):
            prompt_price, completion_price = pricing['gpt-4-32k']
        elif self.model.startswith('gpt-4'):
            prompt_price, completion_price = pricing['gpt-4']
        else:
            raise ChatBotException('Fail to get fee. Invalid model name.')

        total_price = (sum(token_map.values()) * prompt_price + token_map['user'] * completion_price) / 1000

        return total_price

    async def _create_achat(self, messages: List[Dict]):
        logger.debug(f'Raw content: {messages}')

        response = None
        for _ in range(self.retry):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                break
            except openai.error.RateLimitError:
                logger.warning('Rate limit exceeded. Wait 10s before retry.')
                time.sleep(10)
            except openai.error.APIConnectionError:
                logger.warning('API connection error. Wait 30s before retry.')
                time.sleep(30)

        if not response:
            raise ChatBotException('Failed to create a chat.')

        return response

    async def _amessage(self, messages_list: List[List[Dict]]):
        """
        Async send messages to the GPT chatbot.
        """
        async with openai.aiosession.get(ClientSession()):
            results = await asyncio.gather(*(self._create_achat(message) for message in messages_list))

            return results

    def message(self, messages_list: Union[List[Dict], List[List[Dict]]]):
        """
        Send chunked messages to the GPT chatbot.
        """
        assert messages_list, 'Empty message list.'

        if isinstance(messages_list[0], dict):  # convert messages List[Dict] to messages_list List[List[Dict]]
            messages_list = [messages_list]

        # Calculate the total sending token number and approximated billing fee.
        token_numbers = [get_messages_token_number(message) for message in messages_list]
        logger.info(f'Max token num: {max(token_numbers):.0f}, '
                    f'Avg token num: {sum(token_numbers) / len(token_numbers):.0f}')
        # if any of token number exceeds the limit, raise an exception.
        if any(token_number > 3072 for token_number in token_numbers):
            raise ChatBotException(f'Token number {max(token_numbers)} exceeds the limit.')

        # if the approximated billing fee exceeds the limit, raise an exception.
        approximated_fee = sum([self.get_fee(messages) for messages in messages_list])
        logger.info(f'Approximated billing fee: {approximated_fee:.4f} US$')
        if approximated_fee > self.fee_limit:
            raise ChatBotException(f'Approximated billing fee {approximated_fee} '
                                   f'exceeds the limit: {self.fee_limit}$.')

        try:
            results = asyncio.run(self._amessage(messages_list))
        except ChatBotException as e:
            raise e

        return results
