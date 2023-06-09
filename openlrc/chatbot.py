import asyncio
import os
import time
from typing import List, Union

import openai
from aiohttp import ClientSession

from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.utils import get_token_number


class GPTBot:
    def __init__(self, system_prompt="", model='gpt-3.5-turbo', retry=5, max_async=16, fee_limit=0.1):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.model = model
        self.system_prompt = system_prompt
        self.retry = retry
        self.max_async = max_async
        self.fee_limit = fee_limit

    def get_fee(self, user_prompts):
        """
        Calculate the total fee for the given user prompts.
        Pricing info from https://openai.com/pricing
        """
        # Pricing for 1k token
        pricing = {
            'gpt-3.5': (0.002, 0.002),
            'gpt-4': (0.03, 0.06),
            'gpt-4-32k': (0.06, 0.12)
        }

        system_token, user_token = get_token_number(self.system_prompt), get_token_number(' '.join(user_prompts))

        if self.model.startswith('gpt-3.5'):
            prompt_price, completion_price = pricing['gpt-3.5']
        elif self.model.startswith('gpt-4-32k'):
            prompt_price, completion_price = pricing['gpt-4-32k']
        elif self.model.startswith('gpt-4-32k'):
            prompt_price, completion_price = pricing['gpt-4']
        else:
            raise ChatBotException('Fail to get fee. Invalid model name.')

        total_price = ((system_token + user_token) * prompt_price + user_token * completion_price) / 1000

        return total_price

    async def _create_achat(self, user_prompt):
        logger.debug(f'Raw content: {user_prompt}')

        response = None
        for _ in range(self.retry):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            'role': 'system',
                            'content': self.system_prompt
                        },
                        {
                            'role': 'user',
                            'content': user_prompt
                        }
                    ]
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

    async def _amessage(self, messages):
        """
        Async send messages to the GPT chatbot.
        :param messages:
        :return:
        """
        async with openai.aiosession.get(ClientSession()):
            results = await asyncio.gather(*(self._create_achat(message) for message in messages))

            return results

    def message(self, messages: Union[str, List]):
        """
        Send messages to the GPT chatbot.
        """
        if isinstance(messages, str):
            messages = [messages]

        # Calculate the total sending token number and approximated billing fee.
        token_numbers = [get_token_number(message + self.system_prompt) for message in messages]

        # if any of token number exceeds the limit, raise an exception.
        if any(token_number > 2048 for token_number in token_numbers):
            raise ChatBotException(f'Token number {max(token_numbers)} exceeds the limit.')

        # if the approximated billing fee exceeds the limit, raise an exception.
        approximated_fee = self.get_fee(messages)
        logger.info(f'Approximated billing fee: {approximated_fee} US$')
        if approximated_fee > self.fee_limit:
            raise ChatBotException(f'Approximated billing fee {approximated_fee} '
                                   f'exceeds the limit: {self.fee_limit}$.')

        try:
            results = asyncio.run(self._amessage(messages))
        except ChatBotException as e:
            raise e

        return results
