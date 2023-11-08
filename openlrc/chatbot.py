#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import asyncio
import os
import time
from typing import List, Union, Dict, Callable

import openai
from openai import AsyncClient

from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.utils import get_messages_token_number, get_text_token_number


class GPTBot:
    def __init__(self, model='gpt-3.5-turbo-1106', temperature=1, top_p=1, retry=8, max_async=16, json_mode=False,
                 fee_limit=0.05):
        self.client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])

        # Pricing for 1k tokens, info from https://openai.com/pricing
        self.pricing = {
            'gpt-3.5-turbo-1106': (0.001, 0.002),
            'gpt-4-1106-preview': (0.01, 0.03)
        }

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.retry = retry
        self.max_async = max_async
        self.json_mode = json_mode
        self.fee_limit = fee_limit

        self.api_fees = []  # OpenAI API fee for each call

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model not in self.pricing:
            raise ValueError(f'Invalid model {model}.')
        self._model = model

    def update(self, temperature=None, top_p=None):
        if temperature:
            self.temperature = temperature

        if top_p:
            self.top_p = top_p

    def estimate_fee(self, messages: List[Dict]):
        """
        Estimate the total fee for the given messages.
        """
        token_map = {'system': 0, 'user': 0, 'assistant': 0}
        for message in messages:
            token_map[message['role']] += get_text_token_number(message['content'])

        prompt_price, completion_price = self.pricing[self.model]

        total_price = (sum(token_map.values()) * prompt_price + token_map['user'] * completion_price * 2) / 1000

        return total_price

    def update_fee(self, response):
        prompt_price, completion_price = self.pricing[self.model]

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        self.api_fees[-1] += (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000

    async def _create_achat(self, messages: List[Dict], output_checker: Callable = lambda *args, **kw: True):
        logger.debug(f'Raw content: {messages}')

        response = None
        for i in range(self.retry):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    response_format={'type': 'json_object' if self.json_mode else 'text'}
                )
                self.update_fee(response)
                if response.choices[0].finish_reason == 'length':
                    raise ChatBotException(
                        f'Failed to get completion. Exceed max token length. '
                        f'Prompt tokens: {response.usage.prompt_tokens}, '
                        f'Completion tokens: {response.usage.completion_tokens}, '
                        f'Total tokens: {response.usage.total_tokens} '
                        f'Reduce chunk_size may help.'
                    )
                if not output_checker(messages, response.choices[0].message.content):
                    logger.warning(f'Invalid response format. Retry num: {i + 1}.')
                    continue

                break
            except openai.RateLimitError:
                logger.warning(f'Rate limit exceeded. Wait 10s before retry. Retry num: {i + 1}.')
                time.sleep(10)
            except openai.APITimeoutError:
                logger.warning(f'Timeout. Wait 3 before retry. Retry num: {i + 1}.')
                time.sleep(3)
            except openai.APIConnectionError:
                logger.warning(f'API connection error. Wait 15s before retry. Retry num: {i + 1}.')
                time.sleep(15)
            except openai.APIError:
                logger.warning(f'API error. Wait 15s before retry. Retry num: {i + 1}.')
                time.sleep(15)

        if not response:
            raise ChatBotException('Failed to create a chat.')

        return response

    async def _amessage(self, messages_list: List[List[Dict]], output_checker: Callable = lambda *args, **kw: True):
        """
        Async send messages to the GPT chatbot.
        """

        results = await asyncio.gather(
            *(self._create_achat(message, output_checker=output_checker) for message in messages_list)
        )

        return results

    def message(self, messages_list: Union[List[Dict], List[List[Dict]]],
                output_checker: Callable = lambda *args, **kw: True):
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

        # if the approximated billing fee exceeds the limit, raise an exception.
        approximated_fee = sum([self.estimate_fee(messages) for messages in messages_list])
        logger.info(f'Approximated billing fee: {approximated_fee:.4f} USD')
        self.api_fees += [0]  # Actual fee for this translation call.
        if approximated_fee > self.fee_limit:
            raise ChatBotException(f'Approximated billing fee {approximated_fee} '
                                   f'exceeds the limit: {self.fee_limit}$.')

        try:
            results = asyncio.run(self._amessage(messages_list, output_checker=output_checker))
        except ChatBotException as e:
            logger.error(f'Failed to message with GPT. Error: {e}')
            raise e
        finally:
            logger.info(f'Translation fee for this call: {self.api_fees[-1]:.4f} USD')
            logger.info(f'Total bot translation fee: {sum(self.api_fees):.4f} USD')

        return results
