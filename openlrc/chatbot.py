#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import asyncio
import os
import random
import re
import time
from copy import deepcopy
from typing import List, Union, Dict, Callable, Optional

import anthropic
import google.generativeai as genai
import httpx
import openai
from anthropic import AsyncAnthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message
from google.generativeai import GenerationConfig
from google.generativeai.types import AsyncGenerateContentResponse, GenerateContentResponse, \
    HarmCategory, HarmBlockThreshold
from openai import AsyncClient as AsyncGPTClient
from openai.types.chat import ChatCompletion

from openlrc.exceptions import ChatBotException, LengthExceedException
from openlrc.logger import logger
from openlrc.utils import get_messages_token_number, get_text_token_number

model2chatbot = {}
all_pricing = {
    # Third-party provider models from https://api.g4f.icu/pricing
    'mixtral-8x7b-32768': (0.25, 1),
    'llama2-70b-4096': (0.25, 1.25),

    # https://platform.deepseek.com/api-docs/pricing/
    'deepseek-chat': (0.14, 0.28)
}


def _register_chatbot(cls):
    all_pricing.update(cls.pricing)

    for model in cls.pricing:
        model2chatbot[model] = cls

    return cls


def route_chatbot(model):
    if ':' in model:
        chatbot_type, chatbot_model = re.match(r'(.+):(.+)', model).groups()
        chatbot_type, chatbot_model = chatbot_type.strip().lower(), chatbot_model.strip()

        if chatbot_model not in all_pricing:
            raise ValueError(f'Invalid model {chatbot_model}.')

        if chatbot_type == 'openai':
            return GPTBot, chatbot_model
        elif chatbot_type == 'anthropic':
            return ClaudeBot, chatbot_model
        else:
            raise ValueError(f'Invalid chatbot type {chatbot_type}.')

    if model not in model2chatbot:
        raise ValueError(f'Invalid model {model}.')

    return model2chatbot[model], model


class ChatBot:
    pricing = None

    def __init__(self, pricing, temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.8):
        self.pricing = pricing
        self._model = None

        self.temperature = temperature
        self.top_p = top_p
        self.retry = retry
        self.max_async = max_async
        self.fee_limit = fee_limit

        self.api_fees = []

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model not in all_pricing:
            raise ValueError(f'Invalid model {model}.')
        self._model = model

    def estimate_fee(self, messages: List[Dict]):
        """
        Estimate the total fee for the given messages.
        """
        token_map = {'system': 0, 'user': 0, 'assistant': 0}
        for message in messages:
            token_map[message['role']] += get_text_token_number(message['content'])

        prompt_price, completion_price = all_pricing[self.model]

        total_price = (sum(token_map.values()) * prompt_price + token_map['user'] * completion_price * 2) / 1000000

        return total_price

    def update_fee(self, response):
        raise NotImplementedError()

    def get_content(self, response):
        raise NotImplementedError()

    async def _create_achat(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None,
                            output_checker: Callable = lambda user_input, generated_content: True):
        raise NotImplementedError()

    async def _amessage(self, messages_list: List[List[Dict]], stop_sequences: Optional[List[str]] = None,
                        output_checker: Callable = lambda user_input, generated_content: True):
        """
        Async send messages to the GPT chatbot.
        """

        results = await asyncio.gather(
            *(
                self._create_achat(
                    message, stop_sequences=stop_sequences, output_checker=output_checker
                ) for message in messages_list
            )
        )

        return results

    def message(self, messages_list: Union[List[Dict], List[List[Dict]]], stop_sequences: Optional[List[str]] = None,
                output_checker: Callable = lambda user_input, generated_content: True):
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
            results = asyncio.run(
                self._amessage(messages_list, stop_sequences=stop_sequences, output_checker=output_checker)
            )
        except ChatBotException as e:
            logger.error(f'Failed to message with GPT. Error: {e}')
            raise e
        finally:
            logger.info(f'Translation fee for this call: {self.api_fees[-1]:.4f} USD')
            logger.info(f'Total bot translation fee: {sum(self.api_fees):.4f} USD')

        return results

    def __str__(self):
        return f'ChatBot ({self.model})'


@_register_chatbot
class GPTBot(ChatBot):
    # Pricing for 1M tokens, info from https://openai.com/pricing
    pricing = {
        'gpt-4o-mini': (0.15, 0.6),
        'gpt-3.5-turbo-0125': (0.5, 1.5),
        'gpt-3.5-turbo': (0.5, 1.5),
        'gpt-4-0125-preview': (10, 30),
        'gpt-4-turbo-preview': (10, 30),
        'gpt-4-turbo': (10, 30),
        'gpt-4-turbo-2024-04-09': (10, 30),
        'gpt-4o': (5, 15),
    }

    def __init__(self, model='gpt-4o-mini', temperature=1, top_p=1, retry=8, max_async=16, json_mode=False,
                 fee_limit=0.05, proxy=None, base_url_config=None):

        # clamp temperature to 0-2
        temperature = max(0, min(2, temperature))

        super().__init__(self.pricing, temperature, top_p, retry, max_async, fee_limit)

        self.async_client = AsyncGPTClient(
            api_key=os.environ['OPENAI_API_KEY'],
            http_client=httpx.AsyncClient(proxies=proxy),
            base_url=base_url_config['openai'] if base_url_config and base_url_config['openai'] else None
        )

        self.model = model
        self.json_mode = json_mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.async_client.close()

    def update_fee(self, response: ChatCompletion):
        prompt_price, completion_price = all_pricing[self.model]

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        self.api_fees[-1] += (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000000

    def get_content(self, response):
        return response.choices[0].message.content

    async def _create_achat(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None,
                            output_checker: Callable = lambda user_input, generated_content: True):
        # Check stop sequences
        if stop_sequences and len(stop_sequences) > 4:
            logger.warning('Too many stop sequences. For openai, Only the first 4 will be used.')
            stop_sequences = stop_sequences[:4]

        response = None
        for i in range(self.retry):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    response_format={'type': 'json_object' if self.json_mode else 'text'},
                    stop=stop_sequences
                )
                self.update_fee(response)
                if response.choices[0].finish_reason == 'length':
                    raise LengthExceedException(response)
                if not output_checker(messages[-1]['content'], response.choices[0].message.content):
                    logger.warning(f'Invalid response format. Retry num: {i + 1}.')
                    continue

                break
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.APIError) as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f'{type(e).__name__}: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.')
                time.sleep(sleep_time)

        if not response:
            raise ChatBotException('Failed to create a chat.')

        return response

    @staticmethod
    def _get_sleep_time(error):
        if isinstance(error, openai.RateLimitError):
            return random.randint(30, 60)
        elif isinstance(error, openai.APITimeoutError):
            return 3
        else:
            return 15

@_register_chatbot
class ClaudeBot(ChatBot):
    # Pricing for 1M tokens, info from https://docs.anthropic.com/claude/docs/models-overview#model-comparison
    pricing = {
        'claude-3-opus-20240229': (15, 75),
        'claude-3-sonnet-20240229': (3, 15),
        'claude-3-haiku-20240307': (0.25, 1.25),
        'claude-3-5-sonnet-20240620': (3, 15),
    }

    def __init__(self, model='claude-3-sonnet-20240229', temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.8,
                 proxy=None, base_url_config=None):

        # clamp temperature to 0-1
        temperature = max(0, min(1, temperature))

        super().__init__(self.pricing, temperature, top_p, retry, max_async, fee_limit)

        self.async_client = AsyncAnthropic(
            api_key=os.environ['ANTHROPIC_API_KEY'],
            http_client=httpx.AsyncClient(
                proxies=proxy
            ),
            base_url=base_url_config['anthropic'] if base_url_config and base_url_config['anthropic'] else None
        )

        self.model = model

    def update_fee(self, response: Message):
        prompt_price, completion_price = all_pricing[self.model]

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        self.api_fees[-1] += (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000000

    def get_content(self, response):
        return response.content[0].text

    async def _create_achat(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None,
                            output_checker: Callable = lambda user_input, generated_content: True):
        # No need to check stop sequences for Claude (unlimited)

        # Move "system" role into the parameters
        system_msg = NOT_GIVEN
        if messages[0]['role'] == 'system':
            system_msg = messages.pop(0)['content']

        response = None
        for i in range(self.retry):
            try:
                response = await self.async_client.messages.create(
                    max_tokens=4096,
                    model=self.model,
                    messages=messages,
                    system=system_msg,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop_sequences=stop_sequences
                )
                self.update_fee(response)

                if response.stop_reason == 'max_tokens':
                    raise LengthExceedException(response)
                if not output_checker(messages[-1]['content'], response.content[-1].text):
                    logger.warning(f'Invalid response format. Retry num: {i + 1}.')
                    continue

                break
            except (
            anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.APIError) as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f'{type(e).__name__}: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.')
                time.sleep(sleep_time)

        if not response:
            raise ChatBotException('Failed to create a chat.')

        return response

    def _get_sleep_time(self, error):
        if isinstance(error, anthropic.RateLimitError):
            return random.randint(30, 60)
        elif isinstance(error, anthropic.APITimeoutError):
            return 3
        else:
            return 15


@_register_chatbot
class GeminiBot(ChatBot):
    # Pricing for 1M tokens, info from https://ai.google.dev/pricing
    # Note we consider the pricing for prompts longer than 128k
    pricing = {
        'gemini-1.0-pro-latest': (0.5, 1.5),
        'gemini-1.0-pro': (0.5, 1.5),
        'gemini-1.0-pro-001': (0.5, 1.5),
        'gemini-1.5-flash-latest': (0.175, 2.1),
        'gemini-1.5-flash': (0.175, 2.1),
        'gemini-1.5-flash-001': (0.175, 2.1),
        'gemini-1.5-pro-latest': (1.75, 21),
        'gemini-1.5-pro': (1.75, 21),
        'gemini-1.5-pro-001': (1.75, 21),
    }

    def __init__(self, model='gemini-1.5-flash', temperature=1, top_p=1, retry=8, max_async=16, fee_limit=0.8,
                 proxy=None, base_url_config=None):
        self.temperature = max(0, min(1, temperature))

        super().__init__(self.pricing, temperature, top_p, retry, max_async, fee_limit)

        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = model
        self.config = GenerationConfig(temperature=self.temperature, top_p=self.top_p)
        # Should not block any translation-related content.
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        if proxy:
            logger.warning('Google Gemini SDK does not support proxy, try using the system-level proxy if needed.')

        if base_url_config:
            logger.warning('Google Gemini SDK does not support changing base_url.')

    def update_fee(self, response: Union[GenerateContentResponse, AsyncGenerateContentResponse]):
        prompt_price, completion_price = all_pricing[self.model]

        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count

        self.api_fees[-1] += (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000000

    def get_content(self, response):
        return response.text

    async def _create_achat(self, messages: List[Dict], stop_sequences: Optional[List[str]] = None,
                            output_checker: Callable = lambda user_input, generated_content: True):
        # Check stop sequences
        if stop_sequences and len(stop_sequences) > 5:
            logger.warning('Too many stop sequences. Only the first 5 will be used.')
            stop_sequences = stop_sequences[:5]

        history_messages = deepcopy(messages)
        system_msg = None
        if history_messages[0]['role'] == 'system':
            system_msg = history_messages.pop(0)['content']

        if history_messages[-1]['role'] != 'user':
            logger.error('The last message should be user message.')
        user_msg = history_messages.pop(-1)['content']

        # convert assistant role into model
        for i, message in enumerate(history_messages):
            if message['role'] == 'assistant':
                history_messages[i]['role'] = 'model'

            content = message.pop('content')
            history_messages[i]['parts'] = [{'text': content}]

        self.config.stop_sequences = stop_sequences
        generative_model = genai.GenerativeModel(model_name=self.model, safety_settings=self.safety_settings,
                                                 generation_config=self.config, system_instruction=system_msg)
        client = genai.ChatSession(generative_model, history=history_messages)

        response = None
        for i in range(self.retry):
            try:
                # send_message_async is buggy, so we use send_message instead as a workaround
                response = client.send_message(user_msg)
                self.update_fee(response)
                if not output_checker(user_msg, response.text):
                    logger.warning(f'Invalid response format. Retry num: {i + 1}.')
                    continue

                if not response._done:
                    logger.warning(f'Failed to get a complete response. Retry num: {i + 1}.')
                    continue

                break
            except (genai.RateLimitError, genai.APITimeoutError, genai.APIConnectionError, genai.APIError) as e:
                logger.warning(f'{type(e).__name__}: {e}. Retry num: {i + 1}.')

        if not response:
            raise ChatBotException('Failed to create a chat.')

        return response
