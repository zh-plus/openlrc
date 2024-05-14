#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from itertools import zip_longest
from pathlib import Path
from typing import Union, List

import requests

from openlrc.chatbot import model2chatbot
from openlrc.logger import logger
from openlrc.prompter import prompter_map, BaseTranslatePrompter, AtomicTranslatePrompter


class Translator(ABC):

    @abstractmethod
    def translate(self, texts: Union[str, List[str]], src_lang, target_lang):
        pass


class LLMTranslator(Translator):
    def __init__(self, chatbot_model: str = 'gpt-3.5-turbo', prompter: str = 'base_trans', fee_limit=0.2,
                 chunk_size=30, intercept_line=None, proxy=None, base_url_config=None, retry_model=None):
        """
        Args:
            chatbot_model: Chatbot instance. Choices can be found using `LLMTranslator().list_chatbots()`.
            prompter: Translate prompter instance. Choices can be found in `prompter_map` from `prompter.py`.
            fee_limit (float): Fee limit (USD) for the OpenAI API.
            chunk_size (int): Use a small chunk size (<20) for speed (more asynchronous calls) and to enhance translation
                              stability (keeping audio timeline consistency).
            intercept_line (int): Intercepted text line number.
            proxy (str): Proxy server. e.g. http://127.0.0.1:7890
            base_url_config (dict): Base URL configuration for the chatbot API.
            retry_model (str): Retry chatbot model if the translation fails.
        """
        if prompter not in prompter_map:
            raise ValueError(f'Prompter {prompter} not found.')

        if chatbot_model not in model2chatbot.keys():
            raise ValueError(f'Chatbot {chatbot_model} not supported.')

        self.temperature = 0.9

        chatbot_category = model2chatbot[chatbot_model]
        self.chatbot = chatbot_category(model=chatbot_model, fee_limit=fee_limit, proxy=proxy, retry=3,
                                        temperature=self.temperature, base_url_config=base_url_config)
        self.retry_chatbot = model2chatbot[retry_model](
            model=retry_model, fee_limit=fee_limit,
            proxy=proxy, retry=3,
            temperature=self.temperature,
            base_url_config=base_url_config
        ) if retry_model else None

        self.prompter = prompter
        self.fee_limit = fee_limit
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.retry_model = retry_model

    @staticmethod
    def list_chatbots():
        return list(model2chatbot.keys())

    @staticmethod
    def make_chunks(texts, chunk_size=30):
        """
        Split the subtitle into chunks, each chunk has a line number of chunk_size.
        :return: List of chunks, each chunk is a list of (line_number, text) tuples
        """

        chunks = []
        start = 1
        for i in range(0, len(texts), chunk_size):
            chunk = [(start + j, text) for j, text in enumerate(texts[i:i + chunk_size])]
            start += len(chunk)
            chunks.append(chunk)

        # if the last chunk is too small, merge it to the previous chunk
        if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def parse_responses(self, response, potential_prefix_combo, changed_chatbot=None):
        """
        Parse response from OpenAI API.
        :return: summary, scene, translations
        """
        content = changed_chatbot.get_content(response) if changed_chatbot else self.chatbot.get_content(response)

        try:
            # Extract summary tag
            summary = re.search(r'<summary>(.*)</summary>', content)
            scene = re.search(r'<scene>(.*)</scene>', content)

            summary = summary.group(1) if summary else ''
            scene = scene.group(1) if scene else ''

            for _, trans_prefix in potential_prefix_combo:
                translation = re.findall(f'{trans_prefix}\n*(.*?)(?:#\d+|<summary>|\n*$)', content, re.DOTALL)
                if translation:
                    break
            else:
                return summary.strip(), scene.strip(), []

            # Remove "</summary>\nxxx</scene>" tags (or some wierd tags like </p> ❓) from translation
            if any([re.search(r'(<.*?>|</.*?>)', t) for t in translation]):
                logger.warning(f'The extracted translation from response contains tags: {content}, tags removed')
                translation = [
                    re.sub(
                        r'(<.*?>|</.*?>).*',
                        '', t, flags=re.DOTALL
                    )
                    for t in translation
                ]

            return summary.strip(), scene.strip(), [t.strip() for t in translation]

        except Exception as e:
            logger.error(f'Failed to extract contents from response: {content}')
            raise e

    def translate(self, texts: Union[str, List[str]], src_lang, target_lang, audio_type='Anime', title='',
                  background='', description='', compare_path: Path = Path('translate_intermediate.json'),
                  glossary: dict = None):
        if not isinstance(texts, list):
            texts = [texts]

        prompter: BaseTranslatePrompter = prompter_map[self.prompter](
            src_lang, target_lang, audio_type, title=title, background=background, description=description,
            glossary=glossary
        )

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)
        logger.info(f'Translating {title}: {len(chunks)} chunks, {len(texts)} lines in total.')

        # Start chunk-by-chunk translation
        translations = []
        summaries = []
        summary, scene = '', ''
        compare_list = []
        start_chunk = 0

        if compare_path.exists():
            # TODO: Check if the chunk_size is consistent

            logger.info(f'Resume from {compare_path}')
            with open(compare_path, 'r', encoding='utf-8') as f:
                compare_results = json.load(f)
            compare_list = compare_results['compare']
            summaries = compare_results['summaries']
            scene = compare_results['scene']
            translations = [item['output'] for item in compare_list]
            start_chunk = compare_list[-1]['chunk']
            logger.info(f'Resume translation from chunk {start_chunk}')

        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            atomic = False
            user_input = prompter.format_texts(chunk)
            glossary_user_prompt = prompter.user(i, user_input, summaries, scene)
            messages_list = [
                {'role': 'system', 'content': prompter.system()},
                {'role': 'user', 'content': glossary_user_prompt},
            ]
            response = self.chatbot.message(messages_list.copy(), output_checker=prompter.check_format)[0]
            summary, scene, translated = self.parse_responses(response, prompter.potential_prefix_combo)
            # TODO: Check translation consistency (1-to-1 correspondence)

            non_glossary_user_prompt = prompter_map[self.prompter](
                src_lang, target_lang, audio_type, title=title, background=background, description=description
            ).user(i, user_input, summaries, scene)
            # glossary in the prompt can be unstable, try to remove glossary to keep going.
            if len(translated) != len(chunk):
                logger.warning(f'Cant translate chunk {i} with glossary, trying to remove glossary.')
                messages_list[1]['content'] = non_glossary_user_prompt
                default_retry = self.chatbot.retry
                self.chatbot.retry = 3  # only retry 3 times
                response = self.chatbot.message(messages_list.copy(), output_checker=prompter.check_format)[0]
                summary, scene, translated = self.parse_responses(response, prompter.potential_prefix_combo)
                self.chatbot.retry = default_retry

            # Try to change chatbot if the other chatbot is accessible
            if self.retry_chatbot and len(translated) != len(chunk):
                logger.warning(
                    f'Trying to change chatbot to keep performing chunked translation. Retry chatbot: {self.retry_model}'
                )
                messages_list[1]['content'] = glossary_user_prompt
                response = self.retry_chatbot.message(messages_list.copy(), output_checker=prompter.check_format)[0]
                summary, scene, translated = self.parse_responses(response, prompter.potential_prefix_combo,
                                                                  changed_chatbot=self.retry_chatbot)

                if len(translated) != len(chunk):
                    logger.warning(f'New bot: Trying to remove glossary to keep performing chunked translation.')
                    messages_list[1]['content'] = non_glossary_user_prompt
                    response = self.retry_chatbot.message(messages_list.copy(), output_checker=prompter.check_format)[0]
                    summary, scene, translated = self.parse_responses(response, prompter.potential_prefix_combo,
                                                                      changed_chatbot=self.retry_chatbot)

            # Finally, use atomic translation
            if len(translated) != len(chunk):
                logger.warning(f'Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},'
                               f'Trying to use atomic translation instead.')
                chunk_texts = [item[1] for item in chunk]
                translated = self.atomic_translate(chunk_texts, src_lang, target_lang)
                atomic = True

            translations.extend(translated)
            summaries.append(summary)
            logger.info(f'Translated {title}: {i}/{len(chunks)}')
            logger.info(f'summary: {summary}')
            logger.info(f'scene: {scene}')

            compare_list.extend([{'chunk': i,
                                  'idx': item[0] if item else 'N\\A',
                                  'method': 'atomic' if atomic else 'chunked',
                                  'input': item[1] if item else 'N\\A',
                                  'output': trans if trans else 'N\\A'}
                                 for (item, trans) in zip_longest(chunk, translated)])
            compare_results = {'compare': compare_list, 'summaries': summaries, 'scene': scene}
            # Save for resume
            with open(compare_path, 'w', encoding='utf-8') as f:
                json.dump(compare_results, f, indent=4, ensure_ascii=False)

        self.api_fee += sum(self.chatbot.api_fees)

        return translations

    def atomic_translate(self, texts, src_lang, target_lang):
        prompter = AtomicTranslatePrompter(src_lang, target_lang)
        message_lists = [[
            {'role': 'user', 'content': prompter.user(text)}
        ] for text in texts]

        responses = self.chatbot.message(message_lists, output_checker=prompter.check_format)
        translated = list(map(self.chatbot.get_content, responses))

        assert len(translated) == len(texts), f'Atomic translation failed: {len(translated)} vs {len(texts)}'

        return translated


class MSTranslator(Translator):
    def __init__(self):
        self.key = os.environ['MS_TRANSLATOR_KEY']
        self.endpoint = 'https://api.cognitive.microsofttranslator.com'
        self.location = 'eastasia'
        self.path = '/translate'
        self.constructed_url = self.endpoint + self.path

        self.headers = {
            'Ocp-Apim-Subscription-Key': self.key,
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def translate(self, texts: Union[str, List[str]], src_lang, target_lang):
        params = {
            'api-version': '3.0',
            'from': src_lang,
            'to': target_lang
        }

        body = [{'text': text} for text in texts]

        try:
            request = requests.post(self.constructed_url, params=params, headers=self.headers, json=body, timeout=20)
        except TimeoutError:
            raise RuntimeError('Failed to connect to Microsoft Translator API.')
        response = request.json()

        return json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
