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

from openlrc.chatbot import chatbot_map, model2chatbot
from openlrc.logger import logger
from openlrc.prompter import prompter_map, BaseTranslatePrompter, AtomicTranslatePrompter


class Translator(ABC):

    @abstractmethod
    def translate(self, texts: Union[str, List[str]], src_lang, target_lang):
        pass


class LLMTranslator(Translator):
    def __init__(self, chatbot_model: str = 'gpt-3.5-turbo', prompter: str = 'base_trans', fee_limit=0.1,
                 chunk_size=30, intercept_line=None, proxy=None):
        """
        :param chatbot: Chatbot, choices can be found using LLMTranslator().list_chatbots().
        :param prompter: Translate prompter, choices can be found in `prompter_map` from prompter.py.
        :param fee_limit: Fee limit (USD) for OpenAI API.
        :param chunk_size: Use small (<20) chunk size for speed (more async call), and enhance translation
                    stability (keep audio timeline consistency).
        :param intercept_line: Intercepted text line number.
        """
        if prompter not in prompter_map:
            raise ValueError(f'Prompter {prompter} not found.')

        if chatbot_model not in model2chatbot.keys():
            raise ValueError(f'Chatbot {chatbot_model} not supported.')

        chatbot_category = chatbot_map[model2chatbot[chatbot_model]]
        self.chatbot = chatbot_category(model=chatbot_model, fee_limit=fee_limit, proxy=proxy, temperature=0.7)

        self.prompter = prompter
        self.fee_limit = fee_limit
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.proxy = proxy

    @staticmethod
    def list_chatbots():
        return list(chatbot_map.keys())

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

    def parse_responses(self, response):
        """
        Parse response from OpenAI API.
        :return: summary, scene, translations
        """
        content = self.chatbot.get_content(response)

        try:
            # Extract summary tag
            summary = re.search(r'<summary>(.*)</summary>', content)
            scene = re.search(r'<scene>(.*)</scene>', content)

            summary = summary.group(1) if summary else ''
            scene = scene.group(1) if scene else ''

            translation = re.findall(r'Translation>\n*(.*?)(?:#\d+|<summary>|\n*$)', content, re.DOTALL)

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
                  background='', description='', compare_path: Path = Path('translate_intermediate.json')):
        if not isinstance(texts, list):
            texts = [texts]

        prompter: BaseTranslatePrompter = prompter_map[self.prompter](
            src_lang, target_lang, audio_type, title=title, background=background, description=description)

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
            user_input = prompter.format_texts(chunk)
            messages_list = [
                {'role': 'system', 'content': prompter.system()},
                {'role': 'user', 'content': prompter.user(i, user_input, summaries, scene)}
            ]
            response = self.chatbot.message(messages_list, output_checker=prompter.check_format)[0]
            summary, scene, translated = self.parse_responses(response)
            # TODO: Check translation consistency (1-to-1 correspondence)

            # fail to ensure length consistent after retries, use atomic translation instead
            if len(translated) != len(chunk):
                logger.warning(f'Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},'
                               f'Trying to use atomic translation instead.')
                chunk_texts = [item[1] for item in chunk]
                translated = self.atomic_translate(chunk_texts, src_lang, target_lang)

            translations.extend(translated)
            summaries.append(summary)
            logger.info(f'Translated {title}: {i}/{len(chunks)}')
            logger.info(f'summary: {summary}')
            logger.info(f'scene: {scene}')

            compare_list.extend([{'chunk': i,
                                  'idx': item[0] if item else 'N\\A',
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

# Not integrated by the openlrc main function because of performance
#
# class DeepLTranslator(Translator):
#     def __init__(self):
#         self.key = os.environ['DEEPL_KEY']
#         self.translator = deepl.Translator(self.key)
#
#     def _check_limit(self, texts: List[str]):
#         usage = self.translator.get_usage()
#         char_num = sum([len(text) for text in texts])
#
#         if usage.character.count + char_num > usage.character.limit:
#             raise RuntimeError(f'This translate call would exceed DeepL character limit: {usage.character.limit}')
#
#     def translate(self, texts: Union[str, List[str]], src_lang, target_lang):
#         if not isinstance(texts, list):
#             texts = [texts]
#
#         self._check_limit(texts)
#
#         translations = self.translator.translate_text(texts, target_lang=target_lang)
#
#         if not isinstance(translations, list):
#             translations = [translations]
#
#         translations = [translation.text for translation in translations]
#
#         return translations
