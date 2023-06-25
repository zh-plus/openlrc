#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import json
import os
import re
import uuid

import requests

from openlrc.chatbot import GPTBot
from openlrc.logger import logger
from openlrc.prompter import prompter_map, BaseTranslatePrompter


class GPTTranslator:
    def __init__(self, prompter: str = 'base_trans', fee_limit=0.1, chunk_size=30, intercept_line=None,
                 force_translate=False):
        """
        :param prompter: Translate prompter, choices can be found in `prompter_map` from prompter.py.
        :param fee_limit: Fee limit (USD) for OpenAI API.
        :param chunk_size: Use small (<20) chunk size for speed (more async call), and enhance translation
                    stability (keep audio timeline consistency).
        :param intercept_line: Intercepted text line number.
        :param force_translate: Force translation even if the source language is the same as the target language.
        """
        if prompter not in prompter_map:
            raise ValueError(f'Prompter {prompter} not found.')

        self.prompter = prompter
        self.fee_limit = fee_limit
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.force_translate = force_translate

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
        content = response.choices[0].message.content

        try:
            # Extract summary tag
            summary = re.search(r'<summary>(.*)</summary>', content)
            scene = re.search(r'<scene>(.*)</scene>', content)

            summary = summary.group(1) if summary else ''
            scene = scene.group(1) if scene else ''

            translation = re.findall(r'Translation>\n*(.*?)(?:#\d+|<summary>|\n*$)', content, re.DOTALL)

            return summary.strip(), scene.strip(), [t.strip() for t in translation]

        except Exception as e:
            logger.error(f'Failed to extract contents from response: {content}')
            raise e

    def translate(self, texts, src_lang, target_lang, audio_type='Anime', title='', background='', synopsis='',
                  compare_path='test_intermediate.json'):
        prompter: BaseTranslatePrompter = prompter_map[self.prompter](
            src_lang, target_lang, audio_type, title=title, background=background, synopsis=synopsis)
        translate_bot = GPTBot(fee_limit=self.fee_limit)
        translate_bot.update(temperature=0.7)

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)

        # Start chunk-by-chunk translation
        translations = []
        summaries = []
        summary, scene = '', ''
        for i, chunk in enumerate(chunks, start=1):
            user_input = prompter.format_texts(chunk)
            messages_list = [
                {'role': 'system', 'content': prompter.system()},
                {'role': 'user', 'content': prompter.user(i, user_input, summaries, scene)}
            ]
            response = translate_bot.message(messages_list, output_checker=prompter.check_format)[0]
            summary, scene, translated = self.parse_responses(response)
            translations.extend(translated)
            summaries.append(summary)
            logger.info(f'Translating {title}: {i}/{len(chunks)}')
            logger.info(f'summary: {summary}')
            logger.info(f'scene: {scene}')

        self.api_fee += sum(translate_bot.api_fees)
        compare_results = {
            'compare': [{'idx': i, 'input': user_input, 'output': translation} for
                        i, (user_input, translation) in
                        enumerate(zip(texts, translations), start=1)]}

        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(compare_results, f, indent=4, ensure_ascii=False)

        return translations


class MSTranslator:
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

    def translate(self, texts, src_lang, target_lang):
        params = {
            'api-version': '3.0',
            'from': src_lang,
            'to': target_lang
        }

        body = [{'text': text} for text in texts]

        request = requests.post(self.constructed_url, params=params, headers=self.headers, json=body)
        response = request.json()

        return json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))
