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

from openlrc.chatbot import route_chatbot, all_pricing
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
        Initialize the LLMTranslator with given parameters.

        Args:
            chatbot_model (str): The model of the chatbot to use.
            prompter (str): The prompter to format the texts for translation.
            fee_limit (float): The fee limit for the API.
            chunk_size (int): The size of text chunks for translation.
            intercept_line (int): The line number to intercept.
            proxy (str): The proxy server to use.
            base_url_config (dict): The base URL configuration for the chatbot API.
            retry_model (str): The model to use for retrying translation if the primary model fails.
        """
        if prompter not in prompter_map:
            raise ValueError(f'Prompter {prompter} not found.')

        self.temperature = 0.9

        chatbot_cls, model_name = route_chatbot(chatbot_model)
        self.chatbot = chatbot_cls(model=model_name, fee_limit=fee_limit, proxy=proxy, retry=3,
                                   temperature=self.temperature, base_url_config=base_url_config)

        self.retry_chatbot = None
        if retry_model:
            retry_chatbot_cls, retry_model_name = route_chatbot(retry_model)
            self.retry_chatbot = retry_chatbot_cls[retry_model](
                model=retry_model_name, fee_limit=fee_limit, proxy=proxy, retry=3, temperature=self.temperature,
                base_url_config=base_url_config
            )

        self.prompter = prompter
        self.fee_limit = fee_limit
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.retry_model = retry_model

    @staticmethod
    def list_chatbots():
        """
        List available chatbot models.

        Returns:
            List[str]: List of available chatbot models.
        """
        return list(all_pricing.keys())

    @staticmethod
    def make_chunks(texts, chunk_size=30):
        """
        Split the text into chunks of specified size.

        Args:
            texts (List[str]): List of texts to be chunked.
            chunk_size (int): Size of each chunk.

        Returns:
            List[List[Tuple[int, str]]]: List of chunks, each chunk is a list of (line_number, text) tuples.
        """
        chunks = []
        start = 1
        for i in range(0, len(texts), chunk_size):
            chunk = [(start + j, text) for j, text in enumerate(texts[i:i + chunk_size])]
            start += len(chunk)
            chunks.append(chunk)

        if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def _parse_responses(self, resp, potential_prefix_combo, changed_chatbot=None):
        """
        Parse the response from the chatbot API.

        Args:
            resp (str): The response from the chatbot API.
            potential_prefix_combo (List[Tuple[str, str]]): Potential prefix combinations for parsing.
            changed_chatbot: The chatbot instance used for parsing if different from the primary chatbot.

        Returns:
            Tuple[str, str, List[str]]: Parsed summary, scene, and translations from the response.
        """
        content = changed_chatbot.get_content(resp) if changed_chatbot else self.chatbot.get_content(resp)

        try:
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

    def _translate_chunk(self, chunk, prompter, summaries, scene, i):
        """
        Translate a single chunk of text.

        Args:
            chunk (List[Tuple[int, str]]): The chunk of text to be translated.
            prompter (BaseTranslatePrompter): The prompter instance to format the text.
            summaries (List[str]): List of summaries for context.
            scene (str): The current scene context.
            i (int): The chunk index.

        Returns:
            Tuple[str, str, List[str]]: The summary, scene, and translated texts for the chunk.
        """

        def send_and_parse(messages, chatbot):
            """
            Helper function to send messages to the chatbot and parse the response.

            Args:
                messages (List[dict]): List of messages to send.
                chatbot: The chatbot instance to use.

            Returns:
                Tuple[str, str, List[str]]: The parsed summary, scene, and translations.
            """
            resp = chatbot.message(messages, output_checker=prompter.check_format)[0]
            return self._parse_responses(resp, prompter.potential_prefix_combo, changed_chatbot=chatbot)

        user_input = prompter.format_texts(chunk)
        glossary_user_prompt = prompter.user(i, user_input, summaries, scene)
        non_glossary_user_prompt = prompter_map[self.prompter](
            prompter.src_lang, prompter.target_lang, prompter.audio_type, title=prompter.title,
            background=prompter.background, description=prompter.description
        ).user(i, user_input, summaries, scene)
        messages_list = [
            {'role': 'system', 'content': prompter.system()},
            {'role': 'user', 'content': glossary_user_prompt},
        ]
        summary, scene, translated = send_and_parse(messages_list.copy(), self.chatbot)

        if len(translated) != len(chunk):
            logger.warning(f'Cant translate chunk {i} with glossary, trying to remove glossary.')
            messages_list[1]['content'] = non_glossary_user_prompt
            summary, scene, translated = send_and_parse(messages_list.copy(), self.chatbot)

        if self.retry_chatbot and len(translated) != len(chunk):
            logger.warning(
                f'Trying to change chatbot to keep performing chunked translation. Retry chatbot: {self.retry_model}'
            )
            messages_list[1]['content'] = glossary_user_prompt
            summary, scene, translated = send_and_parse(messages_list.copy(), self.retry_chatbot)

            if len(translated) != len(chunk):
                logger.warning(f'New bot: Trying to remove glossary to keep performing chunked translation.')
                messages_list[1]['content'] = non_glossary_user_prompt
                summary, scene, translated = send_and_parse(messages_list.copy(), self.retry_chatbot)

        return summary, scene, translated

    def translate(self, texts: Union[str, List[str]], src_lang, target_lang, audio_type='Anime', title='',
                  background='', description='', compare_path: Path = Path('translate_intermediate.json'),
                  glossary: dict = None):
        """
        Translate a list of texts from source language to target language.

        Args:
            texts (Union[str, List[str]]): The texts to be translated.
            src_lang (str): The source language.
            target_lang (str): The target language.
            audio_type (str): The type of audio (e.g., 'Anime').
            title (str): The title of the content.
            background (str): The background context.
            description (str): The description of the content.
            compare_path (Path): The path to save intermediate translation results.
            glossary (dict): The glossary to use for translation.

        Returns:
            List[str]: The translated texts.
        """
        if not isinstance(texts, list):
            texts = [texts]

        prompter: BaseTranslatePrompter = prompter_map[self.prompter](
            src_lang, target_lang, audio_type, title=title, background=background, description=description,
            glossary=glossary
        )

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)
        logger.info(f'Translating {title}: {len(chunks)} chunks, {len(texts)} lines in total.')

        translations = []
        summaries = []
        summary, scene = '', ''
        compare_list = []
        start_chunk = 0

        if compare_path.exists():
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
            summary, scene, translated = self._translate_chunk(chunk, prompter, summaries, scene, i)

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
                                  'model': self.chatbot.model,
                                  'input': item[1] if item else 'N\\A',
                                  'output': trans if trans else 'N\\A'}
                                 for (item, trans) in zip_longest(chunk, translated)])
            compare_results = {'compare': compare_list, 'summaries': summaries, 'scene': scene}
            with open(compare_path, 'w', encoding='utf-8') as f:
                json.dump(compare_results, f, indent=4, ensure_ascii=False)

        self.api_fee += sum(self.chatbot.api_fees)

        return translations

    def atomic_translate(self, texts, src_lang, target_lang):
        """
        Perform atomic translation for each text.

        Args:
            texts (List[str]): List of texts to be translated.
            src_lang (str): Source language.
            target_lang (str): Target language.

        Returns:
            List[str]: List of translated texts.
        """
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
