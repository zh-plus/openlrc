#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import os
import uuid
from abc import ABC, abstractmethod
from itertools import zip_longest
from pathlib import Path
from typing import Union, List, Optional, Tuple

import requests

from openlrc.agents import ChunkedTranslatorAgent, ContextReviewerAgent
from openlrc.chatbot import all_pricing
from openlrc.context import TranslationContext, TranslateInfo
from openlrc.logger import logger
from openlrc.prompter import AtomicTranslatePrompter


class Translator(ABC):

    @abstractmethod
    def translate(self, texts: Union[str, List[str]], src_lang: str, target_lang: str,
                  info: TranslateInfo) -> List[str]:
        pass


class LLMTranslator(Translator):
    CHUNK_SIZE = 30

    def __init__(self, chatbot_model: str = 'gpt-3.5-turbo', fee_limit: float = 0.2, chunk_size: int = CHUNK_SIZE,
                 intercept_line: Optional[int] = None, proxy: Optional[str] = None,
                 base_url_config: Optional[dict] = None,
                 retry_model: Optional[str] = None):
        """
        Initialize the LLMTranslator with given parameters.
        """
        self.chatbot_model = chatbot_model
        self.fee_limit = fee_limit
        self.proxy = proxy
        self.base_url_config = base_url_config
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.retry_model = retry_model

    @staticmethod
    def list_chatbots() -> List[str]:
        """
        List available chatbot models.

        Returns:
            List[str]: List of available chatbot models.
        """
        return list(all_pricing.keys())

    @staticmethod
    def make_chunks(texts: List[str], chunk_size: int = 30) -> List[List[Tuple[int, str]]]:
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

    def _translate_chunk(self, translator_agent: ChunkedTranslatorAgent, chunk: List[Tuple[int, str]],
                         context: TranslationContext, chunk_id: int,
                         retry_agent: Optional[ChunkedTranslatorAgent] = None) -> Tuple[
        List[str], TranslationContext]:
        """
        Translate a single chunk of text.
        """
        translated, context = translator_agent.translate_chunk(chunk_id, chunk, context)

        if len(translated) != len(chunk) and translator_agent.info.glossary:
            logger.warning(f'Cannot translate chunk {chunk_id} with glossary, trying to remove glossary.')
            translated, context = translator_agent.translate_chunk(chunk_id, chunk, context, use_glossary=False)

        if retry_agent and len(translated) != len(chunk):
            logger.warning(
                f'Trying to change chatbot to keep performing chunked translation. Retry chatbot: {retry_agent}')
            translated, context = retry_agent.translate_chunk(chunk_id, chunk, context)

            if len(translated) != len(chunk) and retry_agent.info.glossary:
                logger.warning(f'New bot: Trying to remove glossary to keep performing chunked translation.')
                translated, context = retry_agent.translate_chunk(chunk_id, chunk, context, use_glossary=False)

        return translated, context

    def translate(self, texts: Union[str, List[str]], src_lang: str, target_lang: str,
                  info: TranslateInfo = TranslateInfo(),
                  compare_path: Path = Path('translate_intermediate.json')) -> List[str]:
        """
        Translate a list of texts from source language to target language.
        """
        if not isinstance(texts, list):
            texts = [texts]

        translator_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, self.chatbot_model, self.fee_limit,
                                                  self.proxy, self.base_url_config)

        retry_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, self.retry_model, self.fee_limit,
                                             self.proxy, self.base_url_config) if self.retry_model else None

        # proofreader = ProofreaderAgent(src_lang, target_lang, info)

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)
        logger.info(f'Translating {info.title}: {len(chunks)} chunks, {len(texts)} lines in total.')

        translations, summaries, compare_list, start_chunk, guideline = self._resume_translation(compare_path)
        if not guideline:
            context_reviewer = ContextReviewerAgent(src_lang, target_lang, info)
            guideline = context_reviewer.build_context(texts, title=info.title, glossary=info.glossary)
            logger.info(f'Translation Guideline:\n{guideline}')

        context = TranslationContext(guideline=guideline)
        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            atomic = False
            translated, context = self._translate_chunk(translator_agent, chunk, context, i, retry_agent=retry_agent)
            # Proofreader Not fully tested
            # localized_trans = proofreader.localize_subtitles(
            #     texts=[c[1] for c in chunk], translations=translated, context=context
            # )

            if len(translated) != len(chunk):
                logger.warning(f'Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},'
                               f' Trying to use atomic translation instead.')
                chunk_texts = [item[1] for item in chunk]
                translated = self.atomic_translate(self.chatbot_model, chunk_texts, src_lang, target_lang)
                atomic = True

            translations.extend(translated)
            summaries.append(context.summary)
            logger.info(f'Translated {info.title}: {i}/{len(chunks)}')
            logger.info(f'summary: {context.summary}')
            logger.info(f'scene: {context.scene}')

            compare_list.extend(self._generate_compare_list(chunk, translated, i, atomic, context))
            self._save_intermediate_results(compare_path, compare_list, summaries, context.scene, guideline)

        self.api_fee += translator_agent.cost + (retry_agent.cost if retry_agent else 0)

        return translations

    def _resume_translation(self, compare_path: Path) -> Tuple[List[str], List[str], List[dict], int, str]:
        translations, summaries, compare_list, start_chunk, guideline = [], [], [], 0, ''

        if compare_path.exists():
            logger.info(f'Resume from {compare_path}')
            with open(compare_path, 'r', encoding='utf-8') as f:
                compare_results = json.load(f)
            compare_list = compare_results['compare']
            summaries = compare_results['summaries']
            translations = [item['output'] for item in compare_list]
            start_chunk = compare_list[-1]['chunk']
            guideline = compare_results['guideline']
            logger.info(f'Resume translation from chunk {start_chunk}')

        return translations, summaries, compare_list, start_chunk, guideline

    def _generate_compare_list(self, chunk: List[Tuple[int, str]], translated: List[str], chunk_id: int,
                               atomic: bool, context: TranslationContext) -> List[dict]:
        return [{'chunk': chunk_id,
                 'idx': item[0] if item else 'N\\A',
                 'method': 'atomic' if atomic else 'chunked',
                 'model': context.model,
                 'input': item[1] if item else 'N\\A',
                 'output': trans if trans else 'N\\A'}
                for (item, trans) in zip_longest(chunk, translated)]

    def _save_intermediate_results(self, compare_path: Path, compare_list: List[dict], summaries: List[str],
                                   scene: str, guideline: str):
        compare_results = {'compare': compare_list, 'summaries': summaries, 'scene': scene, 'guideline': guideline}
        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(compare_results, f, indent=4, ensure_ascii=False)

    def atomic_translate(self, chatbot_model: str, texts: List[str], src_lang: str, target_lang: str) -> List[str]:
        """
        Perform atomic translation for each text.
        """
        chatbot = ChunkedTranslatorAgent(src_lang, target_lang, TranslateInfo(), chatbot_model, self.fee_limit,
                                         self.proxy,
                                         self.base_url_config).chatbot

        prompter = AtomicTranslatePrompter(src_lang, target_lang)
        message_lists = [[{'role': 'user', 'content': prompter.user(text)}] for text in texts]

        responses = chatbot.message(message_lists, output_checker=prompter.check_format)
        self.api_fee += sum(chatbot.api_fees[-(len(texts)):])
        translated = list(map(chatbot.get_content, responses))

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

    def translate(self, texts: Union[str, List[str]], src_lang, target_lang, info=None):
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
