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
from openlrc.context import TranslationContext, TranslateInfo
from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.models import ModelConfig
from openlrc.prompter import AtomicTranslatePrompter


class Translator(ABC):

    @abstractmethod
    def translate(self, texts: Union[str, List[str]], src_lang: str, target_lang: str, info: TranslateInfo) \
            -> List[str]:
        pass


class LLMTranslator(Translator):
    """
    Translator using Large Language Models for translation.
    This class implements a sophisticated translation process using chunking,
    context-aware translation, and fallback mechanisms.
    """

    CHUNK_SIZE = 30

    def __init__(self, chatbot_model: Union[str, ModelConfig] = 'gpt-4o-mini', fee_limit: float = 0.8,
                 chunk_size: int = CHUNK_SIZE,
                 intercept_line: Optional[int] = None, proxy: Optional[str] = None,
                 base_url_config: Optional[dict] = None,
                 retry_model: Optional[Union[str, ModelConfig]] = None):
        """
        Initialize the LLMTranslator with given parameters.

        Args:
            chatbot_model (Union[str, ModelConfig]): Name or ModelConfig of the primary chatbot model to use for translation.
            fee_limit (float): Maximum fee limit for API calls to prevent unexpected costs.
            chunk_size (int): Size of text chunks for processing, balancing efficiency and context.
            intercept_line (Optional[int]): Line number to intercept translation, useful for debugging.
            proxy (Optional[str]): Proxy server URL for API calls.
            base_url_config (Optional[dict]): Base URL configuration for API calls.
            retry_model (Optional[Union[str, ModelConfig]]): Model to use for retry attempts if primary model fails.
        """
        self.chatbot_model = chatbot_model
        self.fee_limit = fee_limit
        self.proxy = proxy
        self.base_url_config = base_url_config
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.retry_model = retry_model
        self.use_retry_cnt = 0

    @staticmethod
    def make_chunks(texts: List[str], chunk_size: int = 30) -> List[List[Tuple[int, str]]]:
        """
        Split the text into chunks of specified size for efficient processing.

        Args:
            texts (List[str]): List of texts to be chunked.
            chunk_size (int): Maximum size of each chunk.

        Returns:
            List[List[Tuple[int, str]]]: List of chunks, each chunk is a list of (line_number, text) tuples.
        """
        chunks = []
        start = 1
        for i in range(0, len(texts), chunk_size):
            chunk = [(start + j, text) for j, text in enumerate(texts[i:i + chunk_size])]
            start += len(chunk)
            chunks.append(chunk)

        # Merge the last chunk if it's too small
        if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def _translate_chunk(self, translator_agent: ChunkedTranslatorAgent, chunk: List[Tuple[int, str]],
                         context: TranslationContext, chunk_id: int,
                         retry_agent: Optional[ChunkedTranslatorAgent] = None) -> Tuple[List[str], TranslationContext]:
        """
        Translate a single chunk of text, with retry mechanism.

        This method attempts to translate the chunk using the primary translator agent.
        If the translation fails or is inconsistent, it may use a retry agent or remove the glossary.

        Args:
            translator_agent (ChunkedTranslatorAgent): Primary agent for translation.
            chunk (List[Tuple[int, str]]): Chunk of text to translate.
            context (TranslationContext): Current translation context.
            chunk_id (int): ID of the current chunk.
            retry_agent (Optional[ChunkedTranslatorAgent]): Agent for retry attempts.

        Returns:
            Tuple[List[str], TranslationContext]: Translated texts and updated context.
        """

        def handle_translation(agent: ChunkedTranslatorAgent) -> Tuple[List[str], TranslationContext]:
            trans, updated_context = None, None
            try:
                trans, updated_context = agent.translate_chunk(chunk_id, chunk, context)
            except ChatBotException as e:
                logger.error(f'Failed to translate chunk {chunk_id}.')

            if len(trans) != len(chunk) and agent.info.glossary:
                logger.warning(
                    f'Agent {agent}: Removing glossary for chunk {chunk_id} due to inconsistent translation.')
                try:
                    trans, updated_context = agent.translate_chunk(chunk_id, chunk, context, use_glossary=False)
                except ChatBotException as e:
                    logger.error(f'Failed to translate chunk {chunk_id}.')

            return trans, updated_context

        if self.use_retry_cnt == 0 or not retry_agent:
            translated, context = handle_translation(translator_agent)

            if retry_agent and len(translated) != len(chunk):
                self.use_retry_cnt = 10  # Use retry_agent for the next 10 chunks
                logger.warning(
                    f'Using retry agent {retry_agent} for chunk {chunk_id}, and next {self.use_retry_cnt} chunks.')
                translated, context = handle_translation(retry_agent)
        else:
            logger.info(f'Using retry agent for chunk {chunk_id}, remaining retries: {self.use_retry_cnt}')
            translated, context = handle_translation(retry_agent)
            self.use_retry_cnt -= 1

        if not translated:
            raise ChatBotException(f'Failed to translate chunk {chunk_id}.')

        return translated, context

    def translate(self, texts: Union[str, List[str]], src_lang: str, target_lang: str,
                  info: TranslateInfo = TranslateInfo(),
                  compare_path: Path = Path('translate_intermediate.json')) -> List[str]:
        """
        Translate a list of texts from source language to target language.

        This method implements the main translation process:
        1. Initialize translation agents and chunk the input texts.
        2. Build or load a translation guideline.
        3. Translate each chunk, maintaining context between chunks.
        4. Handle translation failures with retry mechanisms and atomic translation.
        5. Save intermediate results for potential resumption.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.
            info (TranslateInfo): Additional translation information like title and glossary.
            compare_path (Path): Path to save intermediate results for potential resumption.

        Returns:
            List[str]: List of translated texts.
        """
        if not isinstance(texts, list):
            texts = [texts]

        translator_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, self.chatbot_model, self.fee_limit,
                                                  self.proxy, self.base_url_config)

        retry_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, self.retry_model, self.fee_limit,
                                             self.proxy, self.base_url_config) if self.retry_model else None

        # proofreader = ProofreaderAgent(src_lang, target_lang, info, self.chatbot_model, self.fee_limit, self.proxy,
        #                                self.base_url_config)

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)
        logger.info(f'Translating {info.title}: {len(chunks)} chunks, {len(texts)} lines in total.')

        translations, summaries, compare_list, start_chunk, guideline = self._resume_translation(compare_path)
        if not guideline:
            logger.info('Building translation guideline.')
            context_reviewer = ContextReviewerAgent(src_lang, target_lang, info, self.chatbot_model, self.retry_model,
                                                    self.fee_limit, self.proxy, self.base_url_config)
            guideline = context_reviewer.build_context(
                texts, title=info.title, glossary=info.glossary, forced_glossary=info.forced_glossary
            )
            logger.info(f'Translation Guideline:\n{guideline}')

        context = TranslationContext(guideline=guideline)
        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            atomic = False
            translated, context = self._translate_chunk(translator_agent, chunk, context, i, retry_agent=retry_agent)
            chunk_texts = [c[1] for c in chunk]
            # Proofreader Not fully tested
            # localized_trans = proofreader.proofread(
            #     texts=chunk_texts, translations=translated, context=context
            # )

            if len(translated) != len(chunk):
                logger.warning(f'Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},'
                               f' Attempting atomic translation.')
                translated = self.atomic_translate(self.chatbot_model, chunk_texts, src_lang, target_lang)
                atomic = True

            translations.extend(translated)
            summaries.append(context.summary)
            logger.info(f'Translated {info.title}: {i}/{len(chunks)}')
            logger.info(f'Summary: {context.summary}')
            logger.info(f'Scene: {context.scene}')

            compare_list.extend(self._generate_compare_list(chunk, translated, i, atomic, context))
            self._save_intermediate_results(compare_path, compare_list, summaries, context.scene, guideline)

        self.api_fee += translator_agent.cost + (retry_agent.cost if retry_agent else 0)

        return translations

    def _resume_translation(self, compare_path: Path) -> Tuple[List[str], List[str], List[dict], int, str]:
        """
        Resume translation from a saved state.

        This method allows the translation process to be resumed from a previous point,
        which is useful for long translations or in case of interruptions.

        Args:
            compare_path (Path): Path to the saved translation state.

        Returns:
            Tuple[List[str], List[str], List[dict], int, str]: Tuple containing:
                - translations: List of already translated texts.
                - summaries: List of translation summaries.
                - compare_list: List of dictionaries for comparison.
                - start_chunk: The chunk number to resume from.
                - guideline: The translation guideline.
        """
        translations, summaries, compare_list, start_chunk, guideline = [], [], [], 0, ''

        if compare_path.exists():
            logger.info(f'Resuming translation from {compare_path}')
            with open(compare_path, 'r', encoding='utf-8') as f:
                compare_results = json.load(f)
            compare_list = compare_results['compare']
            summaries = compare_results['summaries']
            translations = [item['output'] for item in compare_list]
            start_chunk = compare_list[-1]['chunk']
            guideline = compare_results['guideline']
            logger.info(f'Resuming translation from chunk {start_chunk}')

        return translations, summaries, compare_list, start_chunk, guideline

    def _generate_compare_list(self, chunk: List[Tuple[int, str]], translated: List[str], chunk_id: int,
                               atomic: bool, context: TranslationContext) -> List[dict]:
        """
        Generate a comparison list for the translated chunk.

        This method creates a detailed record of each translation, including the original text,
        translated text, and metadata about the translation process.

        Args:
            chunk (List[Tuple[int, str]]): Original chunk of text.
            translated (List[str]): Translated texts.
            chunk_id (int): ID of the current chunk.
            atomic (bool): Whether atomic translation was used.
            context (TranslationContext): Current translation context.

        Returns:
            List[dict]: List of dictionaries containing comparison information.
        """
        return [{'chunk': chunk_id,
                 'idx': item[0] if item else 'N/A',
                 'method': 'atomic' if atomic else 'chunked',
                 'model': str(context.model),
                 'input': item[1] if item else 'N/A',
                 'output': trans if trans else 'N/A'}
                for (item, trans) in zip_longest(chunk, translated)]

    def _save_intermediate_results(self, compare_path: Path, compare_list: List[dict], summaries: List[str],
                                   scene: str, guideline: str):
        """
        Save intermediate translation results to a file.

        This method saves the current state of the translation process, allowing for
        potential resumption of the translation task later.

        Args:
            compare_path (Path): Path to save the results.
            compare_list (List[dict]): List of comparison dictionaries.
            summaries (List[str]): List of translation summaries.
            scene (str): Current scene description.
            guideline (str): Translation guideline.
        """
        compare_results = {'compare': compare_list, 'summaries': summaries, 'scene': scene, 'guideline': guideline}
        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(compare_results, f, indent=4, ensure_ascii=False)

    def atomic_translate(self, chatbot_model: Union[str, ModelConfig], texts: List[str], src_lang: str,
                         target_lang: str) -> List[str]:
        """
        Perform atomic translation for each text individually.

        This method is used as a fallback when chunk translation fails. It translates
        each text separately, which can be slower but more reliable for problematic texts.

        Args:
            chatbot_model (Union[str, ModelConfig]): Name or ModelConfig of the chatbot model to use.
            texts (List[str]): List of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.

        Returns:
            List[str]: List of translated texts.

        Raises:
            AssertionError: If the number of translated texts doesn't match the input.
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
    """
    Translator using Microsoft Translator API.
    This class provides an alternative translation method using Microsoft's services.
    """

    def __init__(self):
        """
        Initialize the Microsoft Translator with API key and endpoint.
        The API key is expected to be set in the environment variables.
        """
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
