from typing import Type, Union

from langcodes import Language

from openlrc.chatbot import GPTBot
from openlrc.exceptions import SameLanguageException
from openlrc.logger import logger
from openlrc.prompter import BaseTranslatePrompter, prompter_map
from openlrc.utils import json2dict


class Translator:
    def __init__(self, prompter: Union[str, Type[BaseTranslatePrompter]] = BaseTranslatePrompter(), chunk_size=40,
                 fee_limit=0.1, intercept_line=None, force_translate=False):
        """
        :param prompter: Translate prompter.
        :param chunk_size: Use smaller chunk size to avoid exceeding the token limit & output complete message.
        :param fee_limit: Fee limit (USD) for OpenAI API.
        :param intercept_line: Intercepted text line number.
        :param force_translate: Force translation even if the source language is the same as the target language.
        """
        if isinstance(prompter, str):
            if prompter not in prompter_map:
                raise ValueError(f'Prompter {prompter} not found.')
            prompter = prompter_map[prompter]()  # Initialize appropriate prompter

        self.prompter = prompter
        self.chunk_size = chunk_size
        self.fee_limit = fee_limit
        self.intercept_line = intercept_line
        self.force_translate = force_translate

    def _get_system_prompt(self, src_lang, target_lang):
        system_prompt = str(self.prompter).format(
            src_lang=Language.get(src_lang).display_name('en'),
            target_lang=Language.get(target_lang).display_name('en')
        )

        # Prevent translating text into Traditional Chinese
        if target_lang == 'zh-cn':
            system_prompt.replace(Language.get('zh-cn').display_name('en'), 'Mandarin Chinese')

        return system_prompt

    def _make_chunks(self, texts):
        chunks = [texts[i:i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]

        if len(chunks) > 1 and len(chunks[-1]) <= self.chunk_size / 2:
            # Merge the last two chunks if the last chunk is too small
            last_two = chunks[-2] + chunks[-1]

            # Split the merged into 2 equally sized chunks
            chunks[-2] = last_two[:len(last_two) // 2]
            chunks[-1] = last_two[len(last_two) // 2:]

        return chunks

    def translate(self, texts, src_lang, target_lang):
        """
        Use GPT-3.5 to translate texts.
        TODO: dynamically adjust the chunk size.
        :param src_lang: Source language.
        :param target_lang: Target language.
        :param texts: List of texts in src_lang.
        :return: List of texts in target_lang.
        """
        if not self.force_translate and \
                Language.get(src_lang).language_name() == Language.get(target_lang).language_name():
            raise SameLanguageException()

        system_prompt = self._get_system_prompt(src_lang, target_lang)
        translate_bot = GPTBot(fee_limit=self.fee_limit)

        # Split texts into different chunks
        chunks = self._make_chunks(texts)

        user_prompts = [self.prompter.format_texts(chunk) for chunk in chunks]  # Format the chunks into a single string

        logger.info(f'Translating {len(user_prompts)} user_prompts of source texts with async call.')

        messages = [
            [{'role': 'system', 'content': system_prompt},
             {'role': 'user', 'content': user_prompt}] for user_prompt in user_prompts
        ]

        responses = translate_bot.message(messages)
        results = []
        for i, response in enumerate(responses):
            content = response.choices[0].message.content
            logger.debug(f'Target content - chunk{i}: {content}')

            chunk_json_content = json2dict(content)
            logger.debug(f'Length of the translated chunk: {len(chunk_json_content["list"])}')

            chunk_size = len(chunks[i])
            # Helping OpenAI clean up their mess.
            if len(chunk_json_content['list']) < chunk_size:
                logger.warning(f'The number of translated sentences is less than that of the original list. '
                               f'Add {chunk_size - len(chunk_json_content["list"])} <MANUALLY-ADDED> label')
                chunk_json_content['list'] += [' '] * (chunk_size - len(chunk_json_content['list']))
            elif len(chunk_json_content['list']) > chunk_size:
                logger.warning('The number of translated sentences is more than that of the original list. Truncated')
                chunk_json_content['list'] = chunk_json_content['list'][:chunk_size]

            results += chunk_json_content['list']

        results = self.prompter.post_process(results)

        return results
