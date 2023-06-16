import json
from typing import Callable

from openlrc.chatbot import GPTBot
from openlrc.logger import logger
from openlrc.prompter import prompter_map


class Translator:
    def __init__(self, prompter: str = 'base_trans', chunk_size=160,
                 fee_limit=0.1, intercept_line=None, force_translate=False):
        """
        :param prompter: Translate prompter, choices can be found in `prompter_map` from prompter.py.
        :param chunk_size: Use smaller chunk size to avoid exceeding the token limit & output complete message.
                Larger chunk size may improve the translation result. However, large chunk size make the response
                of OpenAI API slower and more unstable.
        :param fee_limit: Fee limit (USD) for OpenAI API.
        :param intercept_line: Intercepted text line number.
        :param force_translate: Force translation even if the source language is the same as the target language.
        """
        if prompter not in prompter_map:
            raise ValueError(f'Prompter {prompter} not found.')

        self.prompter = prompter
        self.chunk_size = chunk_size
        self.fee_limit = fee_limit
        self.intercept_line = intercept_line
        self.force_translate = force_translate

    def _make_chunks(self, texts):
        chunks = [texts[i:i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]

        if len(chunks) > 1 and len(chunks[-1]) <= self.chunk_size / 2:
            # Merge the last two chunks if the last chunk is too small
            last_two = chunks[-2] + chunks[-1]

            # Split the merged into 2 equally sized chunks
            chunks[-2] = last_two[:len(last_two) // 2]
            chunks[-1] = last_two[len(last_two) // 2:]

        return chunks

    def parse_responses(self, chunks, responses, post_process_fn: Callable = lambda x: x):
        results = []

        for i, response in enumerate(responses):
            content = response.choices[0].message.content
            logger.debug(f'Target content - chunk{i}: {content}')

            chunk_json_content = json.loads(content)
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

        results = post_process_fn(results)

        return results

    def translate(self, texts, src_lang, target_lang, audio_type='Anime'):
        prompter = prompter_map[self.prompter](src_lang, target_lang, audio_type)
        translate_bot = GPTBot(fee_limit=self.fee_limit)

        chunks = self._make_chunks(texts)

        # Step 1: Revision step in source language
        step1_user_inputs = [prompter.format_texts(chunk) for chunk in chunks]

        logger.info(f'Translating {len(step1_user_inputs)} user_prompts of source texts with async call.')

        # Chunked messages
        step1_messages_list = [
            [{'role': 'system', 'content': prompter.system()},
             {'role': 'user', 'content': prompter.step1(user_input)}]
            for user_input in step1_user_inputs
        ]
        responses = translate_bot.message(step1_messages_list, output_checker=prompter.check_format)
        step1_results = self.parse_responses(chunks, responses, post_process_fn=prompter.post_process)

        logger.debug(f'After {src_lang} revision: {step1_results}')

        # Step 2: Translation step from source language to target language (Keep chat history)
        step2_messages_list = [
            messages + [
                {'role': 'assistant', 'content': responses[i].choices[0].message.content},
                {'role': 'user', 'content': prompter.step2()}
            ] for i, messages in enumerate(step1_messages_list)
        ]
        responses = translate_bot.message(step2_messages_list, output_checker=prompter.check_format)
        step2_results = self.parse_responses(chunks, responses, post_process_fn=prompter.post_process)

        logger.debug(f'After translation: {step2_results}')

        # Step 3: Revision step in target language
        chunks = self._make_chunks(step2_results)
        step3_user_inputs = [prompter.format_texts(chunk) for chunk in chunks]
        step3_messages_list = [
            [{'role': 'system', 'content': prompter.system()},
             {'role': 'user', 'content': prompter.step3(user_input)}]
            for user_input in step3_user_inputs
        ]
        responses = translate_bot.message(step3_messages_list, output_checker=prompter.check_format)
        step3_results = self.parse_responses(chunks, responses, post_process_fn=prompter.post_process)

        logger.debug(f'After {target_lang} revision: {step3_results}')

        return step3_results
