#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import abc
import json
import re
from typing import Optional, Tuple, List, Type, Union

from json_repair import repair_json

from openlrc.chatbot import route_chatbot, GPTBot, ClaudeBot
from openlrc.context import TranslationContext, TranslateInfo
from openlrc.logger import logger
from openlrc.prompter import ChunkedTranslatePrompter, ContextReviewPrompter, ProofreaderPrompter, PROOFREAD_PREFIX, \
    ContextReviewerValidatePrompter, TranslationEvaluatorPrompter
from openlrc.validators import POTENTIAL_PREFIX_COMBOS


class Agent(abc.ABC):
    TEMPERATURE = 1
    """
    Base class for all agents.
    """

    def _initialize_chatbot(self, chatbot_model: str, fee_limit: float, proxy: str, base_url_config: Optional[dict]):
        chatbot_cls: Union[Type[ClaudeBot], Type[GPTBot]]
        chatbot_cls, model_name = route_chatbot(chatbot_model)
        return chatbot_cls(model=model_name, fee_limit=fee_limit, proxy=proxy, retry=3,
                           temperature=self.TEMPERATURE, base_url_config=base_url_config)


class ChunkedTranslatorAgent(Agent):
    """
    Translate the well-defined chunked text to the target language and send it to the chatbot for further processing.
    """

    TEMPERATURE = 1.0

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: str = 'gpt-4o-mini', fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        super().__init__()
        self.chatbot_model = chatbot_model
        self.info = info
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)
        self.prompter = ChunkedTranslatePrompter(src_lang, target_lang, info)
        self.cost = 0

    def __str__(self):
        return f'Translator Agent ({self.chatbot_model})'

    def _parse_responses(self, resp) -> Tuple[List[str], str, str]:
        """
        Parse the response from the chatbot API.

        Args:
            resp: The response from the chatbot API.

        Returns:
            Tuple[List[str], str, str]: Parsed translations, summary, and scene from the response.
        """
        content = self.chatbot.get_content(resp)

        try:
            summary = self._extract_tag_content(content, 'summary')
            scene = self._extract_tag_content(content, 'scene')
            translations = self._extract_translations(content)

            return [t.strip() for t in translations], summary.strip(), scene.strip()
        except Exception as e:
            logger.error(f'Failed to extract contents from response: {content}')
            raise e

    def _extract_tag_content(self, content: str, tag: str) -> str:
        match = re.search(rf'<{tag}>(.*?)</{tag}>', content)
        return match.group(1) if match else ''

    def _extract_translations(self, content: str) -> List[str]:
        for _, trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translations = re.findall(f'{trans_prefix}\n*(.*?)(?:#\d+|<summary>|\n*$)', content, re.DOTALL)
            if translations:
                return self._clean_translations(translations, content)
        return []

    def _clean_translations(self, translations: List[str], content: str) -> List[str]:
        if any(re.search(r'(<.*?>|</.*?>)', t) for t in translations):
            logger.warning(f'The extracted translation from response contains tags: {content}, tags removed')
            return [re.sub(r'(<.*?>|</.*?>).*', '', t, flags=re.DOTALL) for t in translations]
        return translations

    def translate_chunk(self, chunk_id: int, chunk: List[Tuple[int, str]],
                        context: TranslationContext = TranslationContext(),
                        use_glossary: bool = True) -> Tuple[List[str], TranslationContext]:
        user_input = self.prompter.format_texts(chunk)
        guideline = context.guideline if use_glossary else context.non_glossary_guideline
        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(chunk_id, user_input, context.summary, guideline)},
        ]
        resp = self.chatbot.message(messages_list, output_checker=self.prompter.check_format)[0]
        translations, summary, scene = self._parse_responses(resp)
        self.cost += self.chatbot.api_fees[-1]
        context.update(summary=summary, scene=scene, model=self.chatbot_model)

        return translations, context


class ContextReviewerAgent(Agent):
    """
    Review the context of the subtitles to ensure accuracy and completeness.
    TODO: Add chunking support.
    """

    TEMPERATURE = 0.6

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: str = 'gpt-4o-mini', retry_model=None,
                 fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        super().__init__()
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.info = info
        self.chatbot_model = chatbot_model
        self.validate_prompter = ContextReviewerValidatePrompter()
        self.prompter = ContextReviewPrompter(src_lang, target_lang)
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)
        self.retry_chatbot = self._initialize_chatbot(
            retry_model, fee_limit, proxy, base_url_config
        ) if retry_model else None

    def __str__(self):
        return f'Context Reviewer Agent ({self.chatbot_model})'

    def _validate_context(self, context: str) -> bool:
        # Use the content to check first
        lowered_context = context.lower()
        keywords = ['glossary', 'characters', 'summary', 'tone and style', 'target audience']
        if all(keyword in lowered_context for keyword in keywords):
            return True

        messages_list = [
            {'role': 'system', 'content': self.validate_prompter.system()},
            {'role': 'user', 'content': self.validate_prompter.user(context)},
        ]
        resp = self.chatbot.message(messages_list, output_checker=self.validate_prompter.check_format)[0]
        return 'true' in self.chatbot.get_content(resp).lower()

    def build_context(self, texts, title='', glossary: Optional[dict] = None, forced_glossary=False) -> str:
        text_content = '\n'.join(texts)

        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(text_content, title=title, given_glossary=glossary)},
        ]
        resp = self.chatbot.message(
            messages_list, stop_sequences=[self.prompter.stop_sequence], output_checker=self.prompter.check_format
        )[0]
        context = self.chatbot.get_content(resp)

        context_pool = [context]
        # Validate
        if not self._validate_context(context):
            validated = False
            if self.retry_chatbot:
                logger.info(f'Failed to validate the context using {self.chatbot}, retrying with {self.retry_chatbot}')
                resp = self.retry_chatbot.message(messages_list, output_checker=self.validate_prompter.check_format)[0]
                context = self.retry_chatbot.get_content(resp)
                context_pool.append(context)
                if self._validate_context(context):
                    validated = True
                else:
                    logger.warning(f'Failed to validate the context using {self.retry_chatbot}: {context}')

            if not validated:
                for i in range(2, 4):
                    logger.warning(f'Retry to generate the context using {self.chatbot} at {i} reties.')
                    resp = self.chatbot.message(messages_list, output_checker=self.validate_prompter.check_format)[0]
                    context = self.chatbot.get_content(resp)
                    context_pool.append(context)
                    if self._validate_context(context):
                        validated = True
                        break

            if not validated:
                logger.warning(
                    f'Finally failed to validate the context: {context}, you may check the context manually.')
                context = max(context_pool, key=len)
                logger.info(f'Now using the longest context: {context}')

        if forced_glossary:
            context = self.add_external_glossary(context, glossary)

        return context

    def add_external_glossary(self, context, glossary: dict) -> str:
        glossary_content = '\n'.join([f'- {key}: {value}' for key, value in glossary.items()])
        return f'### External Glossary:\n{glossary_content}\n\n{context}'


class ProofreaderAgent(Agent):
    """
    Adapt subtitles to ensure cultural relevance and appropriateness.
    """
    TEMPERATURE = 0.8

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: str = 'gpt-4o-mini', fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        super().__init__()
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.info = info
        self.prompter = ProofreaderPrompter(src_lang, target_lang)
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)

    def _parse_responses(self, resp) -> List[str]:
        content = self.chatbot.get_content(resp)
        revised = re.findall(PROOFREAD_PREFIX + r'\s*(.*)', content, re.MULTILINE)

        return revised

    def proofread(self, texts: List[str], translations, context: TranslationContext) -> List[str]:
        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(texts, translations, context.guideline)},
        ]
        resp = self.chatbot.message(messages_list, output_checker=self.prompter.check_format)[0]
        revised = self._parse_responses(resp)
        return revised


class TranslationEvaluatorAgent(Agent):
    TEMPERATURE = 0.95

    def __init__(self, chatbot_model: str = 'gpt-4o-mini', fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        super().__init__()
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)
        self.prompter = TranslationEvaluatorPrompter()

    def evaluate(self, src_texts, target_texts) -> dict:
        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(src_texts, target_texts)},
        ]
        resp = self.chatbot.message(messages_list, stop_sequences=[self.prompter.stop_sequence])[0]
        content = self.chatbot.get_content(resp)

        # Repair potentially broken JSON
        content = repair_json(content)

        # Returned response should be in JSON format
        json_resp = json.loads(content)
        # acc = json_resp['accuracy']
        # fluency = json_resp['fluency']
        # completeness = json_resp['completeness']
        # cultural_adaptation = json_resp['cultural adaptation']
        # consistency = json_resp['consistency']

        return json_resp
