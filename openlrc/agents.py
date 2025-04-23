#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import abc
import json
import re
from typing import Optional, Tuple, List, Type, Union

from json_repair import repair_json

from openlrc.chatbot import route_chatbot, GPTBot, ClaudeBot, GeminiBot, provider2chatbot
from openlrc.context import TranslationContext, TranslateInfo
from openlrc.logger import logger
from openlrc.models import ModelConfig, ModelProvider
from openlrc.prompter import ChunkedTranslatePrompter, ContextReviewPrompter, ProofreaderPrompter, PROOFREAD_PREFIX, \
    ContextReviewerValidatePrompter, TranslationEvaluatorPrompter
from openlrc.validators import POTENTIAL_PREFIX_COMBOS


class Agent(abc.ABC):
    """
    Base class for all agents.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """
    TEMPERATURE = 1

    def _initialize_chatbot(self, chatbot_model: Union[str, ModelConfig], fee_limit: float, proxy: str,
                            base_url_config: Optional[dict]) -> Union[ClaudeBot, GPTBot, GeminiBot]:
        """
        Initialize a chatbot instance based on the provided parameters.

        Args:
            chatbot_model (Union[str, ModelConfig]): The name of the chatbot model or ModelConfig.
            fee_limit (float): The maximum fee allowed for API calls.
            proxy (str): Proxy server to use for API calls.
            base_url_config (Optional[dict]): Configuration for the base URL of the API.

        Returns:
            Union[ClaudeBot, GPTBot]: An instance of the appropriate chatbot class.
        """

        if isinstance(chatbot_model, str):
            chatbot_cls: Union[Type[ClaudeBot], Type[GPTBot], Type[GeminiBot]]
            chatbot_cls, model_name = route_chatbot(chatbot_model)
            return chatbot_cls(model_name=model_name, fee_limit=fee_limit, proxy=proxy, retry=4,
                               temperature=self.TEMPERATURE, base_url_config=base_url_config)
        elif isinstance(chatbot_model, ModelConfig):
            chatbot_cls = provider2chatbot[chatbot_model.provider]
            proxy = chatbot_model.proxy or proxy

            if chatbot_model.base_url:
                if chatbot_model.provider == ModelProvider.OPENAI:
                    base_url_config = {'openai': chatbot_model.base_url}
                elif chatbot_model.provider == ModelProvider.ANTHROPIC:
                    base_url_config = {'anthropic': chatbot_model.base_url}
                else:
                    base_url_config = None
                    logger.warning(f'Unsupported base_url configuration for provider: {chatbot_model.provider}')

            return chatbot_cls(model_name=chatbot_model.name, fee_limit=fee_limit, proxy=proxy, retry=4,
                               temperature=self.TEMPERATURE, base_url_config=base_url_config,
                               api_key=chatbot_model.api_key)


class ChunkedTranslatorAgent(Agent):
    """
    Agent responsible for translating well-defined chunked text to the target language.

    This agent uses a chatbot for processing and translating text chunks.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 1.0

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: Union[str, ModelConfig] = 'gpt-4o-mini', fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        """
        Initialize the ChunkedTranslatorAgent.

        Args:
            src_lang (str): The source language.
            target_lang (str): The target language for translation.
            info (TranslateInfo): Additional translation information.
            chatbot_model (Union[str, ModelConfig]): The name of the chatbot model or ModelConfig.
            fee_limit (float): The maximum fee allowed for API calls.
            proxy (str): Proxy server to use for API calls.
            base_url_config (Optional[dict]): Configuration for the base URL of the API.
        """
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

        Raises:
            Exception: If parsing fails.
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
        """
        Extract content enclosed in specified XML-like tags.

        Args:
            content (str): The string to search in.
            tag (str): The tag name to look for.

        Returns:
            str: The content between the specified tags, or an empty string if not found.
        """
        match = re.search(rf'<{tag}>(.*?)</{tag}>', content)
        return match.group(1) if match else ''

    def _extract_translations(self, content: str) -> List[str]:
        """
        Extract translations from the content using predefined prefix combinations.

        Args:
            content (str): The content to extract translations from.

        Returns:
            List[str]: A list of extracted translations.
        """
        for _, trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translations = re.findall(f'{trans_prefix}\n*(.*?)(?:#\d+|<summary>|\n*$)', content, re.DOTALL)
            if translations:
                return self._clean_translations(translations, content)
        return []

    def _clean_translations(self, translations: List[str], content: str) -> List[str]:
        """
        Clean the extracted translations by removing any XML-like tags.

        Args:
            translations (List[str]): The list of translations to clean.
            content (str): The original content for logging purposes.

        Returns:
            List[str]: A list of cleaned translations.
        """
        if any(re.search(r'(<.*?>|</.*?>)', t) for t in translations):
            logger.warning(f'The extracted translation from response contains tags: {content}, tags removed')
            return [re.sub(r'(<.*?>|</.*?>).*', '', t, flags=re.DOTALL) for t in translations]
        return translations

    def translate_chunk(self, chunk_id: int, chunk: List[Tuple[int, str]],
                        context: TranslationContext = TranslationContext(),
                        use_glossary: bool = True) -> Tuple[List[str], TranslationContext]:
        """
        Translate a chunk of text using the chatbot.

        Args:
            chunk_id (int): The ID of the chunk being translated.
            chunk (List[Tuple[int, str]]): The chunk of text to translate.
            context (TranslationContext): The context for translation.
            use_glossary (bool): Whether to use the glossary in the context.

        Returns:
            Tuple[List[str], TranslationContext]: The translated texts and updated context.
        """
        user_input = self.prompter.format_texts(chunk)
        guideline = context.guideline if use_glossary else context.non_glossary_guideline
        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user',
             'content': self.prompter.user(chunk_id, user_input, context.previous_summaries, guideline)},
        ]
        resp = self.chatbot.message(messages_list, output_checker=self.prompter.check_format)[0]
        translations, summary, scene = self._parse_responses(resp)
        self.cost += self.chatbot.api_fees[-1]
        context.update(summary=summary, scene=scene, model=self.chatbot_model)

        return translations, context


class ContextReviewerAgent(Agent):
    """
    Agent responsible for reviewing the context of subtitles to ensure accuracy and completeness.

    TODO: Add chunking support.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 0.6

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: Union[str, ModelConfig] = 'gpt-4o-mini',
                 retry_model: Optional[Union[str, ModelConfig]] = None, fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        """
        Initialize the ContextReviewerAgent.

        Args:
            src_lang (str): The source language.
            target_lang (str): The target language.
            info (TranslateInfo): Additional translation information.
            chatbot_model (Union[str, ModelConfig]): The name or ModelConfig of the primary chatbot model.
            retry_model (Union[str, ModelConfig]): The name or ModelConfig of the backup chatbot model to use for retries.
            fee_limit (float): The maximum fee allowed for API calls.
            proxy (str): Proxy server to use for API calls.
            base_url_config (Optional[dict]): Configuration for the base URL of the API.
        """
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
        """
        Validate the generated context.

        Args:
            context (str): The context to validate.

        Returns:
            bool: True if the context is valid, False otherwise.
        """
        if not context:
            return False

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
        """
        Build the context for translation based on the provided texts and additional information.

        Args:
            texts (List[str]): The texts to build context from.
            title (str): The title of the content.
            glossary (Optional[dict]): A glossary of terms to include in the context.
            forced_glossary (bool): Whether to force the inclusion of the external glossary.

        Returns:
            str: The built context.
        """
        text_content = '\n'.join(texts)

        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(text_content, title=title, given_glossary=glossary)},
        ]

        context = None
        try:
            resp = self.chatbot.message(
                messages_list, stop_sequences=[self.prompter.stop_sequence], output_checker=self.prompter.check_format
            )[0]
            context = self.chatbot.get_content(resp)
        except Exception as e:
            logger.warning(f'Failed to generate context: {e} using {self.chatbot_model}')

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

        if forced_glossary and glossary:
            context = self.add_external_glossary(context, glossary)

        return context

    def add_external_glossary(self, context, glossary: dict) -> str:
        """
        Add an external glossary to the context.

        Args:
            context (str): The existing context.
            glossary (dict): The glossary to add.

        Returns:
            str: The context with the added external glossary.
        """
        glossary_content = '\n'.join([f'- {key}: {value}' for key, value in glossary.items()])
        return f'### External Glossary:\n{glossary_content}\n\n{context}'


class ProofreaderAgent(Agent):
    """
    Agent responsible for proofreading and adapting subtitles to ensure cultural relevance and appropriateness.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """
    TEMPERATURE = 0.8

    def __init__(self, src_lang, target_lang, info: TranslateInfo = TranslateInfo(),
                 chatbot_model: Union[str, ModelConfig] = 'gpt-4o-mini', fee_limit: float = 0.8, proxy: str = None,
                 base_url_config: Optional[dict] = None):
        """
        Initialize the ProofreaderAgent.

        Args:
            src_lang (str): The source language.
            target_lang (str): The target language.
            info (TranslateInfo): Additional translation information.
            chatbot_model (Union[str, ModelConfig]): The name or ModelConfig of the chatbot model to use.
            fee_limit (float): The maximum fee allowed for API calls.
            proxy (str): Proxy server to use for API calls.
            base_url_config (Optional[dict]): Configuration for the base URL of the API.
        """
        super().__init__()
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.info = info
        self.prompter = ProofreaderPrompter(src_lang, target_lang)
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)

    def _parse_responses(self, resp) -> List[str]:
        """
        Parse the proofread responses from the chatbot.

        Args:
            resp: The response from the chatbot.

        Returns:
            List[str]: A list of proofread texts.
        """
        content = self.chatbot.get_content(resp)
        revised = re.findall(PROOFREAD_PREFIX + r'\s*(.*)', content, re.MULTILINE)

        return revised

    def proofread(self, texts: List[str], translations: List[str], context: TranslationContext) -> List[str]:
        """
        Proofread the given texts and translations using the chatbot.

        Args:
            texts (List[str]): The original texts to be proofread.
            translations (List[str]): The translations of the original texts.
            context (TranslationContext): The context information for translation.

        Returns:
            List[str]: A list of proofread and revised translations.

        This method constructs a message list for the chatbot, sends it for proofreading,
        and then parses the response to extract the revised translations.
        """
        messages_list = [
            {'role': 'system', 'content': self.prompter.system()},
            {'role': 'user', 'content': self.prompter.user(texts, translations, context.guideline)},
        ]
        resp = self.chatbot.message(messages_list, output_checker=self.prompter.check_format)[0]
        revised = self._parse_responses(resp)
        return revised


class TranslationEvaluatorAgent(Agent):
    """
    Agent responsible for evaluating translations using a chatbot model.

    This agent evaluates the quality of translations by comparing source texts
    with their corresponding target texts.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 0.95

    def __init__(self, chatbot_model: Union[str, ModelConfig] = 'gpt-4o-mini', fee_limit: float = 0.8,
                 proxy: str = None,
                 base_url_config: Optional[dict] = None):
        """
        Initialize the TranslationEvaluatorAgent.

        Args:
            chatbot_model (Union[str, ModelConfig]): The name of the chatbot model or ModelConfig.
            fee_limit (float): The maximum fee allowed for API calls.
            proxy (str): Proxy server to use for API calls.
            base_url_config (Optional[dict]): Configuration for the base URL of the API.
        """
        super().__init__()
        self.chatbot = self._initialize_chatbot(chatbot_model, fee_limit, proxy, base_url_config)
        self.prompter = TranslationEvaluatorPrompter()

    def evaluate(self, src_texts: List[str], target_texts: List[str]) -> dict:
        """
        Evaluate the quality of translations.

        This method sends the source and target texts to the chatbot for evaluation
        and returns a dictionary containing various quality metrics.

        Args:
            src_texts (List[str]): The original texts in the source language.
            target_texts (List[str]): The translated texts in the target language.

        Returns:
            dict: A dictionary containing evaluation metrics such as accuracy,
                  fluency, completeness, cultural adaptation, and consistency.

        Note:
            The returned dictionary structure depends on the chatbot's response
            and may include additional or different metrics.
        """
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
        # Example of possible metrics in the response:
        # acc = json_resp['accuracy']
        # fluency = json_resp['fluency']
        # completeness = json_resp['completeness']
        # cultural_adaptation = json_resp['cultural adaptation']
        # consistency = json_resp['consistency']

        return json_resp
