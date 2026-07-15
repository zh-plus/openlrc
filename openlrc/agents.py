#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import abc
import json
import re
from typing import Any

from json_repair import repair_json

from openlrc.chatbot import ChatBot, GPTBot, provider2chatbot, route_chatbot
from openlrc.context import TranslateInfo, TranslationContext
from openlrc.logger import logger
from openlrc.models import ModelConfig, ModelProvider
from openlrc.prompter import (
    PROOFREAD_PREFIX,
    ChunkedTranslatePrompter,
    ContextReviewerValidatePrompter,
    ContextReviewPrompter,
    ContextReviewPrompterBase,
    ProofreaderPrompter,
    TranslationEvaluatorPrompter,
)
from openlrc.utils import get_text_token_number
from openlrc.validators import POTENTIAL_PREFIX_COMBOS


def create_chatbot(
    chatbot_model: str | ModelConfig,
    fee_limit: float = 0.8,
    proxy: str | None = None,
    base_url_config: dict | None = None,
) -> ChatBot:
    """Create a ChatBot instance from a model name or ModelConfig.

    The caller is responsible for closing the returned ChatBot when done
    (via ``close()`` or a ``with`` statement).
    """
    if isinstance(chatbot_model, str):
        chatbot_cls, model_name = route_chatbot(chatbot_model)
        return chatbot_cls(
            model_name=model_name, fee_limit=fee_limit, proxy=proxy, retry=4, base_url_config=base_url_config
        )
    elif isinstance(chatbot_model, ModelConfig):
        # Resolve chatbot class: known provider or custom provider (defaults to GPTBot)
        chatbot_cls: type[Any] = provider2chatbot.get(chatbot_model.provider, GPTBot)
        proxy = chatbot_model.proxy or proxy

        if chatbot_model.base_url:
            if chatbot_model.provider == ModelProvider.ANTHROPIC:
                base_url_config = {"anthropic": chatbot_model.base_url}
            elif chatbot_model.provider == ModelProvider.GOOGLE:
                logger.warning(
                    f"Google Gemini SDK does not support custom base_url ({chatbot_model.base_url}), "
                    f"ignoring. Use system-level proxy if needed."
                )
                base_url_config = None
            elif chatbot_model.provider == ModelProvider.LITELLM:
                base_url_config = {"litellm": chatbot_model.base_url}
            else:
                # OpenAI, THIRD_PARTY, and custom string providers all use GPTBot compatibility layer
                base_url_config = {"openai": chatbot_model.base_url}

        bot = chatbot_cls(
            model_name=chatbot_model.name,
            temperature=chatbot_model.temperature if chatbot_model.temperature is not None else 1,
            top_p=chatbot_model.top_p if chatbot_model.top_p is not None else 1,
            fee_limit=fee_limit,
            proxy=proxy,
            retry=4,
            base_url_config=base_url_config,
            api_key=chatbot_model.api_key,
            extra_body=chatbot_model.extra_body,
        )

        bot.model_config = chatbot_model

        return bot
    else:
        raise ValueError(f"Invalid chatbot model type: {type(chatbot_model)}. Expected str or ModelConfig.")


class Agent(abc.ABC):
    """
    Base class for all agents.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 1


class ChunkedTranslatorAgent(Agent):
    """
    Agent responsible for translating well-defined chunked text to the target language.

    This agent uses a chatbot for processing and translating text chunks.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 1.0
    OUTPUT_RATIO = 2.5  # Source + target output (#id + Original> + Translation>)

    def __init__(self, src_lang: str, target_lang: str, info: TranslateInfo | None = None, *, chatbot: ChatBot):
        """
        Initialize the ChunkedTranslatorAgent.

        Args:
            src_lang: The source language.
            target_lang: The target language for translation.
            info: Additional translation information.
            chatbot: ChatBot instance to use for LLM calls.
        """
        super().__init__()
        if info is None:
            info = TranslateInfo()
        self.info = info
        self.chatbot = chatbot
        self.chatbot_model = chatbot.model_name
        self.prompter = ChunkedTranslatePrompter(src_lang, target_lang, info)
        self.cost = 0

    def __str__(self):
        return f"Translator Agent ({self.chatbot_model})"

    def _parse_responses(self, resp) -> tuple[list[str], str, str]:
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
            summary = self._extract_tag_content(content, "summary")
            scene = self._extract_tag_content(content, "scene")
            translations = self._extract_translations(content)

            return [t.strip() for t in translations], summary.strip(), scene.strip()
        except Exception:
            logger.error(f"Failed to extract contents from response: {content}")
            raise

    def _extract_tag_content(self, content: str, tag: str) -> str:
        """
        Extract content enclosed in specified XML-like tags.

        Args:
            content (str): The string to search in.
            tag (str): The tag name to look for.

        Returns:
            str: The content between the specified tags, or an empty string if not found.
        """
        match = re.search(rf"<{tag}>(.*?)</{tag}>", content)
        return match.group(1) if match else ""

    def _extract_translations(self, content: str) -> list[str]:
        """
        Extract translations from the content using predefined prefix combinations.

        Args:
            content (str): The content to extract translations from.

        Returns:
            List[str]: A list of extracted translations.
        """
        for _, trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translations = re.findall(rf"{trans_prefix}\n*(.*?)(?:#\d+|<summary>|\n*$)", content, re.DOTALL)
            if translations:
                return self._clean_translations(translations, content)
        return []

    def _clean_translations(self, translations: list[str], content: str) -> list[str]:
        """
        Clean the extracted translations by removing any XML-like tags.

        Args:
            translations (List[str]): The list of translations to clean.
            content (str): The original content for logging purposes.

        Returns:
            List[str]: A list of cleaned translations.
        """
        if any(re.search(r"(<.*?>|</.*?>)", t) for t in translations):
            logger.warning(f"The extracted translation from response contains tags: {content}, tags removed")
            return [re.sub(r"(<.*?>|</.*?>).*", "", t, flags=re.DOTALL) for t in translations]
        return translations

    def _estimate_output_tokens(self, chunk: list[tuple[int, str]]) -> int:
        """Estimate how many output tokens this chunk is likely to require."""
        content_tokens = sum(get_text_token_number(text) for _, text in chunk)
        return max(1024, int(content_tokens * self.OUTPUT_RATIO))

    def translate_chunk(
        self,
        chunk_id: int,
        chunk: list[tuple[int, str]],
        context: TranslationContext | None = None,
        use_glossary: bool = True,
    ) -> tuple[list[str], TranslationContext]:
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
        if context is None:
            context = TranslationContext()
        user_input = self.prompter.format_texts(chunk)
        guideline = context.guideline if use_glossary else context.non_glossary_guideline
        messages_list = [
            {"role": "system", "content": self.prompter.system()},
            {
                "role": "user",
                "content": self.prompter.user(chunk_id, user_input, context.previous_summaries or "", guideline or ""),
            },
        ]
        min_tokens = self._estimate_output_tokens(chunk)
        resp = self.chatbot.message(
            messages_list,
            output_checker=self.prompter.check_format,
            temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
            min_tokens=min_tokens,
        )[0]
        translations, summary, scene = self._parse_responses(resp)
        self.cost += self.chatbot.api_fees[-1]
        context.update(summary=summary, scene=scene, model=self.chatbot_model)

        return translations, context


class ContextReviewerAgent(Agent):
    """
    Agent responsible for reviewing the context of subtitles to ensure accuracy and completeness.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
        MIN_OUTPUT_TOKENS (int): Minimum tokens reserved for LLM output.
        MIN_CHUNK_TEXT_TOKENS (int): Minimum text tokens per chunk for chunked generation.
        MERGE_RETRIES (int): Number of retries for merging partial guidelines.
    """

    TEMPERATURE = 0.6
    MIN_OUTPUT_TOKENS = 1024
    MIN_CHUNK_TEXT_TOKENS = 256
    MERGE_RETRIES = 3

    def __init__(
        self,
        src_lang: str,
        target_lang: str,
        info: TranslateInfo | None = None,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        chunked_guideline: bool = False,
        prompter: ContextReviewPrompterBase | None = None,
    ):
        """
        Initialize the ContextReviewerAgent.

        Args:
            src_lang: The source language.
            target_lang: The target language.
            info: Additional translation information.
            chatbot: ChatBot instance to use for LLM calls.
            retry_chatbot: Optional ChatBot instance for retry attempts.
            chunked_guideline: Enable chunked guideline generation for long texts.
                When False (default), long texts that exceed the context window will fail.
                When True, automatically splits into chunks and merges partial guidelines.
            prompter: Optional prompter instance.  Defaults to
                :class:`ContextReviewPrompter` when not provided.
        """
        super().__init__()
        if info is None:
            info = TranslateInfo()
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.info = info
        self.chatbot = chatbot
        self.chatbot_model = chatbot.model_name
        self.chunked_guideline = chunked_guideline
        self.validate_prompter = ContextReviewerValidatePrompter()
        self.prompter = prompter or ContextReviewPrompter(src_lang, target_lang)
        self.retry_chatbot = retry_chatbot

    def __str__(self):
        return f"Context Reviewer Agent ({self.chatbot_model})"

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
        keywords = self.prompter.expected_sections
        if all(keyword in lowered_context for keyword in keywords):
            return True

        messages_list = [
            {"role": "system", "content": self.validate_prompter.system()},
            {"role": "user", "content": self.validate_prompter.user(context)},
        ]
        resp = self.chatbot.message(
            messages_list,
            stop_sequences=[self.prompter.stop_sequence],
            output_checker=self.validate_prompter.check_format,
            temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
        )[0]
        content = self.chatbot.get_content(resp)
        return bool(content and "true" in content.lower())

    def build_context(self, texts, title="", glossary: dict | None = None, forced_glossary=False) -> str:
        """
        Build the context for translation based on the provided texts and additional information.

        Automatically splits into chunks when the full text would exceed the model's
        context window, generates partial guidelines per chunk, then merges them.

        Args:
            texts (List[str]): The texts to build context from.
            title (str): The title of the content.
            glossary (Optional[dict]): A glossary of terms to include in the context.
            forced_glossary (bool): Whether to force the inclusion of the external glossary.

        Returns:
            str: The built context.
        """
        text_content = "\n".join(texts)

        # Estimate whether the full text fits in a single pass.
        system_tokens = get_text_token_number(self.prompter.system())
        user_tokens = get_text_token_number(self.prompter.user(text_content, title=title, given_glossary=glossary))
        mc = getattr(self.chatbot, "model_config", None)
        context_window = mc.context_window if (mc and mc.context_window) else 131072
        available_output = int(context_window * 0.90) - system_tokens - user_tokens

        if available_output >= self.MIN_OUTPUT_TOKENS:
            try:
                context = self._build_context_single(text_content, title, glossary)
            except Exception as e:
                if self.chunked_guideline:
                    logger.warning(f"Single-pass guideline failed ({e}), falling back to chunked generation.")
                    context = self._build_context_chunked(texts, title, glossary)
                else:
                    logger.error(
                        f"Guideline generation failed: {e}. "
                        f"Consider enabling chunked_guideline=True or using a model with a larger context window."
                    )
                    raise
        elif self.chunked_guideline:
            logger.info(
                f"Input too large for single pass (available_output={available_output}), splitting into chunks."
            )
            context = self._build_context_chunked(texts, title, glossary)
        else:
            logger.warning(
                f"Input too large for context window (available_output={available_output}). "
                f"Enable chunked_guideline=True to handle long texts. Returning empty guideline."
            )
            context = ""

        if forced_glossary and glossary:
            context = self.add_external_glossary(context, glossary)

        return context

    def _build_context_single(self, text_content: str, title: str, glossary: dict | None) -> str:
        """Generate guideline from the full text in a single LLM call.

        Retries only on format validation failures. Deterministic errors
        (e.g. context window exceeded) are re-raised immediately.
        """
        from openlrc.exceptions import ChatBotException

        messages_list = [
            {"role": "system", "content": self.prompter.system()},
            {"role": "user", "content": self.prompter.user(text_content, title=title, given_glossary=glossary)},
        ]

        def _try_generate(bot, label: str) -> str | None:
            """Attempt one LLM call. Returns content or None on format failure. Raises on deterministic errors."""
            try:
                resp = bot.message(
                    messages_list,
                    stop_sequences=[self.prompter.stop_sequence],
                    output_checker=self.prompter.check_format,
                    temperature=bot.agent_temperature(self.TEMPERATURE),
                )[0]
                content = bot.get_content(resp)
                if content:
                    return content.rstrip(self.prompter.stop_sequence)
                return None
            except ChatBotException:
                # Deterministic error (bad request, auth failure, etc.) -- no point retrying.
                raise
            except Exception as e:
                logger.warning(f"Failed to generate context using {label}: {e}")
                return None

        # First attempt.
        context = _try_generate(self.chatbot, str(self.chatbot_model))

        context_pool: list[str] = [context] if context else []
        if not context or not self._validate_context(context):
            validated = False

            # Try retry_chatbot if available.
            if self.retry_chatbot:
                logger.info(f"Failed to validate the context using {self.chatbot}, retrying with {self.retry_chatbot}")
                try:
                    context = _try_generate(self.retry_chatbot, str(self.retry_chatbot))
                except ChatBotException as e:
                    logger.warning(f"Retry chatbot also failed with deterministic error: {e}")
                    context = None
                if context:
                    context_pool.append(context)
                if context and self._validate_context(context):
                    validated = True
                else:
                    logger.warning(f"Failed to validate the context using {self.retry_chatbot}: {context}")

            # Retry with main chatbot on format failures.
            if not validated:
                for i in range(2, 4):
                    logger.warning(f"Retry to generate the context using {self.chatbot} at {i} retries.")
                    context = _try_generate(self.chatbot, str(self.chatbot_model))
                    if context:
                        context_pool.append(context)
                    if context and self._validate_context(context):
                        validated = True
                        break

            if not validated:
                logger.warning(
                    f"Finally failed to validate the context: {context}, you may check the context manually."
                )
                context = max(context_pool, key=len) if context_pool else ""
                logger.debug(f"Now using the longest context: {context}")

        if not context:
            context = ""

        return context

    @staticmethod
    def _split_texts_by_tokens(texts: list[str], max_text_tokens: int) -> list[list[str]]:
        """Split texts into chunks where each chunk's total tokens does not exceed max_text_tokens."""
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for line in texts:
            line_tokens = get_text_token_number(line)
            if current_tokens + line_tokens > max_text_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(line)
            current_tokens += line_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _build_context_chunked(self, texts: list[str], title: str, glossary: dict | None) -> str:
        """Generate guideline by splitting texts into chunks, generating partial guidelines, then merging."""
        # Calculate max text tokens per chunk: context_window - system - user_overhead - output_reserve.
        system_tokens = get_text_token_number(self.prompter.system())
        user_overhead = get_text_token_number(self.prompter.user("", title=title, given_glossary=glossary))
        mc = getattr(self.chatbot, "model_config", None)
        context_window = mc.context_window if (mc and mc.context_window) else 131072
        max_text_tokens = int(context_window * 0.90) - system_tokens - user_overhead - self.MIN_OUTPUT_TOKENS

        if max_text_tokens < self.MIN_CHUNK_TEXT_TOKENS:
            logger.warning(
                "Context window too small even for chunked generation. "
                "Consider using a model with a larger context window."
            )
            return ""

        chunks = self._split_texts_by_tokens(texts, max_text_tokens)
        logger.info(f"Split into {len(chunks)} chunks for guideline generation.")

        # Generate partial guidelines.
        partial_guidelines: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            messages = [
                {"role": "system", "content": self.prompter.system()},
                {
                    "role": "user",
                    "content": self.prompter.user_partial(
                        chunk_text, chunk_index=i + 1, total_chunks=len(chunks), title=title, given_glossary=glossary
                    ),
                },
            ]
            try:
                resp = self.chatbot.message(
                    messages,
                    stop_sequences=[self.prompter.stop_sequence],
                    output_checker=self.prompter.check_format,
                    temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
                )[0]
                content = self.chatbot.get_content(resp)
                if content:
                    partial_guidelines.append(content.rstrip(self.prompter.stop_sequence))
                    logger.info(f"Generated partial guideline {i + 1}/{len(chunks)}.")
            except Exception as e:
                logger.warning(f"Failed to generate partial guideline for chunk {i + 1}: {e}")

        if not partial_guidelines:
            return ""

        # Single chunk succeeded: use it directly.
        if len(partial_guidelines) == 1:
            context = partial_guidelines[0]
        else:
            context = self._merge_guidelines(partial_guidelines, title)

        # Final validation.
        if not self._validate_context(context):
            logger.warning("Chunked guideline failed validation, using best available result.")

        return context or ""

    def _merge_guidelines(self, guidelines: list[str], title: str) -> str:
        """Merge multiple partial guidelines, using hierarchical merging if needed.

        Estimates whether all guidelines fit in a single merge call. If not,
        groups them into pairs, merges each pair, and recurses until one remains.
        """
        merge_system_tokens = get_text_token_number(self.prompter.merge_system())
        mc = getattr(self.chatbot, "model_config", None)
        context_window = mc.context_window if (mc and mc.context_window) else 131072
        max_merge_input = int(context_window * 0.90) - merge_system_tokens - self.MIN_OUTPUT_TOKENS

        user_content = self.prompter.merge_user(guidelines, title=title)
        user_tokens = get_text_token_number(user_content)

        if user_tokens <= max_merge_input:
            # All guidelines fit in one merge call.
            return self._merge_call(guidelines, title)

        # Too large: split into pairs and merge hierarchically.
        logger.info(
            f"Merge input too large ({user_tokens} tokens), merging hierarchically ({len(guidelines)} guidelines)."
        )
        merged: list[str] = []
        for i in range(0, len(guidelines), 2):
            pair = guidelines[i : i + 2]
            if len(pair) == 1:
                merged.append(pair[0])
            else:
                result = self._merge_call(pair, title)
                merged.append(result)

        if len(merged) == 1:
            return merged[0]

        # Recurse until one guideline remains.
        return self._merge_guidelines(merged, title)

    def _merge_call(self, guidelines: list[str], title: str) -> str:
        """Execute a single merge LLM call with retries. Raises on failure."""
        merge_messages = [
            {"role": "system", "content": self.prompter.merge_system()},
            {"role": "user", "content": self.prompter.merge_user(guidelines, title=title)},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.MERGE_RETRIES + 1):
            try:
                resp = self.chatbot.message(
                    merge_messages,
                    stop_sequences=[self.prompter.stop_sequence],
                    output_checker=self.prompter.check_format,
                    temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
                )[0]
                content = self.chatbot.get_content(resp)
                if content:
                    content = content.rstrip(self.prompter.stop_sequence)
                    if self._validate_context(content):
                        return content
                    logger.warning(f"Merge attempt {attempt}: output failed validation, retrying.")
                else:
                    logger.warning(f"Merge attempt {attempt}: empty response, retrying.")
            except Exception as e:
                logger.warning(f"Merge attempt {attempt} failed: {e}")
                last_error = e

        raise RuntimeError(
            f"Failed to merge {len(guidelines)} partial guidelines after {self.MERGE_RETRIES} attempts. "
            f"Consider using a model with a larger context window to avoid chunked generation."
        ) from last_error

    def add_external_glossary(self, context, glossary: dict) -> str:
        """
        Add an external glossary to the context.

        Args:
            context (str): The existing context.
            glossary (dict): The glossary to add.

        Returns:
            str: The context with the added external glossary.
        """
        glossary_content = "\n".join([f"- {key}: {value}" for key, value in glossary.items()])
        return f"### External Glossary:\n{glossary_content}\n\n{context}"


class ProofreaderAgent(Agent):
    """
    Agent responsible for proofreading and adapting subtitles to ensure cultural relevance and appropriateness.

    Attributes:
        TEMPERATURE (float): The temperature setting for the language model.
    """

    TEMPERATURE = 0.8

    def __init__(self, src_lang: str, target_lang: str, info: TranslateInfo | None = None, *, chatbot: ChatBot):
        """
        Initialize the ProofreaderAgent.

        Args:
            src_lang: The source language.
            target_lang: The target language.
            info: Additional translation information.
            chatbot: ChatBot instance to use for LLM calls.
        """
        super().__init__()
        if info is None:
            info = TranslateInfo()
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.info = info
        self.prompter = ProofreaderPrompter(src_lang, target_lang)
        self.chatbot = chatbot

    def _parse_responses(self, resp) -> list[str]:
        """
        Parse the proofread responses from the chatbot.

        Args:
            resp: The response from the chatbot.

        Returns:
            List[str]: A list of proofread texts.
        """
        content = self.chatbot.get_content(resp)
        revised = re.findall(PROOFREAD_PREFIX + r"\s*(.*)", content, re.MULTILINE)

        return revised

    def proofread(self, texts: list[str], translations: list[str], context: TranslationContext) -> list[str]:
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
            {"role": "system", "content": self.prompter.system()},
            {"role": "user", "content": self.prompter.user(texts, translations, context.guideline or "")},
        ]
        resp = self.chatbot.message(
            messages_list,
            output_checker=self.prompter.check_format,
            temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
        )[0]
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

    def __init__(self, *, chatbot: ChatBot):
        """
        Initialize the TranslationEvaluatorAgent.

        Args:
            chatbot: ChatBot instance to use for LLM calls.
        """
        super().__init__()
        self.chatbot = chatbot
        self.prompter = TranslationEvaluatorPrompter()

    def evaluate(self, src_texts: list[str], target_texts: list[str]) -> dict:
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
            {"role": "system", "content": self.prompter.system()},
            {"role": "user", "content": self.prompter.user(src_texts, target_texts)},
        ]
        resp = self.chatbot.message(
            messages_list,
            stop_sequences=[self.prompter.stop_sequence],
            temperature=self.chatbot.agent_temperature(self.TEMPERATURE),
        )[0]
        content = self.chatbot.get_content(resp)

        # Repair potentially broken JSON
        repaired = str(repair_json(content))

        # Returned response should be in JSON format
        json_resp = json.loads(repaired)
        # Example of possible metrics in the response:
        # acc = json_resp['accuracy']
        # fluency = json_resp['fluency']
        # completeness = json_resp['completeness']
        # cultural_adaptation = json_resp['cultural adaptation']
        # consistency = json_resp['consistency']

        return json_resp
