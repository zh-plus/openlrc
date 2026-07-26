#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import os
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import anthropic
import httpx
import openai
from anthropic import Anthropic
from anthropic._types import omit
from anthropic.types import Message
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory
from openai import OpenAI
from openai.types.chat import ChatCompletion

from openlrc.exceptions import ChatBotException, LengthExceedException
from openlrc.logger import logger
from openlrc.models import ModelInfo, ModelProvider, Models
from openlrc.utils import get_messages_token_number, get_text_token_number, remove_stop

# The default mapping for model name to chatbot class.
model2chatbot = {}


def _register_chatbot(cls):
    # Get model info from Models class
    for model in Models.__dict__.values():
        if not isinstance(model, ModelInfo):
            continue

        if (
            model.provider in (ModelProvider.OPENAI, ModelProvider.THIRD_PARTY, ModelProvider.LOCAL_LLAMA)
            and cls.__name__ == "GPTBot"
        ):
            model2chatbot[model.name] = cls
            if model.latest_alias:
                model2chatbot[model.latest_alias] = cls
        elif model.provider == ModelProvider.ANTHROPIC and cls.__name__ == "ClaudeBot":
            model2chatbot[model.name] = cls
            if model.latest_alias:
                model2chatbot[model.latest_alias] = cls
        elif model.provider == ModelProvider.GOOGLE and cls.__name__ == "GeminiBot":
            model2chatbot[model.name] = cls
            if model.latest_alias:
                model2chatbot[model.latest_alias] = cls
    return cls


def route_chatbot(model: str) -> tuple[type, str]:
    if ":" in model:
        match = re.match(r"(.+):(.+)", model)
        if match is None:
            raise ValueError(f"Invalid model format: {model!r}")
        chatbot_type, chatbot_model = match.groups()
        chatbot_type, chatbot_model = chatbot_type.strip().lower(), chatbot_model.strip()

        Models.get_model(chatbot_model)

        if chatbot_type == "openai":
            return GPTBot, chatbot_model
        elif chatbot_type == "anthropic":
            return ClaudeBot, chatbot_model
        elif chatbot_type == "litellm":
            return LiteLLMBot, chatbot_model
        else:
            raise ValueError(f"Invalid chatbot type {chatbot_type}.")

    if model not in model2chatbot:
        raise ValueError(f"Invalid model {model}.")

    return model2chatbot[model], model


class ChatBot:
    def __init__(
        self,
        model_name: str,
        temperature: float = 1,
        top_p: float = 1,
        retry: int = 8,
        max_async: int = 16,
        fee_limit: float = 0.8,
        beta: bool = False,
    ):
        try:
            self.model_info = Models.get_model(model_name, beta)
            self.model_name = model_name
        except ValueError:
            raise ValueError(f"Invalid model {model_name}.") from None

        # Default temperature/top_p used when message() is called without explicit values.
        self.temperature = temperature
        self.top_p = top_p
        self.retry = retry
        self.max_async = max_async
        self.fee_limit = fee_limit

        self.api_fees = []
        self.cancellation_token = None

    def _check_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def _sleep(self, seconds: float) -> None:
        if self.cancellation_token is None:
            time.sleep(seconds)
        else:
            self.cancellation_token.wait_or_raise(seconds)

    def _compute_max_tokens(self, messages: list[dict], min_tokens: int | None = None) -> int | None:
        """Return ``max_tokens`` to send to the API, or ``None`` to let the server decide.

        When ``model_config`` has ``max_tokens`` set (user opt-in), compute a safe
        cap that respects the context window and the caller's ``min_tokens`` estimate.
        When unset (the default), return ``None`` — the API call omits ``max_tokens``
        and the server uses its own limit.

        When ``min_tokens`` is ``None`` (no caller estimate), the full budget
        (``model_config.max_tokens`` constrained by context window) is returned.
        When set, the result is clamped to at most ``min_tokens`` (cost control),
        with a warning if the estimate exceeds the budget.
        """
        model_config = getattr(self, "model_config", None)
        if model_config is None or model_config.max_tokens is None:
            return None

        input_tokens = get_messages_token_number(messages)
        if model_config.context_window is not None:
            available = int(model_config.context_window * 0.90) - input_tokens
            upper = min(model_config.max_tokens, available)
        else:
            upper = model_config.max_tokens

        if min_tokens is not None:
            if min_tokens > upper:
                logger.warning(
                    f"Estimated output need ({min_tokens}) exceeds generation budget ({upper}). "
                    f"The translation may be truncated. "
                    f"(input={input_tokens}, max_tokens={model_config.max_tokens})"
                )
            result = min(upper, max(min_tokens, 1024))
        else:
            result = upper

        if result < model_config.max_tokens:
            logger.debug(
                f"Dynamic max_tokens: {result} "
                f"(input={input_tokens}, max_tokens={model_config.max_tokens}, "
                f"min_tokens={min_tokens})"
            )

        return result

    def estimate_fee(self, messages: list[dict]):
        """
        Estimate the total fee for the given messages.
        """
        token_map = {"system": 0, "user": 0, "assistant": 0}
        for message in messages:
            token_map[message["role"]] += get_text_token_number(message["content"])

        input_price = self.model_info.input_price
        output_price = self.model_info.output_price

        total_price = (sum(token_map.values()) * input_price + token_map["user"] * output_price * 2) / 1000000

        return total_price

    def update_fee(self, response):
        raise NotImplementedError()

    def get_content(self, response):
        raise NotImplementedError()

    def _create_chat(
        self,
        messages: list[dict],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        raise NotImplementedError()

    def message(
        self,
        messages_list: list[dict] | list[list[dict]],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        """
        Send chunked messages to the chatbot.

        Args:
            messages_list: A single message list or a list of message lists for batch processing.
            stop_sequences: Optional stop sequences to terminate generation.
            output_checker: Callable to validate the generated output.
            temperature: Sampling temperature for this call. Falls back to the instance default if None.
            top_p: Top-p sampling for this call. Falls back to the instance default if None.
            min_tokens: Estimated output tokens this chunk requires. ``None`` = no estimate,
                ``_compute_max_tokens`` returns the full budget. When set, used for cost
                control (``max_tokens`` is clamped to at most this value).
        """
        if not messages_list:
            raise ValueError("Empty message list.")
        self._check_cancelled()

        # Normalise to list[list[dict]] so downstream code has a single type.
        normalised: list[list[dict]]
        if isinstance(messages_list[0], dict):
            normalised = [messages_list]  # type: ignore[list-item]
        else:
            normalised = messages_list  # type: ignore[assignment]

        # Calculate the total sending token number and approximated billing fee.
        token_numbers = [get_messages_token_number(message) for message in normalised]
        logger.debug(
            f"Max token num: {max(token_numbers):.0f}, Avg token num: {sum(token_numbers) / len(token_numbers):.0f}"
        )

        # if the approximated billing fee exceeds the limit, raise an exception.
        approximated_fee = sum([self.estimate_fee(messages) for messages in normalised])
        logger.debug(f"Approximated billing fee: {approximated_fee:.4f} USD")
        self.api_fees += [0]  # Actual fee for this translation call.
        if approximated_fee > self.fee_limit:
            raise ChatBotException(f"Approximated billing fee {approximated_fee} exceeds the limit: {self.fee_limit}$.")

        try:
            with ThreadPoolExecutor(max_workers=min(len(normalised), self.max_async)) as pool:
                futures = [
                    pool.submit(
                        self._create_chat,
                        message,
                        stop_sequences=stop_sequences,
                        output_checker=output_checker,
                        temperature=temperature,
                        top_p=top_p,
                        min_tokens=min_tokens,
                    )
                    for message in normalised
                ]
                results = []
                for future in futures:
                    results.append(future.result())
                    # Never start a later pipeline call after an in-flight SDK
                    # request has returned and cancellation was requested.
                    self._check_cancelled()
        except ChatBotException as e:
            logger.error(f"Failed to message with GPT. Error: {e}")
            raise
        finally:
            logger.debug(f"Translation fee for this call: {self.api_fees[-1]:.4f} USD")
            logger.debug(f"Total bot translation fee: {sum(self.api_fees):.4f} USD")

        return results

    def agent_temperature(self, fallback: float) -> float | None:
        """Return an agent fallback temperature unless model config owns sampling."""
        model_config = getattr(self, "model_config", None)
        if model_config is not None and (model_config.temperature is not None or model_config.top_p is not None):
            return None
        return fallback

    def close(self):
        """Close the underlying HTTP client and release resources.

        Safe to call multiple times. Subclasses should override to clean up
        provider-specific clients.
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self):
        return f"ChatBot ({self.model_name})"


@_register_chatbot
class GPTBot(ChatBot):
    # Keys in ``extra_body`` that map directly to native OpenAI API parameters.
    # They are extracted from ``extra_body`` and passed as top-level kwargs to
    # ``client.chat.completions.create()``.  Everything else is forwarded via
    # the SDK's ``extra_body`` kwarg (which adds them verbatim to the HTTP
    # request body — useful for vLLM / custom endpoint parameters like
    # ``repetition_penalty``, ``top_k``, or ``chat_template_kwargs``).
    _NATIVE_EXTRA_KEYS = frozenset({"frequency_penalty", "presence_penalty", "seed"})

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        temperature: float = 1,
        top_p: float = 1,
        retry: int = 8,
        max_async: int = 16,
        json_mode: bool = False,
        fee_limit: float = 0.05,
        proxy: str | None = None,
        base_url_config: dict | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
    ):
        # clamp temperature to 0-2
        temperature = max(0, min(2, temperature))

        is_beta = False
        if base_url_config and base_url_config["openai"] == "https://api.deepseek.com/beta":
            is_beta = True

        super().__init__(model_name, temperature, top_p, retry, max_async, fee_limit, is_beta)

        resolved_api_key, resolved_base_url = self._resolve_client_settings(
            api_key=api_key, base_url_config=base_url_config
        )

        self.client = OpenAI(
            api_key=resolved_api_key, http_client=httpx.Client(proxy=proxy), base_url=resolved_base_url
        )

        self.model_name = model_name
        self.json_mode = json_mode
        self.extra_body = dict(extra_body) if extra_body else {}

    @staticmethod
    def _resolve_client_settings(api_key: str | None, base_url_config: dict | None) -> tuple[str, str | None]:
        """
        Resolve API key and base URL, auto-switching to OpenRouter when needed.
        """
        openai_base_url = None
        if base_url_config:
            openai_base_url = base_url_config.get("openai")

        if api_key:
            return api_key, openai_base_url

        is_openrouter = bool(openai_base_url and "openrouter.ai" in openai_base_url.lower())
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

        if is_openrouter:
            if openrouter_api_key:
                return openrouter_api_key, openai_base_url
            if openai_api_key:
                return openai_api_key, openai_base_url
            raise ValueError(
                "OPENROUTER_API_KEY is required when using OpenRouter base_url. "
                "Set OPENROUTER_API_KEY or pass api_key explicitly."
            )

        if openai_api_key:
            return openai_api_key, openai_base_url

        if openrouter_api_key:
            logger.info("OPENAI_API_KEY not found, fallback to OPENROUTER_API_KEY with OpenRouter endpoint.")
            return openrouter_api_key, "https://openrouter.ai/api/v1"

        raise ValueError("No API key found. Set OPENAI_API_KEY or pass api_key explicitly.")

    def update_fee(self, response: ChatCompletion):
        if response.usage is None:
            logger.warning("ChatCompletion.usage is None, skipping fee update.")
            return
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        self.api_fees[-1] += (
            prompt_tokens * self.model_info.input_price + completion_tokens * self.model_info.output_price
        ) / 1000000

    def get_content(self, response):
        return response.choices[0].message.content

    def _create_chat(
        self,
        messages: list[dict],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        # Check stop sequences
        if stop_sequences and len(stop_sequences) > 4:
            logger.warning("Too many stop sequences. For openai, Only the first 4 will be used.")
            stop_sequences = stop_sequences[:4]

        # Fall back to instance defaults when not specified per-call.
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_top_p = top_p if top_p is not None else self.top_p

        # Build the kwargs dict from extra_body.
        # Native OpenAI keys become top-level kwargs; the rest go into extra_body.
        native_kwargs: dict[str, Any] = {}
        passthrough: dict = {}
        for key, value in self.extra_body.items():
            if key in self._NATIVE_EXTRA_KEYS:
                native_kwargs[key] = value
            else:
                passthrough[key] = value

        max_tokens = self._compute_max_tokens(messages, min_tokens=min_tokens)

        response = None
        validated = False
        for i in range(self.retry):
            self._check_cancelled()
            try:
                create_kwargs: dict = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": effective_temperature,
                    "top_p": effective_top_p,
                    "response_format": {"type": "json_object" if self.json_mode else "text"},
                    "stop": stop_sequences,
                    **native_kwargs,
                }
                if max_tokens is not None:
                    create_kwargs["max_tokens"] = max_tokens
                if passthrough:
                    create_kwargs["extra_body"] = passthrough

                response = self.client.chat.completions.create(**create_kwargs)  # pyright: ignore[reportArgumentType]
                self.update_fee(response)
                self._check_cancelled()
                if response.choices[0].finish_reason == "length":
                    usage = response.usage
                    raise LengthExceedException(
                        prompt_tokens=usage.prompt_tokens if usage else -1,
                        completion_tokens=usage.completion_tokens if usage else -1,
                        total_tokens=usage.total_tokens if usage else -1,
                    )

                response_text = remove_stop(self.get_content(response), stop_sequences)

                if not output_checker(messages[-1]["content"], response_text):
                    logger.warning(f"Invalid response format. Retry num: {i + 1}.")
                    continue

                validated = True
                break
            except openai.AuthenticationError as e:
                # Authentication errors are deterministic and should not be retried.
                raise ChatBotException(f"Authentication failed: {e}") from e
            except (
                openai.BadRequestError,
                openai.NotFoundError,
                openai.PermissionDeniedError,
                openai.ConflictError,
                openai.UnprocessableEntityError,
            ) as e:
                # Client errors are deterministic (e.g. context window exceeded, invalid model),
                # retrying will not change the outcome.
                raise ChatBotException(f"Client error: {e}") from e
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError,
                json.decoder.JSONDecodeError,
            ) as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f"{type(e).__name__}: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.")
                self._sleep(sleep_time)

        if not response:
            raise ChatBotException("Failed to create a chat.")

        if not validated:
            logger.warning("Response format validation failed after all retries, returning best-effort response.")

        return response

    @staticmethod
    def _get_sleep_time(error):
        if isinstance(error, openai.RateLimitError):
            return random.randint(30, 60)
        elif isinstance(error, openai.APITimeoutError):
            return 3
        elif isinstance(error, json.decoder.JSONDecodeError):
            return 1
        else:
            return 15

    def close(self):
        """Close the OpenAI client and its underlying HTTP connection pool."""
        self.client.close()


@_register_chatbot
class ClaudeBot(ChatBot):
    # Keys in ``extra_body`` that map to native Anthropic Messages API parameters.
    _NATIVE_EXTRA_KEYS = frozenset({"top_k"})

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 1,
        top_p: float = 1,
        retry: int = 8,
        max_async: int = 16,
        fee_limit: float = 0.8,
        proxy: str | None = None,
        base_url_config: dict | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
    ):
        # clamp temperature to 0-1
        temperature = max(0, min(1, temperature))

        super().__init__(model_name, temperature, top_p, retry, max_async, fee_limit)

        self.client = Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
            http_client=httpx.Client(proxy=proxy),
            base_url=base_url_config["anthropic"] if base_url_config and base_url_config["anthropic"] else None,
        )

        self.model_name = model_name
        self.extra_body = dict(extra_body) if extra_body else {}

    def update_fee(self, response: Message):
        model_info = self.model_info

        if response.usage is None:
            logger.warning("Message.usage is None, skipping fee update.")
            return
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        self.api_fees[-1] += (
            prompt_tokens * model_info.input_price + completion_tokens * model_info.output_price
        ) / 1000000

    def get_content(self, response):
        return response.content[0].text

    def _create_chat(
        self,
        messages: list[dict],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        # No need to check stop sequences for Claude (unlimited)

        # Compute max_tokens before popping system message from the list.
        # Claude API requires max_tokens; use generous default when no user cap.
        max_tokens = self._compute_max_tokens(messages, min_tokens=min_tokens) or 8192

        # Fall back to instance defaults when not specified per-call.
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_top_p = top_p if top_p is not None else self.top_p

        # Move "system" role into the parameters
        system_msg = omit
        messages = list(messages)  # Shallow copy to avoid mutating the caller's list.
        if messages[0]["role"] == "system":
            system_msg = messages.pop(0)["content"]

        # Extract native Anthropic keys from extra_body.
        native_kwargs: dict = {}
        for key, value in self.extra_body.items():
            if key in self._NATIVE_EXTRA_KEYS:
                native_kwargs[key] = value
            else:
                logger.debug(f"ClaudeBot: ignoring unsupported extra_body key: {key!r}")

        response = None
        validated = False
        for i in range(self.retry):
            self._check_cancelled()
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "system": system_msg,
                    "stop_sequences": stop_sequences or omit,
                    "max_tokens": max_tokens,
                    **native_kwargs,
                }
                if effective_temperature is not None:
                    request_kwargs["temperature"] = effective_temperature
                if effective_top_p is not None:
                    request_kwargs["top_p"] = effective_top_p

                response = self.client.messages.create(**request_kwargs)
                self.update_fee(response)
                self._check_cancelled()

                if response.stop_reason == "max_tokens":
                    usage = response.usage
                    raise LengthExceedException(
                        prompt_tokens=usage.input_tokens if usage else -1,
                        completion_tokens=usage.output_tokens if usage else -1,
                        total_tokens=(usage.input_tokens + usage.output_tokens) if usage else -1,
                    )

                response_text = remove_stop(self.get_content(response), stop_sequences)

                if not output_checker(messages[-1]["content"], response_text):
                    logger.warning(f"Invalid response format. Retry num: {i + 1}.")
                    continue

                validated = True
                break
            except anthropic.AuthenticationError as e:
                # Authentication errors are deterministic and should not be retried.
                raise ChatBotException(f"Authentication failed: {e}") from e
            except (
                anthropic.BadRequestError,
                anthropic.NotFoundError,
                anthropic.PermissionDeniedError,
                anthropic.ConflictError,
                anthropic.UnprocessableEntityError,
            ) as e:
                # Client errors are deterministic (e.g. context window exceeded, invalid model),
                # retrying will not change the outcome.
                raise ChatBotException(f"Client error: {e}") from e
            except (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.APIError,
            ) as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f"{type(e).__name__}: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.")
                self._sleep(sleep_time)

        if not response:
            raise ChatBotException("Failed to create a chat.")

        if not validated:
            logger.warning("Response format validation failed after all retries, returning best-effort response.")

        return response

    def _get_sleep_time(self, error):
        if isinstance(error, anthropic.RateLimitError):
            return random.randint(30, 60)
        elif isinstance(error, anthropic.APITimeoutError):
            return 3
        else:
            return 15

    def close(self):
        """Close the Anthropic client and its underlying HTTP connection pool."""
        self.client.close()


@_register_chatbot
class GeminiBot(ChatBot):
    # Keys in ``extra_body`` that map to native Gemini GenerateContentConfig fields.
    _NATIVE_EXTRA_KEYS = frozenset({"top_k", "seed", "presence_penalty", "frequency_penalty"})

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-04-17",
        temperature: float = 1,
        top_p: float = 1,
        retry: int = 8,
        max_async: int = 16,
        fee_limit: float = 0.8,
        proxy: str | None = None,
        base_url_config: dict | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
    ):
        # clamp temperature to 0-1
        temperature = max(0, min(1, temperature))

        super().__init__(model_name, temperature, top_p, retry, max_async, fee_limit)

        self.model_name = model_name
        self.extra_body = dict(extra_body) if extra_body else {}

        # genai.configure(api_key=api_key or os.environ['GOOGLE_API_KEY'])
        self.client = genai.Client(api_key=api_key or os.environ["GOOGLE_API_KEY"])

        # Should not block any translation-related content.
        self.safety_settings = [
            types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE
            ),
        ]

        if proxy:
            logger.warning("Google Gemini SDK does not support proxy, try using the system-level proxy if needed.")

        if base_url_config:
            logger.warning("Google Gemini SDK does not support changing base_url.")

    def update_fee(self, response: types.GenerateContentResponse):
        model_info = self.model_info
        if response.usage_metadata is None:
            logger.warning("GenerateContentResponse.usage_metadata is None, skipping fee update.")
            return
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0

        self.api_fees[-1] += (
            prompt_tokens * model_info.input_price + completion_tokens * model_info.output_price
        ) / 1000000

    def get_content(self, response):
        return response.text

    def _create_chat(
        self,
        messages: list[dict],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        # Check stop sequences
        if stop_sequences and len(stop_sequences) > 5:
            logger.warning("Too many stop sequences. Only the first 5 will be used.")
            stop_sequences = stop_sequences[:5]

        # Fall back to instance defaults when not specified per-call.
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_top_p = top_p if top_p is not None else self.top_p

        history_messages = deepcopy(messages)
        system_msg = None
        if history_messages[0]["role"] == "system":
            system_msg = history_messages.pop(0)["content"]

        if history_messages[-1]["role"] != "user":
            logger.error("The last message should be user message.")
        user_msg = history_messages.pop(-1)["content"]

        # convert assistant role into model
        for i, message in enumerate(history_messages):
            if message["role"] == "assistant":
                history_messages[i]["role"] = "model"

            content = message.pop("content")
            history_messages[i]["parts"] = [{"text": content}]

        # Extract native Gemini keys from extra_body.
        native_kwargs: dict = {}
        for key, value in self.extra_body.items():
            if key in self._NATIVE_EXTRA_KEYS:
                native_kwargs[key] = value
            else:
                logger.debug(f"GeminiBot: ignoring unsupported extra_body key: {key!r}")

        max_output_tokens = self._compute_max_tokens(messages, min_tokens=min_tokens)

        config_kwargs = {}
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        # Build config per-call so temperature/top_p can vary across callers sharing this bot.
        config = types.GenerateContentConfig(
            temperature=effective_temperature,
            top_p=effective_top_p,
            safety_settings=self.safety_settings,
            stop_sequences=stop_sequences,
            system_instruction=system_msg,
            **config_kwargs,
            **native_kwargs,
        )

        response = None
        validated = False
        for i in range(self.retry):
            self._check_cancelled()
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=user_msg, config=config)
                self.update_fee(response)
                self._check_cancelled()
                if not response.text:
                    logger.warning(f"Get None response. Wait 15s. Retry num: {i + 1}.")
                    self._sleep(15)
                    continue

                response_text = remove_stop(response.text, stop_sequences)

                if not output_checker(user_msg, response_text):
                    logger.warning(f"Invalid response format. Retry num: {i + 1}.")
                    continue

                validated = True
                break
            except genai_errors.ClientError as e:
                if e.code == 429:
                    # Rate limit is a client error (4xx) but is retryable.
                    sleep_time = self._get_sleep_time(e)
                    logger.warning(f"Rate limited: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.")
                    self._sleep(sleep_time)
                elif e.code in (401, 403):
                    # Authentication/permission errors are deterministic.
                    raise ChatBotException(f"Authentication failed: {e}") from e
                else:
                    # Other client errors (4xx) are deterministic; retrying will not help.
                    raise ChatBotException(f"Client error: {e}") from e
            except genai_errors.ServerError as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f"ServerError: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.")
                self._sleep(sleep_time)

        if not response:
            raise ChatBotException("Failed to create a chat.")

        if not validated:
            logger.warning("Response format validation failed after all retries, returning best-effort response.")

        return response

    @staticmethod
    def _get_sleep_time(error):
        if isinstance(error, genai_errors.APIError) and error.code == 429:  # Rate limit
            return random.randint(30, 60)
        elif isinstance(error, genai_errors.ServerError):
            return 15
        else:
            return 5

    def close(self):
        """Close the Gemini client and release resources."""
        self.client.close()


class LiteLLMBot(ChatBot):
    """ChatBot backed by the LiteLLM SDK, routing to 100+ LLM providers.

    Note: intentionally not decorated with ``@_register_chatbot`` because
    ``_register_chatbot`` only maps hardcoded model names by provider enum
    (OPENAI / ANTHROPIC / GOOGLE) and has no LITELLM clause.  ``LiteLLMBot``
    is accessed exclusively via the ``"litellm:<model>"`` prefix through
    ``route_chatbot`` or via ``provider2chatbot[ModelProvider.LITELLM]``.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        temperature: float = 1,
        top_p: float = 1,
        retry: int = 8,
        max_async: int = 16,
        fee_limit: float = 0.8,
        proxy: str | None = None,
        base_url_config: dict | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
    ):
        temperature = max(0, min(2, temperature))
        super().__init__(model_name, temperature, top_p, retry, max_async, fee_limit)
        self.api_key = api_key
        self.api_base = (base_url_config or {}).get("litellm")
        self.extra_body = dict(extra_body) if extra_body else {}
        if proxy:
            logger.warning(
                "LiteLLMBot does not support per-instance proxy. "
                "Set HTTP_PROXY/HTTPS_PROXY environment variables instead."
            )

    def update_fee(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.api_fees[-1] += (
            prompt_tokens * self.model_info.input_price + completion_tokens * self.model_info.output_price
        ) / 1000000

    def get_content(self, response):
        content = response.choices[0].message.content
        return content if content is not None else ""

    def _create_chat(
        self,
        messages: list[dict],
        stop_sequences: list[str] | None = None,
        output_checker: Callable = lambda user_input, generated_content: True,
        temperature: float | None = None,
        top_p: float | None = None,
        min_tokens: int | None = None,
    ):
        try:
            import litellm  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "litellm is required for the litellm: provider. Install with: pip install 'openlrc-mac[litellm]'"
            )

        effective_temperature = temperature if temperature is not None else self.temperature
        effective_top_p = top_p if top_p is not None else self.top_p

        max_tokens = self._compute_max_tokens(messages, min_tokens=min_tokens)

        completion_kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": effective_temperature,
            "drop_params": True,
        }
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
        completion_kwargs["top_p"] = effective_top_p
        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        completion_kwargs.update(self.extra_body)

        response = None
        validated = False
        for i in range(self.retry):
            self._check_cancelled()
            try:
                response = litellm.completion(**completion_kwargs)
                self.update_fee(response)
                self._check_cancelled()

                if response.choices[0].finish_reason == "length":
                    usage = getattr(response, "usage", None)
                    raise LengthExceedException(
                        prompt_tokens=getattr(usage, "prompt_tokens", -1) if usage else -1,
                        completion_tokens=getattr(usage, "completion_tokens", -1) if usage else -1,
                        total_tokens=getattr(usage, "total_tokens", -1) if usage else -1,
                    )

                response_text = remove_stop(self.get_content(response), stop_sequences)

                if not output_checker(messages[-1]["content"], response_text):
                    logger.warning(f"Invalid response format. Retry num: {i + 1}.")
                    continue

                validated = True
                break
            except litellm.AuthenticationError as e:
                raise ChatBotException(f"Authentication failed: {e}") from e
            except (
                litellm.BadRequestError,
                litellm.NotFoundError,
                litellm.PermissionDeniedError,
                litellm.UnprocessableEntityError,
            ) as e:
                raise ChatBotException(f"Client error: {e}") from e
            except (
                litellm.RateLimitError,
                litellm.APIConnectionError,
                litellm.InternalServerError,
                litellm.ServiceUnavailableError,
                litellm.Timeout,
                json.decoder.JSONDecodeError,
            ) as e:
                sleep_time = self._get_sleep_time(e)
                logger.warning(f"{type(e).__name__}: {e}. Wait {sleep_time}s before retry. Retry num: {i + 1}.")
                self._sleep(sleep_time)

        if not response:
            raise ChatBotException("Failed to create a chat.")

        if not validated:
            logger.warning("Response format validation failed after all retries, returning best-effort response.")

        return response

    @staticmethod
    def _get_sleep_time(error):
        qualname = type(error).__name__
        if qualname == "RateLimitError":
            return random.randint(30, 60)
        elif qualname in ("Timeout", "APIConnectionError"):
            return 3
        elif qualname == "JSONDecodeError":
            return 1
        else:
            return 15


provider2chatbot: dict[str, type[ChatBot]] = {
    ModelProvider.OPENAI: GPTBot,
    ModelProvider.ANTHROPIC: ClaudeBot,
    ModelProvider.GOOGLE: GeminiBot,
    ModelProvider.THIRD_PARTY: GPTBot,
    ModelProvider.LOCAL_LLAMA: GPTBot,
    ModelProvider.LITELLM: LiteLLMBot,
}
