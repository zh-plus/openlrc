#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai
from pydantic import BaseModel

from openlrc.agents import create_chatbot
from openlrc.chatbot import ClaudeBot, GPTBot, route_chatbot
from openlrc.models import ModelConfig, ModelProvider
from openlrc.workflow import CancellationToken, WorkflowCancelled
from tests.conftest import LIVE_API, TEST_LLM_API_KEY, TEST_LLM_BASE_URL, TEST_MODELS


class Usage(BaseModel):
    pass


class OpenAIUsage(Usage):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AnthropicUsage(Usage):
    input_tokens: int
    output_tokens: int


class OpenAIResponse(BaseModel):
    usage: Usage


class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.gpt_bot = GPTBot(
            model_name=TEST_MODELS["gpt"].name,
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            base_url_config={"openai": TEST_LLM_BASE_URL},
            api_key=TEST_LLM_API_KEY or "test-key",
        )
        self.claude_bot = ClaudeBot(
            model_name="claude-3-5-sonnet-20241022",
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            api_key="test-key",
        )

    def _make_openrouter_bot(self, model_name: str):
        return GPTBot(
            model_name=model_name,
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            fee_limit=0.05,
            base_url_config={"openai": TEST_LLM_BASE_URL},
            api_key=TEST_LLM_API_KEY,
        )

    def test_estimate_fee(self):
        bot = self.gpt_bot
        messages = [{"role": "system", "content": "You are gpt."}, {"role": "user", "content": "Hello"}]
        fee = bot.estimate_fee(messages)
        self.assertIsNotNone(fee)

    def test_gpt_update_fee(self):
        bot = self.gpt_bot
        bot.api_fees += [0]
        response1 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300))
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=200, completion_tokens=400, total_tokens=600))
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = OpenAIResponse(usage=OpenAIUsage(prompt_tokens=300, completion_tokens=600, total_tokens=900))
        bot.update_fee(response3)
        self.assertIsNotNone(bot.api_fees)

    def test_claude_update_fee(self):
        bot = self.claude_bot
        bot.api_fees += [0]
        response1 = OpenAIResponse(usage=AnthropicUsage(input_tokens=100, output_tokens=200))
        bot.update_fee(response1)

        bot.api_fees += [0]
        response2 = OpenAIResponse(usage=AnthropicUsage(input_tokens=200, output_tokens=400))
        bot.update_fee(response2)

        bot.api_fees += [0]
        response3 = OpenAIResponse(usage=AnthropicUsage(input_tokens=300, output_tokens=600))
        bot.update_fee(response3)

        self.assertIsNotNone(bot.api_fees)

    def test_cancellation_after_sdk_return_prevents_next_retry(self):
        token = CancellationToken()
        self.gpt_bot.cancellation_token = token
        response = MagicMock()
        response.choices[0].finish_reason = None
        response.choices[0].message.content = "invalid"

        def cancel_after_response(**_kwargs):
            token.cancel()
            return response

        with (
            patch.object(self.gpt_bot.client.chat.completions, "create", side_effect=cancel_after_response) as create,
            patch.object(self.gpt_bot, "update_fee"),
            self.assertRaises(WorkflowCancelled),
        ):
            self.gpt_bot.message([{"role": "user", "content": "hello"}], output_checker=lambda _source, _target: False)

        create.assert_called_once()

    @unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1")
    def test_gpt_message_async(self):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        bot = self._make_openrouter_bot(TEST_MODELS["gpt"].name)
        messages_list = [[{"role": "user", "content": "Echo hello:"}], [{"role": "user", "content": "Echo hello:"}]]
        results = bot.message(messages_list)

        self.assertTrue(all(["hello" in bot.get_content(r).lower() for r in results]))

    @unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1")
    def test_claude_message_async(self):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        bot = self._make_openrouter_bot(TEST_MODELS["claude"].name)
        messages_list = [[{"role": "user", "content": "Echo hello:"}], [{"role": "user", "content": "Echo hello:"}]]
        results = bot.message(messages_list)

        self.assertTrue(all(["hello" in bot.get_content(r).lower() for r in results]))

    @unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1")
    def test_gpt_message_seq(self):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        bot = self._make_openrouter_bot(TEST_MODELS["gpt"].name)
        messages_list = [[{"role": "user", "content": "Echo hello:"}]]
        results = bot.message(messages_list)

        self.assertIn("hello", bot.get_content(results[0]).lower())

    @unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1")
    def test_claude_message_seq(self):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        bot = self._make_openrouter_bot(TEST_MODELS["claude"].name)
        messages_list = [[{"role": "user", "content": "Echo hello:"}]]
        results = bot.message(messages_list)
        assert "hello" in bot.get_content(results[0]).lower()

        self.assertIn("hello", bot.get_content(results[0]).lower())

    def test_route_chatbot(self):
        chatbot_model1 = "openai: claude-3-5-haiku-20241022"
        chabot_cls1, model_name1 = route_chatbot(chatbot_model1)
        self.assertEqual(chabot_cls1, GPTBot)
        try:
            _ = chabot_cls1(model_name=model_name1, temperature=1, top_p=1, retry=8, max_async=16, api_key="test-key")
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

        chatbot_model2 = "anthropic: gpt-3.5-turbo"
        chabot_cls2, model_name2 = route_chatbot(chatbot_model2)
        self.assertEqual(chabot_cls2, ClaudeBot)
        try:
            _ = chabot_cls2(model_name=model_name2, temperature=1, top_p=1, retry=8, max_async=16, api_key="test-key")
        except Exception as e:
            self.fail(f"Failed to create chatbot model {chatbot_model1}: {e}")

    def test_route_chatbot_undefined(self):
        chatbot_model = "openai: invalid_model_name"
        model_cls, model_name = route_chatbot(chatbot_model)
        self.assertEqual(model_cls, GPTBot)
        self.assertEqual(model_name, chatbot_model.split(":")[-1].strip())

    def test_temperature_clamp(self):
        chatbot1 = GPTBot(temperature=10, top_p=1, retry=8, max_async=16, api_key="test-key")
        chatbot2 = GPTBot(temperature=-1, top_p=1, retry=8, max_async=16, api_key="test-key")
        chatbot3 = ClaudeBot(temperature=2, top_p=1, retry=8, max_async=16, api_key="test-key")
        chatbot4 = ClaudeBot(temperature=-1, top_p=1, retry=8, max_async=16, api_key="test-key")

        self.assertEqual(chatbot1.temperature, 2)
        self.assertEqual(chatbot2.temperature, 0)
        self.assertEqual(chatbot3.temperature, 1)
        self.assertEqual(chatbot4.temperature, 0)

    def test_per_call_temperature_overrides_default(self):
        """message(temperature=X) should override the instance default set in __init__."""
        bot = GPTBot(model_name="gpt-4.1-nano", temperature=1.0, api_key="test-key")

        # Mock the sync completion call to capture the temperature it receives.
        captured: list[float | None] = []

        def fake_create(**kwargs: object) -> None:
            captured.append(kwargs.get("temperature"))  # type: ignore[arg-type]
            raise openai.AuthenticationError(message="test", response=httpx.Response(401), body=None)

        with patch.object(bot.client.chat.completions, "create", side_effect=fake_create):
            try:
                bot.message([{"role": "user", "content": "hi"}], temperature=0.3)
            except Exception:
                pass

        self.assertTrue(len(captured) > 0, "No API call was captured")
        self.assertEqual(captured[0], 0.3)

    def test_default_temperature_used_when_not_overridden(self):
        """message() without temperature should use the instance default from __init__."""
        bot = GPTBot(model_name="gpt-4.1-nano", temperature=0.7, api_key="test-key")

        captured: list[float | None] = []

        def fake_create(**kwargs: object) -> None:
            captured.append(kwargs.get("temperature"))  # type: ignore[arg-type]
            raise openai.AuthenticationError(message="test", response=httpx.Response(401), body=None)

        with patch.object(bot.client.chat.completions, "create", side_effect=fake_create):
            try:
                bot.message([{"role": "user", "content": "hi"}])
            except Exception:
                pass

        self.assertTrue(len(captured) > 0, "No API call was captured")
        self.assertEqual(captured[0], 0.7)

    def test_model_config_sampling_forwarded_to_chatbot(self):
        config = ModelConfig(
            provider=ModelProvider.LOCAL_LLAMA,
            name="hy-mt2-7b-local",
            api_key="openlrc-local",
            temperature=0.7,
            top_p=0.6,
        )

        bot = create_chatbot(config, fee_limit=0.0)

        self.assertEqual(bot.temperature, 0.7)
        self.assertEqual(bot.top_p, 0.6)
        self.assertIsNone(bot.agent_temperature(1.0))


class TestExtraBody(unittest.TestCase):
    """Tests for the extra_body passthrough feature (issue #128)."""

    def test_gpt_extra_body_splits_native_and_passthrough(self):
        """Native keys (frequency_penalty, seed) become top-level kwargs;
        non-native keys (top_k, chat_template_kwargs) go into extra_body."""
        bot = GPTBot(
            model_name="gpt-4.1-nano",
            api_key="test-key",
            extra_body={
                "frequency_penalty": 0.3,
                "seed": 42,
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        captured: list[dict] = []

        def fake_create(**kwargs: object) -> None:
            captured.append(dict(kwargs))
            raise openai.AuthenticationError(message="test", response=httpx.Response(401), body=None)

        with patch.object(bot.client.chat.completions, "create", side_effect=fake_create):
            try:
                bot.message([{"role": "user", "content": "hi"}])
            except Exception:
                pass

        self.assertTrue(len(captured) > 0, "No API call was captured")
        call = captured[0]

        # Native keys should be top-level kwargs
        self.assertEqual(call["frequency_penalty"], 0.3)
        self.assertEqual(call["seed"], 42)

        # Non-native keys should be in extra_body
        self.assertIn("extra_body", call)
        self.assertEqual(call["extra_body"]["top_k"], 20)
        self.assertEqual(call["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})

        # Native keys should NOT be in extra_body
        self.assertNotIn("frequency_penalty", call["extra_body"])
        self.assertNotIn("seed", call["extra_body"])

    def test_gpt_no_extra_body_matches_original_behavior(self):
        """When extra_body is not provided, the API call should not contain
        extra_body or any extra native kwargs — identical to pre-change behavior."""
        bot = GPTBot(model_name="gpt-4.1-nano", api_key="test-key")

        captured: list[dict] = []

        def fake_create(**kwargs: object) -> None:
            captured.append(dict(kwargs))
            raise openai.AuthenticationError(message="test", response=httpx.Response(401), body=None)

        with patch.object(bot.client.chat.completions, "create", side_effect=fake_create):
            try:
                bot.message([{"role": "user", "content": "hi"}])
            except Exception:
                pass

        self.assertTrue(len(captured) > 0, "No API call was captured")
        call = captured[0]

        # No extra_body kwarg should be present
        self.assertNotIn("extra_body", call)

        # No native extra keys should be present
        self.assertNotIn("frequency_penalty", call)
        self.assertNotIn("presence_penalty", call)
        self.assertNotIn("seed", call)

    def test_extra_body_shallow_copy_isolates_mutations(self):
        """Mutating the original dict after constructing the bot should not
        affect the bot's extra_body."""
        original = {"top_k": 20, "seed": 42}
        bot = GPTBot(model_name="gpt-4.1-nano", api_key="test-key", extra_body=original)

        # Mutate the original dict
        original["top_k"] = 999
        original["new_key"] = "surprise"

        # Bot should still have the original values
        self.assertEqual(bot.extra_body["top_k"], 20)
        self.assertEqual(bot.extra_body["seed"], 42)
        self.assertNotIn("new_key", bot.extra_body)


class TestThirdPartyBot(unittest.TestCase):
    def test_beta_base_url(self):
        bot = GPTBot(
            model_name="deepseek-chat",
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            base_url_config={"openai": "https://api.deepseek.com/beta"},
            api_key="test-key",
        )
        self.assertTrue(bot.model_info.beta)

    def test_non_beta_base_url(self):
        bot = GPTBot(
            model_name="deepseek-chat",
            temperature=1,
            top_p=1,
            retry=8,
            max_async=16,
            base_url_config={"openai": "https://api.deepseek.com"},
            api_key="test-key",
        )
        self.assertFalse(bot.model_info.beta)


# TODO: Retry_bot testing


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestGeminiBot(unittest.TestCase):
    # def setUp(self):
    #     import os
    #     os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
    #     os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    #
    # def tearDown(self):
    #     import os
    #     os.environ.pop('HTTP_PROXY')
    #     os.environ.pop('HTTPS_PROXY')

    def test_multi_turn(self):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        bot = GPTBot(
            model_name=TEST_MODELS["gemini"].name,
            base_url_config={"openai": TEST_LLM_BASE_URL},
            api_key=TEST_LLM_API_KEY,
        )
        result = bot.message(
            [
                {"role": "system", "content": "You are a echo machine, echo each word from input."},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "How are you?"},
                {"role": "user", "content": "THen?"},
            ]
        )[0]
        self.assertIsNotNone(bot.get_content(result))
