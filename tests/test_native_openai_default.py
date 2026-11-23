import json

import httpx
import pytest
from openai import OpenAI

from openlrc import LRCer
from openlrc.agents import create_chatbot
from openlrc.chatbot import GPTBot
from openlrc.evaluate import LLMTranslationEvaluator
from openlrc.models import ModelConfig


@pytest.fixture
def openai_requests(monkeypatch):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url, body))
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "created": 0,
                "model": body["model"],
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def client(**kwargs):
        kwargs["http_client"].close()
        return OpenAI(**{**kwargs, "http_client": httpx.Client(transport=httpx.MockTransport(respond))})

    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("openlrc.chatbot.OpenAI", client)
    monkeypatch.setattr("openlrc.chatbot.get_text_token_number", lambda _: 1)
    monkeypatch.setattr("openlrc.chatbot.get_messages_token_number", lambda _: 1)
    return requests


def test_implicit_native_default_uses_replacement(openai_requests):
    with GPTBot(json_mode=True) as bot:
        bot.message([{"role": "user", "content": "Translate this"}], stop_sequences=["END"])
        assert bot.model_info.input_price == 0.20
        assert bot.model_info.output_price == 1.20

    url, body = openai_requests[0]
    assert url.host == "api.openai.com"
    assert body["model"] == "gpt-5.6-luna"
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 1
    assert body["top_p"] == 1
    assert body["response_format"] == {"type": "json_object"}
    assert body["stop"] == ["END"]
    assert "max_completion_tokens" not in body


def test_explicit_native_luna_uses_completion_token_limit(openai_requests):
    with create_chatbot(ModelConfig(name="gpt-5.6-luna", max_tokens=32)) as bot:
        bot.message([{"role": "user", "content": "Translate this"}])

    body = openai_requests[0][1]
    assert body["max_completion_tokens"] == 32
    assert "max_tokens" not in body


def test_native_luna_does_not_send_sampling_with_explicit_reasoning(openai_requests):
    with GPTBot("gpt-5.6-luna", extra_body={"reasoning_effort": "low"}) as bot:
        bot.message([{"role": "user", "content": "Translate this"}])

    body = openai_requests[0][1]
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body
    assert "top_p" not in body


@pytest.mark.parametrize("gateway", ["configured", "environment", "openrouter"])
def test_implicit_gateway_keeps_existing_model_and_request(gateway, monkeypatch, openai_requests):
    kwargs = {}
    if gateway == "configured":
        kwargs["base_url_config"] = {"openai": "https://gateway.example/v1"}
    elif gateway == "environment":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    else:
        monkeypatch.delenv("OPENAI_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")

    with GPTBot(**kwargs) as bot:
        bot.message([{"role": "user", "content": "Translate this"}])

    url, body = openai_requests[0]
    assert url.host == ("openrouter.ai" if gateway == "openrouter" else "gateway.example")
    assert body["model"] == "gpt-4.1-nano"
    assert "reasoning_effort" not in body


def test_explicit_other_model_keeps_existing_request_on_native_host(openai_requests):
    with create_chatbot(ModelConfig(name="gpt-4.1-nano", max_tokens=32)) as bot:
        bot.message([{"role": "user", "content": "Translate this"}])

    body = openai_requests[0][1]
    assert body["model"] == "gpt-4.1-nano"
    assert body["max_tokens"] == 32
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


def test_luna_on_compatible_gateway_keeps_compatible_request(openai_requests):
    config = ModelConfig(name="gpt-5.6-luna", max_tokens=32, base_url="https://gateway.example/v1")
    with create_chatbot(config) as bot:
        bot.message([{"role": "user", "content": "Translate this"}])

    body = openai_requests[0][1]
    assert body["model"] == "gpt-5.6-luna"
    assert body["max_tokens"] == 32
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


def test_app_and_evaluator_select_default_native_model(openai_requests):
    lrcer = LRCer()
    with lrcer.chatbot as bot, LLMTranslationEvaluator() as evaluator:
        assert bot.model_name == evaluator._chatbot.model_name == "gpt-5.6-luna"
