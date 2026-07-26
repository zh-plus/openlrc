"""Provider connection checks used by interactive settings."""

from __future__ import annotations

from openlrc.agents import create_chatbot
from openlrc.application.credentials import CredentialStore
from openlrc.application.settings import ProviderSettings
from openlrc.models import ModelConfig, ModelProvider
from openlrc.workflow import redact_sensitive_text

PROVIDER_MODELS = {
    "openai": ModelProvider.OPENAI,
    "anthropic": ModelProvider.ANTHROPIC,
    "google": ModelProvider.GOOGLE,
    "litellm": ModelProvider.LITELLM,
    "third_party": ModelProvider.THIRD_PARTY,
}


def build_provider_model(name: str, settings: ProviderSettings, credentials: CredentialStore) -> ModelConfig:
    """Build a provider probe from unsaved interactive settings."""
    if name not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported provider: {name}.")
    if not settings.model.strip():
        raise ValueError(f"Configure a model for provider {name}.")
    credential = credentials.resolve(name)
    if credential.value is None:
        raise ValueError(f"No API key configured for provider {name}.")
    return ModelConfig(
        provider=PROVIDER_MODELS[name],
        name=settings.model.strip(),
        base_url=settings.base_url.strip() or None,
        api_key=credential.value,
        proxy=settings.proxy.strip() or None,
    )


def test_provider_connection(model: ModelConfig) -> str:
    """Send one tiny, non-retried request and return a safe status message."""
    probe = ModelConfig(
        provider=model.provider,
        name=model.name,
        base_url=model.base_url,
        api_key=model.api_key,
        proxy=model.proxy,
        max_tokens=1,
        temperature=0,
        top_p=1,
    )
    chatbot = create_chatbot(probe, fee_limit=0.01)
    chatbot.retry = 0
    try:
        response = chatbot.message([{"role": "user", "content": "Reply OK."}])
        chatbot.get_content(response)
    except Exception as exc:
        raise RuntimeError(redact_sensitive_text(str(exc) or type(exc).__name__)) from exc
    finally:
        chatbot.close()
    return "Connection succeeded. The provider may have charged for one minimal request."
