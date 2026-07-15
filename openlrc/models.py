#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import sys
from dataclasses import dataclass
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnum
else:

    class _StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""

        def __str__(self) -> str:
            return self.value


class ModelProvider(_StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    LITELLM = "litellm"
    THIRD_PARTY = "third_party"
    LOCAL_LLAMA = "local_llama"


@dataclass
class ModelConfig:
    """
    Configuration for a specific model.

    Combines identity, connection, and capability parameters into a single
    config object.  Capability fields (``context_window``, ``max_tokens``)
    are optional: when set they define output and context-window limits;
    when ``None`` (the default) the API server's own limits apply, which
    is the recommended setting for most users.

    Attributes:
        provider (ModelProvider | str): The provider of the model.
        name (str): The name of the model.
        base_url (Optional[str]): The base URL for the model API.
        api_key (Optional[str]): The API key for authentication.
        proxy (Optional[str]): The proxy server to use for requests.
        context_window (Optional[int]): Total context window size in tokens.
            When set, enables context-window safety checks in ``_compute_max_tokens``.
        max_tokens (Optional[int]): Maximum output tokens per request.
            When set, ``_compute_max_tokens`` caps output at this value;
            when ``None`` (default), ``max_tokens`` is not sent to the API.
        temperature (Optional[float]): Default sampling temperature for this
            model. When ``None`` (default), the chatbot class default is used.
        top_p (Optional[float]): Default nucleus-sampling value for this model.
            When ``None`` (default), the chatbot class default is used.
        extra_body (Optional[dict]): Provider-specific parameters passed through
            to the underlying SDK.  Each chatbot subclass extracts the keys it
            recognises (e.g. ``frequency_penalty`` for OpenAI, ``top_k`` for
            Anthropic/Gemini) and forwards the remainder via the SDK's own
            extension mechanism.  ``None`` (the default) produces the exact same
            API call as before this field existed.

            Do **not** include keys that are already managed by the chatbot
            constructor or ``_create_chat`` (e.g. ``temperature``, ``top_p``,
            ``stop``, ``max_tokens``, ``model``, ``messages``).  Behaviour is
            undefined if such keys are present.
    """

    provider: ModelProvider | str = ModelProvider.OPENAI
    name: str = "gpt-4.1-nano"
    base_url: str | None = None
    api_key: str | None = None
    proxy: str | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    extra_body: dict | None = None

    def __post_init__(self):
        if isinstance(self.provider, str):
            try:
                self.provider = ModelProvider(self.provider.lower())
            except ValueError:
                # Custom provider strings remain strings, known ones get coerced
                pass

    def __str__(self):
        provider_str = self.provider.value if isinstance(self.provider, ModelProvider) else self.provider
        return f"{provider_str}:{self.name}"


@dataclass
class ModelInfo:
    name: str
    provider: ModelProvider
    input_price: float  # per million tokens
    output_price: float  # per million tokens
    vision_support: bool = False
    knowledge_cutoff: str | None = None
    latest_alias: str | None = None
    beta: bool = False


class Models:
    # Claude Models
    CLAUDE_3_OPUS = ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_price=15.0,
        output_price=75.0,
        vision_support=True,
        knowledge_cutoff="Aug 2023",
        latest_alias="claude-3-opus-latest",
    )

    CLAUDE_3_SONNET = ModelInfo(
        name="claude-3-sonnet-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        vision_support=True,
        knowledge_cutoff="Aug 2023",
    )

    CLAUDE_3_HAIKU = ModelInfo(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        input_price=0.25,
        output_price=1.25,
        vision_support=True,
        knowledge_cutoff="Aug 2023",
    )

    CLAUDE_3_5_SONNET = ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        vision_support=True,
        knowledge_cutoff="Apr 2024",
        latest_alias="claude-3-5-sonnet-latest",
    )

    CLAUDE_3_7_SONNET = ModelInfo(
        name="claude-3-7-sonnet",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        vision_support=True,
        knowledge_cutoff="Apr 2024",
        latest_alias="claude-3-7-sonnet-latest",
    )

    CLAUDE_3_5_HAIKU = ModelInfo(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_price=0.80,
        output_price=4.0,
        vision_support=False,
        knowledge_cutoff="July 2024",
        latest_alias="claude-3-5-haiku-latest",
    )

    # GPT Models
    GPT_4O = ModelInfo(
        name="gpt-4o-2024-08-06",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        vision_support=False,
        knowledge_cutoff="Oct 2023",
        latest_alias="gpt-4o",
    )

    GPT_4O_MINI = ModelInfo(
        name="gpt-4o-mini-2024-07-18",
        provider=ModelProvider.OPENAI,
        input_price=5.0,
        output_price=15.0,
        vision_support=False,
        knowledge_cutoff="Oct 2023",
        latest_alias="gpt-4o-mini",
    )

    GPT_4_TURBO = ModelInfo(
        name="gpt-4-turbo-2024-04-09",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        vision_support=True,
        knowledge_cutoff="Dec 2023",
        latest_alias="gpt-4-turbo",
    )

    GPT_4_TURBO_PREVIEW = ModelInfo(
        name="gpt-4-0125-preview",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        vision_support=True,
        knowledge_cutoff="Dec 2023",
        latest_alias="gpt-4-turbo-preview",
    )

    GPT_4_1106_PREVIEW = ModelInfo(
        name="gpt-4-1106-preview",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        vision_support=True,
        knowledge_cutoff="Apr 2023",
    )

    GPT_4 = ModelInfo(
        name="gpt-4-0613",
        provider=ModelProvider.OPENAI,
        input_price=30.0,
        output_price=60.0,
        vision_support=False,
        knowledge_cutoff="Sep 2021",
        latest_alias="gpt-4",
    )

    GPT_35_TURBO = ModelInfo(
        name="gpt-3.5-turbo-0125",
        provider=ModelProvider.OPENAI,
        input_price=0.5,
        output_price=1.5,
        knowledge_cutoff="Sep 2021",
        latest_alias="gpt-3.5-turbo",
    )

    GPT_41_NANO = ModelInfo(
        name="gpt-4.1-nano",
        provider=ModelProvider.OPENAI,
        input_price=0.1,
        output_price=0.4,
        knowledge_cutoff="Jun 01, 2024",
    )

    GPT_41_MINI = ModelInfo(
        name="gpt-4.1-mini",
        provider=ModelProvider.OPENAI,
        input_price=0.4,
        output_price=1.6,
        knowledge_cutoff="Jun 01, 2024",
    )

    GPT_41 = ModelInfo(
        name="gpt-4.1",
        provider=ModelProvider.OPENAI,
        input_price=2.0,
        output_price=8.0,
        knowledge_cutoff="Jun 01, 2024",
    )

    # Gemini Models
    GEMINI_PRO = ModelInfo(
        name="gemini-1.5-pro", provider=ModelProvider.GOOGLE, input_price=1.25, output_price=5.0, vision_support=True
    )

    GEMINI_FLASH = ModelInfo(
        name="gemini-1.5-flash", provider=ModelProvider.GOOGLE, input_price=0.075, output_price=0.30
    )

    GEMINI_FLASH_8B = ModelInfo(
        name="gemini-1.5-flash-8b", provider=ModelProvider.GOOGLE, input_price=0.0375, output_price=0.15
    )

    GEMINI_2_0_FLASH_LITE = ModelInfo(
        name="gemini-2.0-flash-lite-preview-02-05",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        vision_support=True,
        knowledge_cutoff="Aug 2024",
    )

    GEMINI_2_0_FLASH_EXP = ModelInfo(
        name="gemini-2.0-flash-exp",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        vision_support=True,
        knowledge_cutoff="Aug 2024",
    )

    GEMINI_2_0_FLASH = ModelInfo(
        name="gemini-2.0-flash",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        vision_support=True,
        knowledge_cutoff="Aug 2024",
    )

    GEMINI_2_0_PRO_EXP = ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        vision_support=True,
        knowledge_cutoff="Aug 2024",
    )

    GEMINI_2_5_PRO_EXP = ModelInfo(
        name="gemini-2.5-pro-exp-03-25",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        vision_support=True,
        knowledge_cutoff="Jan 2025",
    )

    # Third Party Models
    DEEPSEEK = ModelInfo(name="deepseek-chat", provider=ModelProvider.THIRD_PARTY, input_price=0.14, output_price=0.28)

    DEEPSEEK_BETA = ModelInfo(
        name="deepseek-chat", provider=ModelProvider.THIRD_PARTY, input_price=0.14, output_price=0.28, beta=True
    )

    DEEPSEEK_REASONER = ModelInfo(
        name="deepseek-reasoner", provider=ModelProvider.THIRD_PARTY, input_price=0.14, output_price=0.28, beta=False
    )

    DEEPSEEK_REASONER_2 = ModelInfo(
        name="deepseek-ai/DeepSeek-R1",
        provider=ModelProvider.THIRD_PARTY,
        input_price=0.14,
        output_price=0.28,
        beta=False,
    )

    LOCAL_QWEN35_9B = ModelInfo(
        name="qwen3.5-9b-local", provider=ModelProvider.LOCAL_LLAMA, input_price=0.0, output_price=0.0
    )

    LOCAL_HY_MT2_7B = ModelInfo(
        name="hy-mt2-7b-local", provider=ModelProvider.LOCAL_LLAMA, input_price=0.0, output_price=0.0
    )

    LOCAL_HY_MT2_30B_A3B = ModelInfo(
        name="hy-mt2-30b-a3b-local", provider=ModelProvider.LOCAL_LLAMA, input_price=0.0, output_price=0.0
    )

    class DefaultOpenAIModelInfo(ModelInfo):
        """Default configuration for unrecognized OpenAI models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.OPENAI,
                input_price=10.0,
                output_price=30.0,
                vision_support=False,
                knowledge_cutoff=None,
                latest_alias=None,
            )

    class DefaultAnthropicModelInfo(ModelInfo):
        """Default configuration for unrecognized Anthropic models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.ANTHROPIC,
                input_price=8.0,
                output_price=24.0,
                vision_support=False,
                knowledge_cutoff=None,
                latest_alias=None,
            )

    class DefaultGeminiModelInfo(ModelInfo):
        """Default configuration for unrecognized Google models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.GOOGLE,
                input_price=1.0,
                output_price=3.0,
                vision_support=False,
                knowledge_cutoff=None,
                latest_alias=None,
            )

    class DefaultLiteLLMModelInfo(ModelInfo):
        """Default configuration for LiteLLM-routed models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.LITELLM,
                input_price=1.0,
                output_price=1.0,
                vision_support=False,
                knowledge_cutoff=None,
                latest_alias=None,
            )

    class DefaultThirdPartyModelInfo(ModelInfo):
        """Default configuration for unrecognized third-party models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.THIRD_PARTY,
                input_price=1.0,
                output_price=0.2,
                vision_support=False,
                knowledge_cutoff=None,
                latest_alias=None,
            )

    @classmethod
    def get_model(cls, model_name: str, beta: bool = False) -> ModelInfo:
        """Get model info by name and beta status

        Args:
            model_name: Name or latest alias of the model
            beta: Whether to include beta models in search

        Returns:
            ModelInfo: Information about the requested model

        Notes:
            If the exact model is not found, it returns a default ModelInfo based
            on the inferred provider, logging a warning rather than raising an error.
        """
        # First try to find an exact match in our known models
        for model in cls.__dict__.values():
            if not isinstance(model, ModelInfo):
                continue

            name_matches = model.name == model_name or model.latest_alias == model_name
            beta_matches = model.beta == beta

            if name_matches and beta_matches:
                return model

        # If no exact match found, try to infer provider from model name
        # Note the model_provider is not vital for next processing
        # LiteLLM uses provider/model format -- infer from prefix
        lower_name = model_name.lower()
        if lower_name.startswith("openai/"):
            default_model = cls.DefaultOpenAIModelInfo(model_name)
        elif lower_name.startswith("anthropic/"):
            default_model = cls.DefaultAnthropicModelInfo(model_name)
        elif lower_name.startswith(("gemini/", "google/")):
            default_model = cls.DefaultGeminiModelInfo(model_name)
        elif "/" in model_name and lower_name.split("/")[0] in (
            "groq",
            "together_ai",
            "deepseek",
            "mistral",
            "bedrock",
            "vertex_ai",
            "azure",
            "cohere",
            "fireworks",
            "replicate",
        ):
            default_model = cls.DefaultThirdPartyModelInfo(model_name)
        elif any(name in lower_name for name in ["gpt", "openai", "davinci", "text-", "curie"]):
            default_model = cls.DefaultOpenAIModelInfo(model_name)
        elif any(name in lower_name for name in ["claude", "anthropic"]):
            default_model = cls.DefaultAnthropicModelInfo(model_name)
        elif any(name in lower_name for name in ["gemini", "google", "palm"]):
            default_model = cls.DefaultGeminiModelInfo(model_name)
        else:
            default_model = cls.DefaultThirdPartyModelInfo(model_name)

        return default_model


def list_chatbot_models() -> list[str]:
    """
    List available chatbot models for translation.

    Returns:
        List[str]: List of available chatbot model names and their latest aliases.
    """
    models = []
    for model in Models.__dict__.values():
        if not isinstance(model, ModelInfo):
            continue

        models.append(model.name)
        if model.latest_alias:
            models.append(model.latest_alias)

    return models
