#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from openlrc.logger import logger


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    THIRD_PARTY = "third_party"


@dataclass
class ModelConfig:
    """
    Configuration for a specific model.

    Attributes:
        provider (ModelProvider): The provider of the model.
        name (str): The name of the model.
        base_url (Optional[str]): The base URL for the model API.
        api_key (Optional[str]): The API key for authentication.
        proxy (Optional[str]): The proxy server to use for requests.
    """

    provider: ModelProvider
    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None

    def __str__(self):
        return f'{self.provider.value}:{self.name}'


@dataclass
class ModelInfo:
    name: str
    provider: ModelProvider
    input_price: float  # per million tokens
    output_price: float  # per million tokens
    max_tokens: int
    context_window: int
    vision_support: bool = False
    knowledge_cutoff: Optional[str] = None
    latest_alias: Optional[str] = None
    beta: bool = False


class Models:
    # Claude Models
    CLAUDE_3_OPUS = ModelInfo(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_price=15.0,
        output_price=75.0,
        max_tokens=4096,
        context_window=200000,
        vision_support=True,
        knowledge_cutoff="Aug 2023",
        latest_alias="claude-3-opus-latest"
    )

    CLAUDE_3_SONNET = ModelInfo(
        name="claude-3-sonnet-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        max_tokens=4096,
        context_window=200000,
        vision_support=True,
        knowledge_cutoff="Aug 2023"
    )

    CLAUDE_3_HAIKU = ModelInfo(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        input_price=0.25,
        output_price=1.25,
        max_tokens=4096,
        context_window=200000,
        vision_support=True,
        knowledge_cutoff="Aug 2023"
    )

    CLAUDE_3_5_SONNET = ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        max_tokens=8192,
        context_window=200000,
        vision_support=True,
        knowledge_cutoff="Apr 2024",
        latest_alias="claude-3-5-sonnet-latest"
    )

    CLAUDE_3_7_SONNET = ModelInfo(
        name="claude-3-7-sonnet",
        provider=ModelProvider.ANTHROPIC,
        input_price=3.0,
        output_price=15.0,
        max_tokens=8192,
        context_window=200000,
        vision_support=True,
        knowledge_cutoff="Apr 2024",
        latest_alias="claude-3-7-sonnet-latest"
    )

    CLAUDE_3_5_HAIKU = ModelInfo(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_price=0.80,
        output_price=4.0,
        max_tokens=8192,
        context_window=200000,
        vision_support=False,
        knowledge_cutoff="July 2024",
        latest_alias="claude-3-5-haiku-latest"
    )

    # GPT Models
    GPT_4O = ModelInfo(
        name="gpt-4o-2024-08-06",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        max_tokens=16384,
        context_window=128000,
        vision_support=False,
        knowledge_cutoff="Oct 2023",
        latest_alias="gpt-4o"
    )

    GPT_4O_MINI = ModelInfo(
        name="gpt-4o-mini-2024-07-18",
        provider=ModelProvider.OPENAI,
        input_price=5.0,
        output_price=15.0,
        max_tokens=16384,
        context_window=128000,
        vision_support=False,
        knowledge_cutoff="Oct 2023",
        latest_alias="gpt-4o-mini"
    )

    GPT_4_TURBO = ModelInfo(
        name="gpt-4-turbo-2024-04-09",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        max_tokens=4096,
        context_window=128000,
        vision_support=True,
        knowledge_cutoff="Dec 2023",
        latest_alias="gpt-4-turbo"
    )

    GPT_4_TURBO_PREVIEW = ModelInfo(
        name="gpt-4-0125-preview",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        max_tokens=4096,
        context_window=128000,
        vision_support=True,
        knowledge_cutoff="Dec 2023",
        latest_alias="gpt-4-turbo-preview"
    )

    GPT_4_1106_PREVIEW = ModelInfo(
        name="gpt-4-1106-preview",
        provider=ModelProvider.OPENAI,
        input_price=10.0,
        output_price=30.0,
        max_tokens=4096,
        context_window=128000,
        vision_support=True,
        knowledge_cutoff="Apr 2023",
    )

    GPT_4 = ModelInfo(
        name="gpt-4-0613",
        provider=ModelProvider.OPENAI,
        input_price=30.0,
        output_price=60.0,
        max_tokens=8192,
        context_window=8192,
        vision_support=False,
        knowledge_cutoff="Sep 2021",
        latest_alias="gpt-4"
    )

    GPT_35_TURBO = ModelInfo(
        name="gpt-3.5-turbo-0125",
        provider=ModelProvider.OPENAI,
        input_price=0.5,
        output_price=1.5,
        max_tokens=4096,
        context_window=16385,
        knowledge_cutoff="Sep 2021",
        latest_alias="gpt-3.5-turbo"
    )

    # Gemini Models
    GEMINI_PRO = ModelInfo(
        name="gemini-1.5-pro",
        provider=ModelProvider.GOOGLE,
        input_price=1.25,
        output_price=5.0,
        max_tokens=8192,
        context_window=2097152,
        vision_support=True
    )

    GEMINI_FLASH = ModelInfo(
        name="gemini-1.5-flash",
        provider=ModelProvider.GOOGLE,
        input_price=0.075,
        output_price=0.30,
        max_tokens=8192,
        context_window=1048576
    )

    GEMINI_FLASH_8B = ModelInfo(
        name="gemini-1.5-flash-8b",
        provider=ModelProvider.GOOGLE,
        input_price=0.0375,
        output_price=0.15,
        max_tokens=8192,
        context_window=1048576
    )

    GEMINI_2_0_FLASH_LITE = ModelInfo(
        name="gemini-2.0-flash-lite-preview-02-05",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        max_tokens=8192,
        context_window=2097152,
        vision_support=True,
        knowledge_cutoff="Aug 2024"
    )

    GEMINI_2_0_FLASH_EXP = ModelInfo(
        name="gemini-2.0-flash-exp",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        max_tokens=8192,
        context_window=1048576,
        vision_support=True,
        knowledge_cutoff="Aug 2024"
    )

    GEMINI_2_0_FLASH = ModelInfo(
        name="gemini-2.0-flash",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        max_tokens=8192,
        context_window=1048576,
        vision_support=True,
        knowledge_cutoff="Aug 2024"
    )

    GEMINI_2_0_PRO_EXP = ModelInfo(
        name="gemini-2.0-pro-exp-02-05",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        max_tokens=8192,
        context_window=2097152,
        vision_support=True,
        knowledge_cutoff="Aug 2024"
    )

    GEMINI_2_5_PRO_EXP = ModelInfo(
        name="gemini-2.5-pro-exp-03-25",
        provider=ModelProvider.GOOGLE,
        input_price=0,
        output_price=0,
        max_tokens=8192,
        context_window=1048576,
        vision_support=True,
        knowledge_cutoff="Jan 2025"
    )

    # Third Party Models
    DEEPSEEK = ModelInfo(
        name="deepseek-chat",
        provider=ModelProvider.THIRD_PARTY,
        input_price=0.14,
        output_price=0.28,
        max_tokens=4096,
        context_window=32768
    )

    DEEPSEEK_BETA = ModelInfo(
        name="deepseek-chat",
        provider=ModelProvider.THIRD_PARTY,
        input_price=0.14,
        output_price=0.28,
        max_tokens=8192,
        context_window=32768,
        beta=True
    )

    DEEPSEEK_REASONER = ModelInfo(
        name="deepseek-reasoner",
        provider=ModelProvider.THIRD_PARTY,
        input_price=0.14,
        output_price=0.28,
        max_tokens=8192,
        context_window=65536,
        beta=False
    )

    DEEPSEEK_REASONER_2 = ModelInfo(
        name="deepseek-ai/DeepSeek-R1",
        provider=ModelProvider.THIRD_PARTY,
        input_price=0.14,
        output_price=0.28,
        max_tokens=8192,
        context_window=65536,
        beta=False
    )

    class DefaultOpenAIModelInfo(ModelInfo):

        """Default configuration for unrecognized OpenAI models."""

        def __init__(self, model_name: str):
            super().__init__(
                name=model_name,
                provider=ModelProvider.OPENAI,
                input_price=10.0,
                output_price=30.0,
                max_tokens=8192,
                context_window=8192,
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
                max_tokens=4096,
                context_window=100000,
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
                max_tokens=8192,
                context_window=32768,
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
                max_tokens=4096,
                context_window=32768,
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
        if any(name in model_name.lower() for name in ['gpt', 'openai', 'davinci', 'text-', 'curie']):
            provider_name = "OpenAI"
            default_model = cls.DefaultOpenAIModelInfo(model_name)
        elif any(name in model_name.lower() for name in ['claude', 'anthropic']):
            provider_name = "Anthropic"
            default_model = cls.DefaultAnthropicModelInfo(model_name)
        elif any(name in model_name.lower() for name in ['gemini', 'google', 'palm']):
            provider_name = "Google"
            default_model = cls.DefaultGeminiModelInfo(model_name)
        else:
            provider_name = "third-party"
            default_model = cls.DefaultThirdPartyModelInfo(model_name)

        logger.warning(f"Model '{model_name}' not found in predefined models" +
                       (" (beta)" if beta else "") +
                       f". Using with inferred {provider_name} provider and default parameters.")

        return default_model


def list_chatbot_models() -> List[str]:
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
