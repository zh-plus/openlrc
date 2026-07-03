#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from typing import get_args, get_type_hints

from openlrc.config import TranscriptionConfig, TranslationConfig
from openlrc.models import ModelConfig, ModelProvider


class TestTranslationConfigAnnotations(unittest.TestCase):
    def test_annotations_remain_serialization_friendly(self):
        hints = get_type_hints(TranslationConfig)

        self.assertEqual(set(get_args(hints["chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["retry_chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["cr_chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["glossary"])), {str, type(None)})


class TestTranscriptionConfig(unittest.TestCase):
    def test_faster_whisper_fields_are_not_supported(self):
        with self.assertRaises(TypeError):
            TranscriptionConfig(compute_type="float16")

        with self.assertRaises(TypeError):
            TranscriptionConfig(device="cuda")

        with self.assertRaises(TypeError):
            TranscriptionConfig(vad_options={"threshold": 0.5})


class TestModelConfig(unittest.TestCase):
    def test_provider_string_coercion(self):
        # Lowercase known provider → coerced to enum
        mc = ModelConfig(provider="openai", name="gpt-4")
        self.assertIs(mc.provider, ModelProvider.OPENAI)

        # Uppercase known provider → .lower() normalizes
        mc = ModelConfig(provider="ANTHROPIC", name="claude")
        self.assertIs(mc.provider, ModelProvider.ANTHROPIC)

        # Unknown provider string → stays as string
        mc = ModelConfig(provider="my-custom-provider", name="my-model")
        self.assertEqual(mc.provider, "my-custom-provider")

        # Enum value → no change
        mc = ModelConfig(provider=ModelProvider.GOOGLE, name="gemini")
        self.assertIs(mc.provider, ModelProvider.GOOGLE)

    def test_model_config_str(self):
        mc = ModelConfig(provider=ModelProvider.OPENAI, name="gpt-4")
        self.assertEqual(str(mc), "openai:gpt-4")

        mc = ModelConfig(provider="custom", name="my-model")
        self.assertEqual(str(mc), "custom:my-model")
