#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import inspect
import unittest
from pathlib import Path
from typing import get_args, get_type_hints

from openlrc.config import (
    ContextAssistance,
    ContextLLMConfig,
    EditConfig,
    HyMT2Mode,
    LocalLLMConfig,
    SubtitleOptimizationMode,
    TranscriptionConfig,
    TranslationConfig,
)
from openlrc.context import TranslationBriefInput
from openlrc.glossary import GlossaryCatalog
from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_ALIAS,
    DEFAULT_LLAMA_MODEL_FILE,
    HY_MT2_7B_MODEL_ALIAS,
    HY_MT2_7B_MODEL_FILE,
    HY_MT2_30B_A3B_PROFILE,
    HY_MT2_PROMPT_PROFILE,
)
from openlrc.models import ModelConfig, ModelProvider


class TestTranslationConfigAnnotations(unittest.TestCase):
    def test_annotations_remain_serialization_friendly(self):
        hints = get_type_hints(TranslationConfig)

        self.assertEqual(set(get_args(hints["chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["retry_chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["cr_chatbot"])), {ModelConfig, type(None)})
        self.assertEqual(set(get_args(hints["glossary"])), {dict, str, Path, GlossaryCatalog, type(None)})
        self.assertEqual(set(get_args(hints["translation_brief"])), {TranslationBriefInput, dict, type(None)})
        self.assertEqual(set(get_args(hints["local_llm"])), {LocalLLMConfig, type(None)})


class TestTranscriptionConfig(unittest.TestCase):
    def test_faster_whisper_fields_are_not_supported(self):
        with self.assertRaises(TypeError):
            TranscriptionConfig(compute_type="float16")

        with self.assertRaises(TypeError):
            TranscriptionConfig(device="cuda")

        with self.assertRaises(TypeError):
            TranscriptionConfig(vad_options={"threshold": 0.5})


class TestSubtitleOptimizationMode(unittest.TestCase):
    def test_public_values(self):
        self.assertEqual(SubtitleOptimizationMode.AGGRESSIVE.value, "aggressive")
        self.assertEqual(SubtitleOptimizationMode.RELAXED.value, "relaxed")


class TestLocalLLMConfig(unittest.TestCase):
    def test_default_values(self):
        config = LocalLLMConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.model_path, DEFAULT_LLAMA_MODEL_FILE)
        self.assertEqual(config.alias, DEFAULT_LLAMA_MODEL_ALIAS)

    def test_local_qwen35_9b_factory(self):
        config = TranslationConfig.local_qwen35_9b(idle_timeout=12, port=9090)

        self.assertIsNotNone(config.local_llm)
        self.assertEqual(config.local_llm.model_path, DEFAULT_LLAMA_MODEL_FILE)
        self.assertEqual(config.local_llm.idle_timeout, 12)
        self.assertEqual(config.local_llm.port, 9090)
        self.assertEqual(config._translator_engine, "classic")
        self.assertTrue(config.enable_cr)
        self.assertEqual(config.consumer_thread, 1)
        self.assertEqual(config.fee_limit, 0.0)
        self.assertIs(config.chatbot.provider, ModelProvider.LOCAL_LLAMA)
        self.assertEqual(config.chatbot.name, DEFAULT_LLAMA_MODEL_ALIAS)
        self.assertNotIn("translate_mode", inspect.signature(TranslationConfig.local_qwen35_9b).parameters)

    def test_local_hy_mt2_7b_factory(self):
        config = TranslationConfig.local_hy_mt2_7b(idle_timeout=12, port=9090)

        self.assertIsNotNone(config.local_llm)
        self.assertEqual(config.local_llm.model_path, HY_MT2_7B_MODEL_FILE)
        self.assertEqual(config.local_llm.alias, HY_MT2_7B_MODEL_ALIAS)
        self.assertEqual(config.local_llm.idle_timeout, 12)
        self.assertEqual(config.local_llm.port, 9090)
        self.assertEqual(config._translator_engine, "lean")
        self.assertFalse(config.enable_cr)
        self.assertEqual(config.consumer_thread, 1)
        self.assertEqual(config.fee_limit, 0.0)
        self.assertEqual(config.prompt_profile, HY_MT2_PROMPT_PROFILE)
        self.assertIs(config.chatbot.provider, ModelProvider.LOCAL_LLAMA)
        self.assertEqual(config.chatbot.name, HY_MT2_7B_MODEL_ALIAS)
        self.assertEqual(config.chatbot.temperature, 0.7)
        self.assertEqual(config.chatbot.top_p, 0.6)
        self.assertEqual(config.chatbot.max_tokens, 4096)
        self.assertEqual(config.chatbot.extra_body, {"top_k": 20, "repeat_penalty": 1.05})
        self.assertIs(config.hy_mt2_mode, HyMT2Mode.FAST)

    def test_hy_mt2_factory_does_not_expose_translation_engine(self):
        self.assertNotIn("translate_mode", inspect.signature(TranslationConfig.local_hy_mt2_7b).parameters)
        self.assertNotIn("translate_mode", inspect.signature(TranslationConfig).parameters)

    def test_hy_mt2_context_requires_explicit_context_model(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit context_llm"):
            TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.CONTEXT)

    def test_hy_mt2_context_accepts_online_context_model(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        config = TranslationConfig.local_hy_mt2_7b(mode="context", context_llm=context_llm)

        self.assertIs(config.hy_mt2_mode, HyMT2Mode.NORMAL)
        self.assertIs(config.context_llm, context_llm)

    def test_complete_manual_brief_removes_normal_context_dependency(self):
        manual = TranslationBriefInput(summary="A complete story summary.", characters=[], tone_style="")

        config = TranslationConfig.local_hy_mt2_7b(mode="normal", translation_brief=manual)

        self.assertIsNone(config.context_llm)
        self.assertIs(config.translation_brief, manual)

    def test_context_assistance_off_accepts_complete_manual_brief_without_context_model(self):
        manual = TranslationBriefInput(summary="A complete story summary.", characters=[], tone_style="")

        config = TranslationConfig.local_hy_mt2_7b(
            mode="normal", context_assistance=ContextAssistance.OFF, translation_brief=manual
        )

        self.assertIs(config.context_assistance, ContextAssistance.OFF)
        self.assertIsNone(config.context_llm)

    def test_context_assistance_off_rejects_incomplete_brief(self):
        with self.assertRaisesRegex(ValueError, "complete manual Translation Brief"):
            TranslationConfig.local_hy_mt2_7b(
                mode="normal",
                context_assistance="off",
                translation_brief=TranslationBriefInput(summary="A fixed summary."),
            )

    def test_context_assistance_off_rejects_required_and_explicit_context(self):
        complete = TranslationBriefInput(summary="Story", characters=[], tone_style="")
        context = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="context")

        with self.assertRaisesRegex(ValueError, "cannot be off in Hy-MT2 pro"):
            TranslationConfig.local_hy_mt2_7b(mode="pro", context_assistance="off", translation_brief=complete)
        with self.assertRaisesRegex(ValueError, "Normal Plus semantic review"):
            TranslationConfig.local_hy_mt2_7b(mode="normal-plus", context_assistance="off", translation_brief=complete)
        with self.assertRaisesRegex(ValueError, "conflicts with an explicit context_llm"):
            TranslationConfig.local_hy_mt2_7b(
                mode="normal", context_assistance="off", translation_brief=complete, context_llm=context
            )

    def test_context_assistance_off_allows_zero_round_normal_plus(self):
        complete = TranslationBriefInput(summary="Story", characters=[], tone_style="")

        config = TranslationConfig.local_hy_mt2_7b(
            mode="normal-plus",
            context_assistance="off",
            translation_brief=complete,
            edit_config=EditConfig(max_rounds=0, semantic_review=False),
        )

        self.assertIsNone(config.context_llm)
        self.assertIs(config.context_assistance, ContextAssistance.OFF)

    def test_partial_manual_brief_still_requires_context_model(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit context_llm"):
            TranslationConfig.local_hy_mt2_7b(
                mode="normal", translation_brief=TranslationBriefInput(summary="A fixed summary.")
            )

    def test_empty_manual_brief_mapping_keeps_automatic_context_behavior(self):
        context = ContextLLMConfig.online(provider="openai", model="general-model")

        config = TranslationConfig.local_hy_mt2_7b(mode="normal", context_llm=context, translation_brief={})

        self.assertIsNone(config.translation_brief)

    def test_complete_normal_plus_round_zero_can_skip_context_model(self):
        manual = TranslationBriefInput(summary="A complete story summary.", characters=[], tone_style="")

        config = TranslationConfig.local_hy_mt2_7b(
            mode="normal-plus", translation_brief=manual, edit_config=EditConfig(max_rounds=0, semantic_review=False)
        )

        self.assertIsNone(config.context_llm)

    def test_complete_normal_plus_semantic_round_still_requires_context_model(self):
        manual = TranslationBriefInput(summary="A complete story summary.", characters=[], tone_style="")

        with self.assertRaisesRegex(ValueError, "requires an explicit context_llm"):
            TranslationConfig.local_hy_mt2_7b(mode="normal-plus", translation_brief=manual)

    def test_fast_rejects_manual_brief(self):
        manual = TranslationBriefInput(summary="A complete story summary.", characters=[], tone_style="")

        with self.assertRaisesRegex(ValueError, "fast mode does not use"):
            TranslationConfig.local_hy_mt2_7b(translation_brief=manual)

    def test_hy_mt2_pro_requires_and_accepts_context_model(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit context_llm"):
            TranslationConfig.local_hy_mt2_7b(mode=HyMT2Mode.PRO)

        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        config = TranslationConfig.local_hy_mt2_7b(mode="pro", context_llm=context_llm)

        self.assertEqual(HyMT2Mode.PRO.value, "pro")
        self.assertIs(config.hy_mt2_mode, HyMT2Mode.PRO)
        self.assertIs(config.context_llm, context_llm)

    def test_local_context_model_is_staged_without_idle_timer(self):
        context_llm = ContextLLMConfig.local_qwen35_9b(model="qwen3.5-9b", port=9091)

        self.assertIsNotNone(context_llm.local_llm)
        self.assertEqual(context_llm.local_llm.port, 9091)
        self.assertEqual(context_llm.local_llm.idle_timeout, 0)
        self.assertIs(context_llm.chatbot.provider, ModelProvider.LOCAL_LLAMA)

    def test_local_hy_mt2_30b_requires_explicit_model(self):
        with self.assertRaises(ValueError):
            TranslationConfig.local_hy_mt2(size=HY_MT2_30B_A3B_PROFILE)

        config = TranslationConfig.local_hy_mt2(size=HY_MT2_30B_A3B_PROFILE, model="/models/hy-mt2-30b-a3b-q6.gguf")

        self.assertIsNotNone(config.local_llm)
        self.assertEqual(config.local_llm.model_path, "/models/hy-mt2-30b-a3b-q6.gguf")
        self.assertEqual(config.chatbot.temperature, 0.7)
        self.assertEqual(config.chatbot.top_p, 1.0)
        self.assertEqual(config.chatbot.extra_body, {"top_k": -1, "repeat_penalty": 1.0})


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
