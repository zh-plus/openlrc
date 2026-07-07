#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import json
import os
import time
import unittest
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from openlrc.agents import ChunkedTranslatorAgent, ContextReviewerAgent, TranslationContext, create_chatbot
from openlrc.chatbot import ClaudeBot, GPTBot, LiteLLMBot
from openlrc.context import TranslateInfo
from openlrc.models import ModelConfig
from openlrc.prompter import ChunkedTranslatePrompter
from tests.conftest import LIVE_API, STRESS_TEST, TEST_LLM_API_KEY, TEST_MODELS


class DummyMessage(BaseModel):
    content: str


class DummyChoice(BaseModel):
    message: DummyMessage


class DummyResponse(BaseModel):
    choices: list[DummyChoice]


class TestTranslatorAgent(unittest.TestCase):
    @patch(
        "openlrc.chatbot.GPTBot.message",
        MagicMock(
            return_value=[
                DummyResponse(
                    choices=[
                        DummyChoice(
                            message=DummyMessage(
                                content="<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n"
                            )
                        )
                    ]
                )
            ]
        ),
    )
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_translate_chunk_success(self):
        bot = GPTBot(api_key="test-dummy")
        agent = ChunkedTranslatorAgent(
            src_lang="en",
            target_lang="fr",
            info=TranslateInfo(title="Example Title", audio_type="Book", glossary={"hello": "bonjour"}),
            chatbot=bot,
        )
        agent.chatbot.api_fees = [0.00035]
        translations, context = agent.translate_chunk(
            chunk_id=1,
            chunk=[(1, "Hello, how are you?"), (2, "I am fine, thank you.")],
            context=TranslationContext(
                summary="Example Summary", previous_summaries=["s1", "s2"], scene="Example Scene"
            ),
        )

        self.assertListEqual(translations, ["Bonjour, comment ça va?", "Je vais bien, merci."])
        self.assertEqual(context.summary, "Example Summary")
        self.assertEqual(context.scene, "Example Scene")

    #  Handle invalid chatbot model names gracefully
    def test_invalid_chatbot_model(self):
        with self.assertRaises(ValueError):
            create_chatbot("invalid-model")

    def test_create_chatbot_with_model_config(self):
        from openlrc.models import ModelConfig, ModelProvider

        # OpenAI ModelConfig
        config_openai = ModelConfig(
            provider=ModelProvider.OPENAI, name="gpt-4.1-nano", api_key="my-key", proxy="http://proxy"
        )
        bot = create_chatbot(config_openai)
        self.assertEqual(bot.model_name, "gpt-4.1-nano")
        self.assertEqual(bot.fee_limit, 0.8)
        self.assertIsInstance(bot, GPTBot)

        # Anthropic ModelConfig
        config_anthropic = ModelConfig(
            provider=ModelProvider.ANTHROPIC, name="claude-3-5-sonnet-20241022", api_key="dummy-key"
        )
        bot = create_chatbot(config_anthropic)
        self.assertIsInstance(bot, ClaudeBot)

        # litellm provider string → LiteLLMBot (known provider via ModelProvider coercion)
        config_litellm = ModelConfig(provider="litellm", name="gpt-4o", api_key="key")
        bot = create_chatbot(config_litellm)
        self.assertIsInstance(bot, LiteLLMBot)

        # Local llama.cpp provider → GPTBot compatibility layer with zero-cost model info
        config_local = ModelConfig(provider=ModelProvider.LOCAL_LLAMA, name="qwen3.5-9b-local", api_key="local-key")
        bot = create_chatbot(config_local)
        self.assertIsInstance(bot, GPTBot)
        self.assertEqual(bot.model_info.input_price, 0.0)
        self.assertEqual(bot.model_info.output_price, 0.0)

        # Unknown custom provider string → falls back to GPTBot
        config_custom = ModelConfig(provider="openrouter", name="gpt-4o", api_key="key")
        bot = create_chatbot(config_custom)
        self.assertIsInstance(bot, GPTBot)

        # ModelConfig with capability overrides
        config_with_extra = ModelConfig(
            provider=ModelProvider.OPENAI, name="custom-model", api_key="dummy-key",
            context_window=65536, max_tokens=4096,
        )
        bot = create_chatbot(config_with_extra)
        self.assertEqual(bot.model_config.context_window, 65536)
        self.assertEqual(bot.model_config.max_tokens, 4096)

    @patch(
        "openlrc.chatbot.GPTBot.get_content",
        MagicMock(
            return_value="<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n"
        ),
    )
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_parse_response_success(self):
        bot = GPTBot(api_key="test-dummy")
        agent = ChunkedTranslatorAgent(src_lang="en", target_lang="fr", chatbot=bot)
        translations, summary, scene = agent._parse_responses("dummy_response")

        self.assertListEqual(translations, ["Bonjour, comment ça va?", "Je vais bien, merci."])
        self.assertEqual(summary, "Example Summary")
        self.assertEqual(scene, "Example Scene")

    #  Properly format texts for translation
    def test_format_texts_success(self):
        texts = [(1, "Hello, how are you?"), (2, "I am fine, thank you.")]
        formatted_text = ChunkedTranslatePrompter.format_texts(texts)

        expected_output = (
            "#1\nOriginal>\nHello, how are you?\nTranslation>\n\n#2\nOriginal>\nI am fine, thank you.\nTranslation>\n"
        )
        self.assertEqual(formatted_text, expected_output)

    #  Use glossary terms in translations when provided
    def test_use_glossary_terms_success(self):
        glossary = {"hello": "bonjour", "how are you": "comment ça va"}
        prompter = ChunkedTranslatePrompter(src_lang="en", target_lang="fr", context=TranslateInfo(glossary=glossary))

        formatted_glossary = prompter.formatted_glossary

        expected_output = "\n# Glossary\nUse the following glossary to ensure consistency in your translations:\n<preferred-translation>\nhello: bonjour\nhow are you: comment ça va\n</preferred-translation>\n"
        self.assertEqual(formatted_glossary, expected_output)


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestContextReviewerAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")

    def test_generates_valid_context(self):
        texts = [
            "John and Sarah discuss their plan to locate a suspect",
            "John: 'As a 10 years experienced detector, my advice is we should start our search in the uptown area.'",
            "Sarah: 'Agreed. Let's gather more information before we move.'",
            "Then, they prepare to start their investigation.",
        ]
        title = "The Detectors"
        glossary = {"suspect": "嫌疑人", "uptown": "市中心"}

        bot = create_chatbot(TEST_MODELS["gemini"])
        self.addCleanup(bot.close)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot)
        context = agent.build_context(texts, title, glossary)

        self.assertIsNotNone(context)
        self.assertIsInstance(context, str)
        self.assertIn("Glossary", context)
        self.assertIn("Characters", context)
        self.assertIn("Summary", context)
        self.assertIn("Tone and Style", context)
        self.assertIn("Target Audience", context)


VALID_GUIDELINE = (
    "### Glossary:\n- suspect: 嫌疑人\n\n"
    "### Characters:\n- John: 约翰\n\n"
    "### Summary:\nA detective story.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

PARTIAL_GUIDELINE_1 = (
    "### Glossary:\n- suspect: 嫌疑人\n\n"
    "### Characters:\n- John: 约翰\n\n"
    "### Summary:\nJohn investigates.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

PARTIAL_GUIDELINE_2 = (
    "### Glossary:\n- uptown: 市中心\n\n"
    "### Characters:\n- Sarah: 萨拉\n\n"
    "### Summary:\nSarah joins the investigation.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

MERGED_GUIDELINE = (
    "### Glossary:\n- suspect: 嫌疑人\n- uptown: 市中心\n\n"
    "### Characters:\n- John: 约翰\n- Sarah: 萨拉\n\n"
    "### Summary:\nJohn and Sarah investigate together.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)


def _make_dummy_response(content: str) -> DummyResponse:
    return DummyResponse(choices=[DummyChoice(message=DummyMessage(content=content))])


class TestContextReviewerChunking(unittest.TestCase):
    """Mock tests for chunked guideline generation."""

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_short_text_no_chunking(self, mock_message):
        """Short text should use single-pass, message called once."""
        mock_message.return_value = [_make_dummy_response(VALID_GUIDELINE)]

        bot = GPTBot(api_key="test-dummy")
        bot.model_config = ModelConfig(name="test", context_window=100000)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot)
        result = agent.build_context(["Hello", "World"], title="Test")

        self.assertEqual(mock_message.call_count, 1)
        self.assertIn("Glossary", result)
        self.assertIn("Summary", result)

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_long_text_triggers_chunking(self, mock_message):
        """When context window is small and chunked_guideline=True, should split and merge."""
        mock_message.return_value = [_make_dummy_response(MERGED_GUIDELINE)]

        bot = GPTBot(api_key="test-dummy")
        bot.model_config = ModelConfig(name="test", context_window=2500, max_tokens=1024)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]
        result = agent.build_context(texts, title="Test")

        # Should have called message multiple times: N chunks + 1 merge.
        self.assertGreaterEqual(mock_message.call_count, 3)
        self.assertIn("Glossary", result)

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_long_text_without_flag_returns_empty(self, mock_message):
        """When chunked_guideline=False (default), long text should return empty without calling LLM."""
        mock_message.return_value = [_make_dummy_response(VALID_GUIDELINE)]

        bot = GPTBot(api_key="test-dummy")
        bot.model_config = ModelConfig(name="test", context_window=2500, max_tokens=1024)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot)
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]
        result = agent.build_context(texts, title="Test")

        # Should not call LLM at all, return empty guideline.
        self.assertEqual(mock_message.call_count, 0)
        self.assertEqual(result, "")

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_merge_failure_raises(self, mock_message):
        """If all merge retries fail, should raise RuntimeError."""
        from openlrc.exceptions import ChatBotException

        def side_effect_fn(*args, **kwargs):
            system_content = args[0][0]["content"] if args else ""
            if "merge" in system_content.lower():
                raise ChatBotException("merge failed")
            return [_make_dummy_response(PARTIAL_GUIDELINE_1)]

        mock_message.side_effect = side_effect_fn

        bot = GPTBot(api_key="test-dummy")
        bot.model_config = ModelConfig(name="test", context_window=2500, max_tokens=1024)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]

        with self.assertRaises(RuntimeError):
            agent.build_context(texts, title="Test")

    def test_split_texts_by_tokens(self):
        """Verify token-based splitting produces chunks within budget."""
        texts = [f"Word{i} " * 10 for i in range(20)]  # ~10 tokens each
        chunks = ContextReviewerAgent._split_texts_by_tokens(texts, max_text_tokens=50)

        self.assertGreater(len(chunks), 1)
        # All original lines should be preserved.
        flat = [line for chunk in chunks for line in chunk]
        self.assertEqual(flat, texts)


# ---------------------------------------------------------------------------
# Helpers for live chunked-guideline integration tests
# ---------------------------------------------------------------------------

LONG_SUBTITLE_PATH = Path(__file__).parent / "data" / "test_long_subtitle.json"

# Known entities in the Jensen Huang / Lex Fridman podcast that a guideline
# should capture.  These are used to measure recall, not as an exhaustive list.
EXPECTED_CHARACTERS = {"Jensen Huang", "Lex Fridman", "Elon Musk"}
EXPECTED_GLOSSARY_TERMS = {"CUDA", "NVLink", "TSMC", "GPU", "scaling law", "AGI"}
REQUIRED_SECTIONS = ["Glossary", "Characters", "Summary", "Tone and Style", "Target Audience"]


def _recall(guideline: str, expected: set[str]) -> tuple[float, set[str], set[str]]:
    """Check what fraction of *expected* terms appear anywhere in *guideline* (case-insensitive).

    Returns (recall, found, missed) so callers can log details.
    This avoids fragile markdown-section parsing -- we simply search the full
    guideline text for each expected term.
    """
    if not expected:
        return 1.0, set(), set()
    lowered = guideline.lower()
    found = {e for e in expected if e.lower() in lowered}
    missed = expected - found
    return len(found) / len(expected), found, missed


def _has_all_sections(guideline: str) -> bool:
    lowered = guideline.lower()
    return all(s.lower() in lowered for s in REQUIRED_SECTIONS)


def _print_guideline(label: str, guideline: str) -> None:
    """Print the full guideline text with a clear header/footer for easy reading."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"  GUIDELINE OUTPUT: {label}")
    print(separator)
    print(guideline)
    print(separator)


@unittest.skipUnless(
    STRESS_TEST, "Requires OPENLRC_TEST_STRESS=1 (uses live LLM API with simulated small context windows)"
)
class TestChunkedGuidelineLive(unittest.TestCase):
    """Live integration tests for chunked guideline generation.

    Uses a ~30 000-token podcast transcript (Jensen Huang on Lex Fridman #494)
    to verify that chunked generation produces a guideline comparable to
    single-pass generation.  Addresses the review concerns in PR #103:
      1. Do characters / glossary terms survive the merge?
      2. Is the summary coherent?
      3. What are the latency and success rate?
    """

    lines: list[str]
    title: str
    _baseline_guideline: str = ""
    _chunked_guideline: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")
        if not LONG_SUBTITLE_PATH.exists():
            raise unittest.SkipTest(f"Test fixture not found: {LONG_SUBTITLE_PATH}")

        with open(LONG_SUBTITLE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        cls.lines = data["lines"]
        cls.title = data["title"]

    # -- Test A: baseline single-pass with large context window ----------------

    def test_baseline_single_pass(self) -> None:
        """Large-window model generates a valid guideline in one pass."""
        bot = create_chatbot(TEST_MODELS["gemini"])
        self.addCleanup(bot.close)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot)

        t0 = time.monotonic()
        guideline = agent.build_context(self.lines, title=self.title)
        elapsed = time.monotonic() - t0

        self.assertTrue(_has_all_sections(guideline), "Baseline guideline missing required sections")

        g_recall, g_found, g_missed = _recall(guideline, EXPECTED_GLOSSARY_TERMS)
        c_recall, c_found, c_missed = _recall(guideline, EXPECTED_CHARACTERS)
        print(f"\n[baseline] time={elapsed:.1f}s")
        print(f"[baseline] glossary recall={g_recall:.2f}  found={g_found}  missed={g_missed}")
        print(f"[baseline] character recall={c_recall:.2f}  found={c_found}  missed={c_missed}")
        _print_guideline("baseline (single-pass)", guideline)

        self.__class__._baseline_guideline = guideline

    # -- Test B: chunked generation with simulated 16K window ------------------

    @unittest.skipUnless(STRESS_TEST, "Requires OPENLRC_TEST_STRESS=1 (tokenizer mismatch may cause failures on CI)")
    def test_chunked_16k_window(self) -> None:
        """Simulated 16K window triggers ~3 chunks; closest to the real scenario in PR #103."""
        model = copy(TEST_MODELS["gemini"])
        model.context_window = 16384
        model.max_tokens = 4096

        bot = create_chatbot(model)
        self.addCleanup(bot.close)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)

        t0 = time.monotonic()
        guideline = agent.build_context(self.lines, title=self.title)
        elapsed = time.monotonic() - t0

        # Hard requirement: chunked path must produce *something*.
        self.assertTrue(len(guideline) > 0, "Chunked 16K guideline is empty")

        # Soft metrics: log for review but do not fail the test.
        has_sections = _has_all_sections(guideline)
        g_recall, g_found, g_missed = _recall(guideline, EXPECTED_GLOSSARY_TERMS)
        c_recall, c_found, c_missed = _recall(guideline, EXPECTED_CHARACTERS)
        print(f"\n[chunked-16k] time={elapsed:.1f}s  all_sections={has_sections}")
        print(f"[chunked-16k] glossary recall={g_recall:.2f}  found={g_found}  missed={g_missed}")
        print(f"[chunked-16k] character recall={c_recall:.2f}  found={c_found}  missed={c_missed}")
        _print_guideline("chunked (16K window, ~3 chunks)", guideline)

        self.__class__._chunked_guideline = guideline

    # -- Test C: chunked generation with simulated 8K window -------------------

    def test_chunked_8k_window(self) -> None:
        """Simulated 8K window triggers ~6 chunks; merged guideline should be non-empty."""
        model = copy(TEST_MODELS["gemini"])
        model.context_window = 8192
        model.max_tokens = 2048

        bot = create_chatbot(model)
        self.addCleanup(bot.close)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)

        t0 = time.monotonic()
        guideline = agent.build_context(self.lines, title=self.title)
        elapsed = time.monotonic() - t0

        # Hard requirement: chunked path must produce *something*.
        self.assertTrue(len(guideline) > 0, "Chunked guideline is empty")

        # Soft metrics: log for review but do not fail the test.
        has_sections = _has_all_sections(guideline)
        g_recall, g_found, g_missed = _recall(guideline, EXPECTED_GLOSSARY_TERMS)
        c_recall, c_found, c_missed = _recall(guideline, EXPECTED_CHARACTERS)
        print(f"\n[chunked-8k] time={elapsed:.1f}s  all_sections={has_sections}")
        print(f"[chunked-8k] glossary recall={g_recall:.2f}  found={g_found}  missed={g_missed}")
        print(f"[chunked-8k] character recall={c_recall:.2f}  found={c_found}  missed={c_missed}")
        _print_guideline("chunked (8K window, ~6 chunks)", guideline)

        self.__class__._chunked_guideline = guideline

    # -- Test D: quality comparison between baseline and chunked ---------------

    def test_quality_comparison(self) -> None:
        """Log quality comparison between baseline and chunked guidelines (informational only)."""
        baseline = self.__class__._baseline_guideline
        chunked = self.__class__._chunked_guideline

        if not baseline or not chunked:
            self.skipTest("Baseline or chunked results not available (earlier test may have failed)")

        # All metrics are logged for PR review but do not cause test failure.
        bg_recall, bg_found, bg_missed = _recall(baseline, EXPECTED_GLOSSARY_TERMS)
        cg_recall, cg_found, cg_missed = _recall(chunked, EXPECTED_GLOSSARY_TERMS)
        bc_recall, bc_found, bc_missed = _recall(baseline, EXPECTED_CHARACTERS)
        cc_recall, cc_found, cc_missed = _recall(chunked, EXPECTED_CHARACTERS)

        print(f"\n[comparison] baseline glossary recall={bg_recall:.2f}  found={bg_found}  missed={bg_missed}")
        print(f"[comparison] chunked  glossary recall={cg_recall:.2f}  found={cg_found}  missed={cg_missed}")
        print(f"[comparison] baseline character recall={bc_recall:.2f}  found={bc_found}  missed={bc_missed}")
        print(f"[comparison] chunked  character recall={cc_recall:.2f}  found={cc_found}  missed={cc_missed}")
        print(f"[comparison] baseline all_sections={_has_all_sections(baseline)}")
        print(f"[comparison] chunked  all_sections={_has_all_sections(chunked)}")

    # -- Test E: latency and reliability (3 runs) -----------------------------

    def test_latency_and_reliability(self) -> None:
        """Run chunked generation multiple times and log latency/success statistics."""
        model = copy(TEST_MODELS["gemini"])
        model.context_window = 8192
        model.max_tokens = 2048

        runs = 3
        results: list[dict] = []

        for i in range(runs):
            bot = create_chatbot(model)
            self.addCleanup(bot.close)
            agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)
            t0 = time.monotonic()
            try:
                guideline = agent.build_context(self.lines, title=self.title)
                elapsed = time.monotonic() - t0
                results.append(
                    {
                        "run": i + 1,
                        "success": True,
                        "time_sec": round(elapsed, 1),
                        "has_all_sections": _has_all_sections(guideline),
                        "output_length": len(guideline),
                    }
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                results.append({"run": i + 1, "success": False, "time_sec": round(elapsed, 1), "error": str(e)})

        successes = sum(1 for r in results if r["success"])
        times = [r["time_sec"] for r in results if r["success"]]

        print(f"\n[reliability] {successes}/{runs} succeeded")
        for r in results:
            print(f"  run {r['run']}: {r}")
        if times:
            print(f"  avg={sum(times) / len(times):.1f}s  min={min(times):.1f}s  max={max(times):.1f}s")

        # Only require that the feature doesn't crash every time.
        self.assertGreaterEqual(successes, 1, f"All runs failed: {results}")

    # -- Test F: stress test with 4K window (hierarchical merge) ---------------

    def test_stress_4k_window(self) -> None:
        """4K window forces ~15 chunks and hierarchical merging; must not crash."""
        model = copy(TEST_MODELS["gemini"])
        model.context_window = 4096
        model.max_tokens = 1024

        bot = create_chatbot(model)
        self.addCleanup(bot.close)
        agent = ContextReviewerAgent("en", "zh", chatbot=bot, chunked_guideline=True)

        t0 = time.monotonic()
        guideline = agent.build_context(self.lines, title=self.title)
        elapsed = time.monotonic() - t0

        # Hard requirement: must produce *something* without crashing.
        self.assertTrue(len(guideline) > 0, "Stress-test guideline is empty")

        # Soft metrics: log for review.
        has_sections = _has_all_sections(guideline)
        g_recall, g_found, g_missed = _recall(guideline, EXPECTED_GLOSSARY_TERMS)
        c_recall, c_found, c_missed = _recall(guideline, EXPECTED_CHARACTERS)
        print(f"\n[stress-4k] time={elapsed:.1f}s  all_sections={has_sections}")
        print(f"[stress-4k] glossary recall={g_recall:.2f}  found={g_found}  missed={g_missed}")
        print(f"[stress-4k] character recall={c_recall:.2f}  found={c_found}  missed={c_missed}")
        _print_guideline("chunked (4K window, ~15 chunks, hierarchical merge)", guideline)
