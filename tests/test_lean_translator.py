#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openlrc.agents import create_chatbot
from openlrc.llama_resources import HY_MT2_PROMPT_PROFILE
from openlrc.media_utils import get_similarity
from openlrc.prompter import (
    LeanContextReviewPrompter,
    LeanTranslatePrompter,
    create_atomic_translate_prompter,
    create_lean_translate_prompter,
)
from openlrc.translate import LeanTranslator
from openlrc.validators import HyMT2DelimiterTranslateValidator, LeanTranslateValidator
from tests.conftest import LIVE_API, TEST_LLM_API_KEY, TEST_MODELS


def _make_mock_chatbot(name: str = "gpt-4.1-nano") -> MagicMock:
    bot = MagicMock()
    bot.model_name = name
    bot.api_fees = [0.0]
    bot.close = MagicMock()
    return bot


class TestLeanTranslateValidator(unittest.TestCase):
    """Unit tests for anchor parsing and validation."""

    def test_parse_basic(self):
        content = "#1\nHello\n#2\nWorld\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed, {1: "Hello", 2: "World"})

    def test_parse_multiline_translation_joined(self):
        content = "#1\nLine one\nLine two\n#2\nSingle\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed[1], "Line one Line two")
        self.assertEqual(parsed[2], "Single")

    def test_parse_ignores_hash_in_translation(self):
        """#<digits> followed by non-whitespace is NOT an anchor."""
        content = "#1\n#hashtag is trending\n#2\nWorld\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed[1], "#hashtag is trending")
        self.assertEqual(parsed[2], "World")

    def test_parse_empty_translation_skipped(self):
        content = "#1\n\n#2\nWorld\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertNotIn(1, parsed)
        self.assertEqual(parsed[2], "World")

    def test_parse_preamble_ignored(self):
        content = "Some preamble text\n#1\nHello\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed, {1: "Hello"})

    def test_parse_angle_bracket_anchor(self):
        content = "#<1>\nHello\n#<2>\nWorld\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed, {1: "Hello", 2: "World"})

    def test_parse_pure_markup_tag_skipped(self):
        content = "#1\n</翻译>\n#2\nWorld\n"
        parsed = LeanTranslateValidator.parse_anchored_translations(content)
        self.assertEqual(parsed, {2: "World"})

    def test_validate_passes_above_threshold(self):
        validator = LeanTranslateValidator(expected_ids=[1, 2, 3, 4, 5])
        content = "#1\nA\n#2\nB\n#3\nC\n#4\nD\n"  # 4/5 = 80%
        self.assertTrue(validator.validate("", content))

    def test_validate_fails_below_threshold(self):
        validator = LeanTranslateValidator(expected_ids=[1, 2, 3, 4, 5])
        content = "#1\nA\n#2\nB\n#3\nC\n"  # 3/5 = 60%
        self.assertFalse(validator.validate("", content))

    def test_validate_empty_content(self):
        validator = LeanTranslateValidator(expected_ids=[1, 2])
        self.assertFalse(validator.validate("", ""))
        self.assertFalse(validator.validate("", None))  # type: ignore[arg-type]


class TestHyMT2DelimiterTranslateValidator(unittest.TestCase):
    def test_parse_delimited_translations(self):
        content = '<seg id="1">你好</seg>\n<seg id="2">世界</seg>'
        parsed = HyMT2DelimiterTranslateValidator.parse_delimited_translations(content)
        self.assertEqual(parsed, {1: "你好", 2: "世界"})

    def test_parse_multiline_and_extra_text(self):
        content = 'Note:\n<seg id="1">第一行\n第二行</seg>\nDone.'
        parsed = HyMT2DelimiterTranslateValidator.parse_delimited_translations(content)
        self.assertEqual(parsed, {1: "第一行 第二行"})

    def test_parse_html_entities(self):
        content = '<seg id="1">Tom &amp; Jerry &lt;3</seg>'
        parsed = HyMT2DelimiterTranslateValidator.parse_delimited_translations(content)
        self.assertEqual(parsed, {1: "Tom & Jerry <3"})

    def test_validate_match_ratio(self):
        validator = HyMT2DelimiterTranslateValidator(expected_ids=[1, 2, 3, 4, 5])
        self.assertTrue(
            validator.validate("", '<seg id="1">A</seg><seg id="2">B</seg><seg id="3">C</seg><seg id="4">D</seg>')
        )
        self.assertFalse(validator.validate("", '<seg id="1">A</seg><seg id="2">B</seg><seg id="3">C</seg>'))

    def test_validate_accepts_anchor_fallback(self):
        validator = HyMT2DelimiterTranslateValidator(expected_ids=[1, 2])
        self.assertTrue(validator.validate("", "#<1>\n你好\n#<2>\n世界\n"))

    def test_validate_rejects_duplicate_unexpected_and_out_of_order_ids(self):
        validator = HyMT2DelimiterTranslateValidator(expected_ids=[1, 2, 3])

        self.assertFalse(validator.validate("", '<seg id="1">A</seg><seg id="1">B</seg>'))
        self.assertFalse(validator.validate("", '<seg id="1">A</seg><seg id="9">B</seg>'))
        self.assertFalse(validator.validate("", '<seg id="2">B</seg><seg id="1">A</seg>'))
        self.assertEqual(len(validator.issues), 3)
        self.assertIn("Duplicate", validator.issues[0])
        self.assertIn("Unexpected", validator.issues[1])
        self.assertIn("out of order", validator.issues[2])


class TestLeanTranslatePrompter(unittest.TestCase):
    def setUp(self):
        self.prompter = LeanTranslatePrompter("en", "zh")

    def test_system_prompt_contains_languages(self):
        system = self.prompter.system()
        self.assertIn("English", system)
        self.assertIn("Chinese", system)

    def test_format_texts(self):
        texts = [(1, "Hello"), (2, "World")]
        formatted = self.prompter.format_texts(texts)
        self.assertEqual(formatted, "#1\nHello\n#2\nWorld")

    def test_user_prompt_full_context(self):
        user = self.prompter.user(
            "#1\nHello",
            summary="A greeting scene.",
            characters="- John: 约翰, detective",
            terminology="- hello: 你好",
            sliding_window="#0 Hi | 嗨",
        )
        self.assertIn("[Context]", user)
        self.assertIn("Summary: A greeting scene.", user)
        self.assertIn("Characters:", user)
        self.assertIn("- John: 约翰, detective", user)
        self.assertIn("Terminology:", user)
        self.assertIn("- hello: 你好", user)
        self.assertIn("Recent translations:", user)
        self.assertIn("#0 Hi | 嗨", user)
        self.assertIn("#1\nHello", user)

    def test_user_prompt_no_context(self):
        user = self.prompter.user("#1\nHello")
        self.assertNotIn("[Context]", user)
        self.assertIn("#1\nHello", user)

    def test_user_prompt_partial_context(self):
        user = self.prompter.user("#1\nHello", summary="A scene.")
        self.assertIn("[Context]", user)
        self.assertIn("Summary: A scene.", user)
        self.assertNotIn("Characters:", user)
        self.assertNotIn("Terminology:", user)
        self.assertNotIn("Recent translations:", user)

    def test_context_layer_order(self):
        """Context layers appear in order: Summary > Characters > Terminology > Recent."""
        user = self.prompter.user(
            "#1\nHello", summary="sum", characters="chars", terminology="terms", sliding_window="window"
        )
        idx_summary = user.index("Summary:")
        idx_chars = user.index("Characters:")
        idx_terms = user.index("Terminology:")
        idx_recent = user.index("Recent translations:")
        self.assertLess(idx_summary, idx_chars)
        self.assertLess(idx_chars, idx_terms)
        self.assertLess(idx_terms, idx_recent)

    def test_update_expected_ids(self):
        self.assertIsNone(self.prompter.validator)
        self.prompter.update_expected_ids([1, 2, 3])
        self.assertIsNotNone(self.prompter.validator)
        self.assertEqual(self.prompter.validator.expected_ids, {1, 2, 3})


class TestHyMT2TranslatePrompter(unittest.TestCase):
    def test_lean_prompt_omits_system_message(self):
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)
        formatted = prompter.format_texts([(1, "Hello & <world>")])
        user = prompter.user(formatted, terminology="- OpenLRC: OpenLRC")
        messages = LeanTranslator._prompt_messages(prompter, user)

        self.assertIsNone(prompter.system())
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertEqual(formatted, '<seg id="1">Hello &amp; &lt;world&gt;</seg>')
        self.assertIn("Please accurately translate the following subtitle text", user)
        self.assertIn("without any additional explanation", user)
        self.assertIn('Retain every opening <seg id="N"> delimiter', user)
        self.assertIn("Output exactly 1 <seg> blocks", user)
        self.assertNotIn("#<id>", user)
        self.assertIn('<seg id="1">Hello &amp; &lt;world&gt;</seg>', user)

    def test_lean_prompt_parses_delimited_output(self):
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)

        parsed = prompter.parse_translations('<seg id="1">你好</seg>')

        self.assertEqual(parsed, {1: "你好"})

    def test_lean_prompt_falls_back_to_anchor_parser(self):
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)

        parsed = prompter.parse_translations("#<1>\n你好\n")

        self.assertEqual(parsed, {1: "你好"})

    def test_lean_prompt_retry_instruction_uses_delimiters(self):
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)

        self.assertIn('<seg id="N">translated text</seg>', prompter.retry_instruction())

    def test_atomic_prompt_uses_hy_mt2_official_style(self):
        prompter = create_atomic_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)
        user = prompter.user("Hello")

        self.assertIn("Translate the following text into Chinese", user)
        self.assertIn("only output the translated result", user)
        self.assertIn("Hello", user)


class TestLeanContextReviewPrompter(unittest.TestCase):
    def setUp(self):
        self.prompter = LeanContextReviewPrompter("en", "zh")

    def test_expected_sections(self):
        self.assertEqual(self.prompter.expected_sections, ["glossary", "characters", "summary"])

    def test_system_prompt_contains_languages(self):
        system = self.prompter.system()
        self.assertIn("English", system)
        self.assertIn("Chinese", system)

    def test_system_prompt_requests_yaml_format(self):
        system = self.prompter.system()
        self.assertIn("glossary:", system)
        self.assertIn("characters:", system)
        self.assertIn("summary:", system)
        # Should NOT request tone/style or target audience
        self.assertNotIn("tone and style", system.lower())
        self.assertNotIn("target audience", system.lower())

    def test_user_prompt(self):
        user = self.prompter.user("Hello world", title="Test Movie")
        self.assertIn("Test Movie", user)
        self.assertIn("Hello world", user)
        self.assertIn("glossary, characters, and summary", user)

    def test_user_partial_prompt(self):
        user = self.prompter.user_partial("Hello", chunk_index=1, total_chunks=3, title="Test")
        self.assertIn("Section 1 of 3", user)
        self.assertIn("Test", user)

    def test_merge_system_prompt(self):
        system = self.prompter.merge_system()
        self.assertIn("glossary", system)
        self.assertIn("characters", system)
        self.assertIn("summary", system)
        self.assertNotIn("tone", system.lower())

    def test_merge_user_prompt(self):
        user = self.prompter.merge_user(["guideline 1", "guideline 2"], title="Test")
        self.assertIn("Partial guideline 1", user)
        self.assertIn("Partial guideline 2", user)
        self.assertIn("Test", user)

    def test_stop_sequence(self):
        self.assertEqual(self.prompter.stop_sequence, "<--END-OF-CONTEXT-->")


class TestAnchorAlignment(unittest.TestCase):
    def _make_translator(self):
        return LeanTranslator(chatbot=_make_mock_chatbot())

    def test_exact_match(self):
        t = self._make_translator()
        aligned, missing = t._align_translations([1, 2, 3], {1: "A", 2: "B", 3: "C"})
        self.assertEqual(aligned, ["A", "B", "C"])
        self.assertEqual(missing, [])

    def test_offset_match(self):
        t = self._make_translator()
        aligned, missing = t._align_translations([1, 2, 3], {2: "A", 3: "B", 4: "C"})
        self.assertEqual(aligned, ["A", "B", "C"])
        self.assertEqual(missing, [])

    def test_partial_missing(self):
        t = self._make_translator()
        aligned, missing = t._align_translations([1, 2, 3, 4, 5], {1: "A", 2: "B", 3: "C", 4: "D"})
        self.assertEqual(aligned, ["A", "B", "C", "D", None])
        self.assertEqual(missing, [5])

    def test_all_missing(self):
        t = self._make_translator()
        aligned, missing = t._align_translations([1, 2, 3], {})
        self.assertEqual(aligned, [None, None, None])
        self.assertEqual(missing, [1, 2, 3])

    def test_offset_beyond_tolerance(self):
        t = self._make_translator()
        t.ANCHOR_OFFSET_TOLERANCE = 2
        aligned, missing = t._align_translations([1], {100: "A"})
        self.assertEqual(aligned, [None])
        self.assertEqual(missing, [1])


class TestSlidingWindow(unittest.TestCase):
    def test_basic(self):
        pairs = [(1, "Hello", "你好"), (2, "World", "世界")]
        result = LeanTranslator._build_sliding_window(pairs, budget=1000)
        self.assertIn("#1 Hello | 你好", result)
        self.assertIn("#2 World | 世界", result)

    def test_budget_truncation(self):
        pairs = [(i, f"word{i}", f"词{i}") for i in range(100)]
        result = LeanTranslator._build_sliding_window(pairs, budget=30)
        lines = result.strip().splitlines()
        self.assertTrue(len(lines) < 100)
        self.assertIn("#99", result)

    def test_empty_pairs(self):
        result = LeanTranslator._build_sliding_window([], budget=150)
        self.assertEqual(result, "")


class TestExtractCRContext(unittest.TestCase):
    def test_yaml_format(self):
        """Parse YAML-like output from LeanContextReviewPrompter."""
        guideline = """glossary:
  - suspect: 嫌疑人
  - uptown: 市中心
characters:
  - John: 约翰, a detective
  - Sarah: 萨拉, partner
summary:
John investigates a case in the uptown area."""
        summary, characters, terminology = LeanTranslator._extract_cr_context(guideline)
        self.assertIn("John investigates", summary)
        self.assertIn("约翰", characters)
        self.assertIn("萨拉", characters)
        self.assertIn("suspect: 嫌疑人", terminology)

    def test_markdown_format(self):
        """Parse ### markdown output from ContextReviewPrompter."""
        guideline = """### Glossary:
- suspect: 嫌疑人
- uptown: 市中心

### Characters:
- John: 约翰, a detective

### Summary:
John investigates a case in the uptown area.

### Tone and Style:
Formal and professional."""
        summary, characters, terminology = LeanTranslator._extract_cr_context(guideline)
        self.assertIn("John investigates", summary)
        self.assertIn("约翰", characters)
        self.assertIn("suspect: 嫌疑人", terminology)

    def test_missing_sections(self):
        guideline = "characters:\n- John: 约翰"
        summary, characters, terminology = LeanTranslator._extract_cr_context(guideline)
        self.assertEqual(summary, "")
        self.assertEqual(terminology, "")
        self.assertIn("约翰", characters)

    def test_yaml_single_line_summary(self):
        """summary: <text> on the same line should be extracted."""
        guideline = """glossary:
  - suspect: 嫌疑人
characters:
  - John: 约翰
summary: John investigates a case."""
        summary, characters, terminology = LeanTranslator._extract_cr_context(guideline)
        self.assertIn("John investigates", summary)
        self.assertIn("约翰", characters)
        self.assertIn("suspect: 嫌疑人", terminology)

    def test_empty_guideline(self):
        summary, characters, terminology = LeanTranslator._extract_cr_context("")
        self.assertEqual(summary, "")
        self.assertEqual(characters, "")
        self.assertEqual(terminology, "")


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
class TestLeanTranslatorTranslate(unittest.TestCase):
    def _make_translator(self, enable_cr=True, cr_chatbot=None):
        return LeanTranslator(chatbot=_make_mock_chatbot(), enable_cr=enable_cr, cr_chatbot=cr_chatbot)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_single_chunk_no_cr(self, mock_reviewer_cls):
        """With enable_cr=False, CR is skipped and translation proceeds."""
        translator = self._make_translator(enable_cr=False)
        texts = ["Hello", "World"]

        response_mock = MagicMock()
        translator.chatbot.get_content.return_value = "#1\n你好\n#2\n世界\n"
        translator.chatbot.message.return_value = [response_mock]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        mock_reviewer_cls.assert_not_called()

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_single_chunk_with_cr(self, mock_reviewer_cls):
        """With enable_cr=True, CR runs and guideline is used."""
        translator = self._make_translator(enable_cr=True)
        texts = ["Hello", "World"]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = (
            "glossary:\n  - hello: 你好\ncharacters:\n  - Bob: 鲍勃\nsummary:\nA greeting.\n"
        )

        response_mock = MagicMock()
        translator.chatbot.get_content.return_value = "#1\n你好\n#2\n世界\n"
        translator.chatbot.message.return_value = [response_mock]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        mock_reviewer.build_context.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_cr_passes_lean_prompter(self, mock_reviewer_cls):
        """CR agent is created with LeanContextReviewPrompter."""
        translator = self._make_translator(enable_cr=True)
        texts = ["Hello"]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "summary:\nA greeting.\n"

        translator.chatbot.get_content.return_value = "#1\n你好\n"
        translator.chatbot.message.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", compare_path=compare_path)

        call_kwargs = mock_reviewer_cls.call_args
        prompter_arg = call_kwargs.kwargs.get("prompter")
        self.assertIsInstance(prompter_arg, LeanContextReviewPrompter)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_atomic_fill_on_partial_missing(self, mock_reviewer_cls):
        """When ≤20% lines are missing, atomic translation fills the gaps."""
        translator = self._make_translator(enable_cr=False)
        texts = [f"line{i}" for i in range(10)]

        anchored = "\n".join(f"#{i + 1}\ntrans{i + 1}" for i in range(9))
        translator.chatbot.get_content.return_value = anchored
        translator.chatbot.message.return_value = [MagicMock()]

        with patch.object(translator, "atomic_translate", return_value=["atomic10"]) as mock_atomic:
            with tempfile.TemporaryDirectory() as tmpdir:
                compare_path = Path(tmpdir) / "compare.json"
                result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[9], "atomic10")
        mock_atomic.assert_called_once()

    def test_try_single_attempt_uses_prompter_parser(self):
        translator = self._make_translator(enable_cr=False)
        prompter = LeanTranslatePrompter("en", "zh")
        prompter.parse_translations = MagicMock(return_value={1: "你好"})  # type: ignore[method-assign]
        translator.chatbot.get_content.return_value = "custom parser payload"
        translator.chatbot.message.return_value = [MagicMock()]

        result = translator._try_single_attempt(
            translator.chatbot, prompter, [{"role": "user", "content": "Translate"}], [1], {1: "Hello"}, "en", "zh"
        )

        self.assertEqual(result, (["你好"], False))
        prompter.parse_translations.assert_called_once_with("custom parser payload")

    def test_delimiter_atomic_fill_on_partial_missing(self):
        translator = self._make_translator(enable_cr=False)
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)
        prompter.update_expected_ids([1, 2, 3, 4, 5])
        translator.chatbot.get_content.return_value = (
            '<seg id="1">一</seg><seg id="2">二</seg><seg id="3">三</seg><seg id="4">四</seg>'
        )
        translator.chatbot.message.return_value = [MagicMock()]

        with patch.object(translator, "atomic_translate", return_value=["五"]) as mock_atomic:
            result = translator._try_single_attempt(
                translator.chatbot,
                prompter,
                [{"role": "user", "content": "Translate"}],
                [1, 2, 3, 4, 5],
                {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"},
                "en",
                "zh-cn",
            )

        self.assertEqual(result, (["一", "二", "三", "四", "五"], True))
        mock_atomic.assert_called_once_with(translator.chatbot, ["five"], "en", "zh-cn")

    def test_delimiter_alignment_does_not_shift_missing_middle_id(self):
        translator = self._make_translator(enable_cr=False)
        prompter = create_lean_translate_prompter("en", "zh-cn", HY_MT2_PROMPT_PROFILE)
        translator.chatbot.get_content.return_value = (
            '<seg id="1">一</seg><seg id="2">二</seg><seg id="4">四</seg><seg id="5">五</seg>'
        )
        translator.chatbot.message.return_value = [MagicMock()]

        with patch.object(translator, "atomic_translate", return_value=["三"]):
            result = translator._try_single_attempt(
                translator.chatbot,
                prompter,
                [{"role": "user", "content": "Translate"}],
                [1, 2, 3, 4, 5],
                {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"},
                "en",
                "zh-cn",
            )

        self.assertEqual(result, (["一", "二", "三", "四", "五"], True))

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_full_atomic_fallback_on_total_failure(self, mock_reviewer_cls):
        """When all retries fail, full atomic translation is used."""
        translator = self._make_translator(enable_cr=False)
        translator.MAX_CHUNK_RETRIES = 1
        texts = ["Hello", "World"]

        translator.chatbot.get_content.return_value = ""
        translator.chatbot.message.return_value = [MagicMock()]

        with patch.object(translator, "atomic_translate", return_value=["你好", "世界"]) as mock_atomic:
            with tempfile.TemporaryDirectory() as tmpdir:
                compare_path = Path(tmpdir) / "compare.json"
                result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        mock_atomic.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_checkpoint_resume(self, mock_reviewer_cls):
        """Translation resumes from saved checkpoint."""
        translator = self._make_translator(enable_cr=False)
        translator.chunk_size = 3
        texts = [f"text{i}" for i in range(6)]

        saved_state = {
            "compare": [
                {"chunk": 1, "idx": i + 1, "method": "chunked", "model": "None", "input": f"text{i}", "output": f"t{i}"}
                for i in range(3)
            ],
            "guideline": "",
            "recent_pairs": [(i + 1, f"text{i}", f"t{i}") for i in range(3)],
        }

        anchored = "#4\nt3\n#5\nt4\n#6\nt5\n"
        translator.chatbot.get_content.return_value = anchored
        translator.chatbot.message.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            with open(compare_path, "w") as f:
                json.dump(saved_state, f)

            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(len(result), 6)
        self.assertEqual(result[:3], ["t0", "t1", "t2"])

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_user_glossary_injected_without_cr(self, mock_reviewer_cls):
        """User glossary appears in translation prompt even when CR is disabled."""
        from openlrc.context import TranslateInfo

        translator = self._make_translator(enable_cr=False)
        texts = ["Hello"]
        info = TranslateInfo(glossary={"hello": "你好", "world": "世界"})

        captured_messages: list = []

        def capture_message(msgs, **kwargs):
            captured_messages.append(msgs)
            return [MagicMock()]

        translator.chatbot.message.side_effect = capture_message
        translator.chatbot.get_content.return_value = "#1\n你好\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", info=info, compare_path=compare_path)

        # The user prompt (second message in the list) should contain glossary terms
        self.assertTrue(len(captured_messages) > 0)
        user_content = captured_messages[0][1]["content"]
        self.assertIn("hello: 你好", user_content)
        self.assertIn("world: 世界", user_content)
        mock_reviewer_cls.assert_not_called()

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_user_glossary_merged_with_cr(self, mock_reviewer_cls):
        """User glossary is merged with CR-generated terminology."""
        from openlrc.context import TranslateInfo

        translator = self._make_translator(enable_cr=True)
        texts = ["Hello"]
        info = TranslateInfo(glossary={"world": "世界"})

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = (
            "glossary:\n  - hello: 你好\ncharacters:\n  - Bob: 鲍勃\nsummary:\nA greeting.\n"
        )

        captured_messages: list = []

        def capture_message(msgs, **kwargs):
            captured_messages.append(msgs)
            return [MagicMock()]

        translator.chatbot.message.side_effect = capture_message
        translator.chatbot.get_content.return_value = "#1\n你好\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", info=info, compare_path=compare_path)

        self.assertTrue(len(captured_messages) > 0)
        user_content = captured_messages[0][1]["content"]
        # Both CR-generated and user-provided glossary should be present
        self.assertIn("hello: 你好", user_content)
        self.assertIn("world: 世界", user_content)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_retry_chatbot_used_on_primary_failure(self, mock_reviewer_cls):
        """When primary chatbot returns >20% missing, retry_chatbot is tried."""
        retry_bot = _make_mock_chatbot("retry-model")
        translator = LeanTranslator(chatbot=_make_mock_chatbot(), retry_chatbot=retry_bot, enable_cr=False)
        texts = ["Hello", "World"]

        # Primary chatbot returns empty (total failure)
        translator.chatbot.get_content.return_value = ""
        translator.chatbot.message.return_value = [MagicMock()]

        # Retry chatbot succeeds
        retry_bot.get_content.return_value = "#1\n你好\n#2\n世界\n"
        retry_bot.message.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        retry_bot.message.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_binary_split_on_persistent_failure(self, mock_reviewer_cls):
        """When all chatbot retries fail, binary split translates halves."""
        translator = self._make_translator(enable_cr=False)
        translator.MAX_CHUNK_RETRIES = 1
        texts = [f"line{i}" for i in range(6)]

        get_count = 0

        def mock_get_content(resp):
            nonlocal get_count
            get_count += 1
            # First get_content call (full chunk attempt): empty → failure
            if get_count == 1:
                return ""
            # All subsequent calls (binary split halves): succeed
            return "#1\nt0\n#2\nt1\n#3\nt2\n#4\nt3\n#5\nt4\n#6\nt5\n"

        translator.chatbot.message.return_value = [MagicMock()]
        translator.chatbot.get_content.side_effect = mock_get_content

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(len(result), 6)
        self.assertEqual(translator.metrics["splits"], 1)
        self.assertEqual(translator.metrics["max_split_depth"], 1)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_glossary_removal_retry(self, mock_reviewer_cls):
        """On retry, glossary-removal variant is used when terminology is present."""
        from openlrc.context import TranslateInfo

        translator = self._make_translator(enable_cr=False)
        translator.MAX_CHUNK_RETRIES = 2
        texts = ["Hello"]
        info = TranslateInfo(glossary={"hello": "你好"})

        captured_messages: list = []

        def capture_message(msgs, **kwargs):
            captured_messages.append(msgs)
            return [MagicMock()]

        translator.chatbot.message.side_effect = capture_message

        call_count = 0

        def mock_get_content(resp):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""  # First attempt fails
            return "#1\n你好\n"  # Second attempt succeeds

        translator.chatbot.get_content.side_effect = mock_get_content

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", info=info, compare_path=compare_path)

        self.assertEqual(result, ["你好"])
        # Second attempt should NOT contain the glossary terminology
        self.assertTrue(len(captured_messages) >= 2)
        second_user = captured_messages[1][1]["content"]
        self.assertNotIn("hello: 你好", second_user)
        # And should contain the retry instruction
        retry_msg = captured_messages[1][-1]["content"]
        self.assertIn("formatting issues", retry_msg)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_cr_uses_cr_chatbot_when_provided(self, mock_reviewer_cls):
        """When cr_chatbot is provided, CR uses it instead of chatbot."""
        cr_bot = _make_mock_chatbot("cr-model")
        translator = self._make_translator(enable_cr=True, cr_chatbot=cr_bot)
        texts = ["Hello"]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "summary:\nA greeting.\n"

        translator.chatbot.get_content.return_value = "#1\n你好\n"
        translator.chatbot.message.return_value = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", compare_path=compare_path)

        call_kwargs = mock_reviewer_cls.call_args
        self.assertIs(call_kwargs.kwargs.get("chatbot") or call_kwargs[1].get("chatbot"), cr_bot)

    @patch("openlrc.translate.ContextReviewerAgent")
    def test_cr_fee_tracked_when_chatbot_used_for_cr(self, mock_reviewer_cls):
        """When enable_cr=True and cr_chatbot=None, CR uses chatbot and its fee is tracked."""
        translator = self._make_translator(enable_cr=True)
        texts = ["Hello"]

        # Simulate chatbot accumulating fees during CR and translation
        translator.chatbot.api_fees = [0.0, 0.01, 0.02]  # 3 entries before translate()

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "summary:\nA greeting.\n"

        # After CR, chatbot gets one more fee entry; after translation, another
        def mock_message(msgs, **kwargs):
            translator.chatbot.api_fees.append(0.05)
            return [MagicMock()]

        translator.chatbot.message.side_effect = mock_message
        translator.chatbot.get_content.return_value = "#1\n你好\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", compare_path=compare_path)

        # api_fee should include fees from index 3 onward (after the 3 pre-existing entries)
        # The mock_message appends 0.05 each call; at least one call for translation
        self.assertGreater(translator.api_fee, 0)


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestLeanTranslatorLive(unittest.TestCase):
    """Live API tests for LeanTranslator.

    These tests make real LLM API calls and require:
    - OPENLRC_TEST_LIVE_API=1
    - OPENLRC_TEST_LLM_API_KEY set to a valid API key
    """

    @classmethod
    def setUpClass(cls):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("OPENLRC_TEST_LLM_API_KEY is required for LLM integration tests.")

    def tearDown(self):
        compare_path = Path("translate_intermediate.json")
        compare_path.unlink(missing_ok=True)

    def test_single_chunk_no_cr(self):
        """Basic single-chunk translation without CR."""
        chatbot = create_chatbot(TEST_MODELS["gemini"])
        try:
            translator = LeanTranslator(chatbot=chatbot, enable_cr=False)
            translations = translator.translate("Hello, how are you?", "en", "es")
        finally:
            chatbot.close()

        self.assertEqual(len(translations), 1)
        self.assertTrue(translations[0].strip())
        self.assertGreater(get_similarity(translations[0], "Hola, ¿cómo estás?"), 0.5)

    def test_multiple_chunk_no_cr(self):
        """Multiple chunks without CR, verifying all lines are translated."""
        texts = ["Hello, how are you?", "I am fine, thank you.", "See you tomorrow.", "Good night."]
        chatbot = create_chatbot(TEST_MODELS["gemini"])
        try:
            translator = LeanTranslator(chatbot=chatbot, enable_cr=False, chunk_size=2)
            translations = translator.translate(texts, "en", "es")
        finally:
            chatbot.close()

        self.assertEqual(len(translations), len(texts))
        self.assertTrue(all(t.strip() for t in translations))

    def test_with_cr(self):
        """Full pipeline: CR generates guideline, then translation uses it."""
        texts = ["The suspect fled to the uptown area.", "Detective John began the investigation."]
        chatbot = create_chatbot(TEST_MODELS["gemini"])
        try:
            translator = LeanTranslator(chatbot=chatbot, enable_cr=True)
            translations = translator.translate(texts, "en", "zh")
        finally:
            chatbot.close()

        self.assertEqual(len(translations), len(texts))
        self.assertTrue(all(t.strip() for t in translations))

    def test_cr_chatbot_separation(self):
        """Mixed-model: GPT does CR, Gemini does translation."""
        texts = ["The suspect fled to the uptown area.", "Detective John began the investigation."]
        cr_bot = create_chatbot(TEST_MODELS["gpt"])
        mt_bot = create_chatbot(TEST_MODELS["gemini"])
        try:
            translator = LeanTranslator(chatbot=mt_bot, cr_chatbot=cr_bot, enable_cr=True)
            translations = translator.translate(texts, "en", "zh")
        finally:
            cr_bot.close()
            mt_bot.close()

        self.assertEqual(len(translations), len(texts))
        self.assertTrue(all(t.strip() for t in translations))
        self.assertGreater(translator.api_fee, 0)

    def test_empty_texts(self):
        """Empty input returns empty list without API calls."""
        chatbot = create_chatbot(TEST_MODELS["gemini"])
        try:
            translator = LeanTranslator(chatbot=chatbot, enable_cr=False)
            translations = translator.translate([], "en", "es")
        finally:
            chatbot.close()

        self.assertEqual(translations, [])
