#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openlrc.context import TranslationContext
from openlrc.translate import LLMTranslator


class TestMakeChunks(unittest.TestCase):
    """Unit tests for LLMTranslator.make_chunks — pure logic, no mocks needed."""

    def test_basic(self):
        """10 texts, chunk_size=5 -> 2 chunks of 5."""
        texts = [f"text{i}" for i in range(10)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)

    def test_exact_fit(self):
        """30 texts, chunk_size=30 -> 1 chunk."""
        texts = [f"text{i}" for i in range(30)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 30)

    def test_merge_small_tail(self):
        """35 texts, chunk_size=30 -> tail (5) < 30/2, merged into previous -> 1 chunk of 35."""
        texts = [f"text{i}" for i in range(35)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 35)

    def test_no_merge_large_tail(self):
        """46 texts, chunk_size=30 -> tail (16) >= 30/2, not merged -> 2 chunks."""
        texts = [f"text{i}" for i in range(46)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=30)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 30)
        self.assertEqual(len(chunks[1]), 16)

    def test_empty(self):
        """Empty input -> empty result."""
        chunks = LLMTranslator.make_chunks([], chunk_size=30)
        self.assertEqual(chunks, [])

    def test_single_item(self):
        """1 text -> 1 chunk with 1 item."""
        chunks = LLMTranslator.make_chunks(["hello"], chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(chunks[0][0], (1, "hello"))

    def test_line_numbers(self):
        """Line numbers start at 1 and increment continuously across chunks."""
        texts = [f"text{i}" for i in range(8)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=3)
        # 8 items, chunk_size=3 -> [3, 3, 2], tail 2 >= 1.5 so 3 chunks
        all_line_numbers = [num for chunk in chunks for num, _ in chunk]
        self.assertEqual(all_line_numbers, list(range(1, 9)))


class TestMakeChunksByTokens(unittest.TestCase):
    """Unit tests for LLMTranslator.make_chunks_by_tokens."""

    def _make_translator(self, chunk_size=30, max_chunk_tokens=1000, scene_threshold=30.0):
        bot = _make_mock_chatbot()
        t = LLMTranslator(chatbot=bot, chunk_size=chunk_size)
        t.MAX_CHUNK_TOKENS = max_chunk_tokens
        t.SCENE_THRESHOLD = scene_threshold
        return t

    def test_no_timestamps_splits_by_tokens(self):
        """Without timestamps, chunks are split by token budget only."""
        # Use a very small token budget to force splitting.
        translator = self._make_translator(max_chunk_tokens=20)
        texts = [f"word{i} " * 5 for i in range(6)]  # Each line ~5 tokens
        chunks = translator.make_chunks_by_tokens(texts)
        # With ~5 tokens per line and budget=20, expect ~4 lines per chunk.
        self.assertTrue(all(len(c) <= 5 for c in chunks))
        # All lines present.
        all_nums = [num for c in chunks for num, _ in c]
        self.assertEqual(all_nums, list(range(1, 7)))

    def test_scene_boundary_forces_split(self):
        """A time gap > SCENE_THRESHOLD forces a chunk break."""
        translator = self._make_translator(max_chunk_tokens=10000, scene_threshold=30.0)
        texts = ["line1", "line2", "line3", "line4"]
        # 60s gap between line2 and line3 → scene boundary.
        translator.timestamps = [(0.0, 1.0), (1.0, 2.0), (62.0, 63.0), (63.0, 64.0)]
        chunks = translator.make_chunks_by_tokens(texts)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([n for n, _ in chunks[0]], [1, 2])
        self.assertEqual([n for n, _ in chunks[1]], [3, 4])

    def test_line_count_cap(self):
        """chunk_size acts as a line-count upper bound."""
        translator = self._make_translator(chunk_size=3, max_chunk_tokens=10000)
        texts = [f"short{i}" for i in range(9)]
        chunks = translator.make_chunks_by_tokens(texts)
        self.assertTrue(all(len(c) <= 3 for c in chunks))

    def test_empty_input(self):
        translator = self._make_translator()
        self.assertEqual(translator.make_chunks_by_tokens([]), [])

    def test_small_tail_does_not_break_line_cap(self):
        """A small trailing chunk stays separate when merging would exceed chunk_size."""
        translator = self._make_translator(chunk_size=5, max_chunk_tokens=10000)
        texts = [f"text{i}" for i in range(7)]
        chunks = translator.make_chunks_by_tokens(texts)
        self.assertEqual([len(chunk) for chunk in chunks], [5, 2])

    def test_best_split_at_largest_gap(self):
        """When token budget is exceeded, split at the largest time gap."""
        translator = self._make_translator(max_chunk_tokens=30, chunk_size=30)
        texts = ["word " * 5] * 6  # Each line ~5 tokens, total ~30 → triggers split
        # Largest gap between line 3 and line 4 (10s gap vs 1s gaps).
        timestamps = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (13.0, 14.0), (14.0, 15.0), (15.0, 16.0)]
        translator.timestamps = timestamps
        chunks = translator.make_chunks_by_tokens(texts)
        # Should split at the 10s gap (between line 3 and 4).
        self.assertTrue(len(chunks) >= 2)
        first_chunk_lines = [n for n, _ in chunks[0]]
        self.assertIn(3, first_chunk_lines)
        self.assertNotIn(4, first_chunk_lines)


def _make_mock_chatbot(name: str = "gpt-4.1-nano") -> MagicMock:
    """Return a lightweight mock ChatBot."""
    bot = MagicMock()
    bot.model_name = name
    bot.close = MagicMock()
    return bot


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
class TestLLMTranslatorTranslate(unittest.TestCase):
    """Mock tests for LLMTranslator.translate() — no real API calls."""

    def _make_translator(self, chunk_size=30, retry_chatbot=None):
        return LLMTranslator(chatbot=_make_mock_chatbot(), chunk_size=chunk_size, retry_chatbot=retry_chatbot)

    def _mock_translate_chunk(self, translations, summary="summary", scene="scene"):
        """Return a side_effect function that returns translations matching chunk length."""
        offset = 0

        def side_effect(chunk_id, chunk, context, use_glossary=True):
            nonlocal offset
            ctx = TranslationContext(
                summary=summary, scene=scene, guideline=context.guideline, previous_summaries=context.previous_summaries
            )
            result = translations[offset : offset + len(chunk)]
            offset += len(chunk)
            return result, ctx

        return side_effect

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_single_chunk(self, mock_agent_cls, mock_reviewer_cls):
        """Texts fitting in one chunk -> translate_chunk called once, correct result."""
        texts = ["hello", "world"]
        expected = ["你好", "世界"]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0.001
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(expected)

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "test guideline"

        translator = self._make_translator(chunk_size=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, expected)
        mock_agent.translate_chunk.assert_called_once()
        mock_reviewer.build_context.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_multiple_chunks(self, mock_agent_cls, mock_reviewer_cls):
        """Texts spanning 2 chunks -> translate_chunk called twice.

        With chunk_size=3 and 6 texts, translate() produces 2 chunks.
        The mock side_effect advances an internal offset so that each
        chunk receives its own slice of the translations list:
          chunk 1 -> ['trans0', 'trans1', 'trans2']
          chunk 2 -> ['trans3', 'trans4', 'trans5']
        The final result should be the full list in order.
        """
        texts = [f"text{i}" for i in range(6)]
        translations = [f"trans{i}" for i in range(6)]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0.002
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(translations)

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "test guideline"

        translator = self._make_translator(chunk_size=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, translations)
        self.assertEqual(mock_agent.translate_chunk.call_count, 2)

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_context_passing_between_chunks(self, mock_agent_cls, mock_reviewer_cls):
        """Context (summary, scene) from chunk N is passed to chunk N+1."""
        texts = [f"text{i}" for i in range(6)]
        call_contexts = []

        def capture_context(chunk_id, chunk, context, use_glossary=True):
            call_contexts.append({"chunk_id": chunk_id, "previous_summaries": list(context.previous_summaries or [])})
            ctx = TranslationContext(
                summary=f"summary_{chunk_id}",
                scene=f"scene_{chunk_id}",
                guideline=context.guideline,
                previous_summaries=context.previous_summaries,
            )
            # Return one translation per source line, using the line number from chunk
            return [f"trans{line_num}" for line_num, _ in chunk], ctx

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.translate_chunk.side_effect = capture_context

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            translator.translate(texts, "en", "zh", compare_path=compare_path)

        # Chunk 1: no previous summaries
        self.assertEqual(call_contexts[0]["previous_summaries"], [])
        # Chunk 2: has summary from chunk 1
        self.assertEqual(call_contexts[1]["previous_summaries"], ["summary_1"])

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_length_mismatch_triggers_atomic(self, mock_agent_cls, mock_reviewer_cls):
        """When translate_chunk returns wrong length, atomic_translate is used as fallback."""
        texts = ["hello", "world"]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.info.glossary = None
        # Return 1 translation for 2 texts -> length mismatch
        mock_agent.translate_chunk.return_value = (
            ["only_one"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=30)
        with patch.object(translator, "atomic_translate", return_value=["你好", "世界"]) as mock_atomic:
            with tempfile.TemporaryDirectory() as tmpdir:
                compare_path = Path(tmpdir) / "compare.json"
                result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        mock_atomic.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_retry_agent_used_on_primary_failure(self, mock_agent_cls, mock_reviewer_cls):
        """When primary agent returns wrong length, retry agent is activated."""
        texts = ["hello", "world"]

        # Two ChunkedTranslatorAgent instances: primary (call 1) and retry (call 2)
        primary_agent = MagicMock()
        primary_agent.cost = 0
        primary_agent.info.glossary = None
        # Primary returns wrong length
        primary_agent.translate_chunk.return_value = (
            ["only_one"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        retry_agent = MagicMock()
        retry_agent.cost = 0
        retry_agent.info.glossary = None
        # Retry returns correct length
        retry_agent.translate_chunk.return_value = (
            ["你好", "世界"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        mock_agent_cls.side_effect = [primary_agent, retry_agent]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=30, retry_chatbot=_make_mock_chatbot())
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        primary_agent.translate_chunk.assert_called_once()
        retry_agent.translate_chunk.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_retry_streak_resets_when_retry_agent_also_fails(self, mock_agent_cls, mock_reviewer_cls):
        """When retry agent also returns wrong length, use_retry_cnt resets to 0."""
        texts = ["Hello", "World"]

        primary_agent = MagicMock()
        primary_agent.cost = 0
        primary_agent.info.glossary = None
        primary_agent.translate_chunk.return_value = (
            ["only_one"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        retry_agent = MagicMock()
        retry_agent.cost = 0
        retry_agent.info.glossary = None
        # Retry also returns wrong length
        retry_agent.translate_chunk.return_value = (
            ["still_wrong"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        mock_agent_cls.side_effect = [primary_agent, retry_agent]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=30, retry_chatbot=_make_mock_chatbot())

        # Mock atomic_translate to provide fallback
        translator.atomic_translate = MagicMock(return_value=["你好", "世界"])

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        # Both agents were tried
        primary_agent.translate_chunk.assert_called_once()
        retry_agent.translate_chunk.assert_called_once()
        # Streak should be reset to 0 after retry agent also failed
        self.assertEqual(translator.use_retry_cnt, 0)

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_binary_split_retry_on_mismatch(self, mock_agent_cls, mock_reviewer_cls):
        """When full chunk fails, binary split translates each half successfully."""
        texts = [f"line{i}" for i in range(6)]

        call_count = 0

        def translate_chunk_side_effect(chunk_id, chunk, context, use_glossary=True):
            nonlocal call_count
            call_count += 1
            ctx = TranslationContext(summary="s", scene="sc", guideline=context.guideline)
            # First call (full 6-line chunk) returns wrong length
            if len(chunk) == 6:
                return ["wrong"], ctx
            # Halves (3 lines each) succeed
            return [f"trans_{c[0]}" for c in chunk], ctx

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.info.glossary = None
        mock_agent.translate_chunk.side_effect = translate_chunk_side_effect

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        # 6 translations from two successful halves
        self.assertEqual(len(result), 6)
        self.assertTrue(all(r.startswith("trans_") for r in result))

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_binary_split_falls_back_to_atomic(self, mock_agent_cls, mock_reviewer_cls):
        """When halves are below MIN_SPLIT_SIZE and still fail, atomic is used."""
        texts = ["Hello", "World"]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.info.glossary = None
        # Always return wrong length
        mock_agent.translate_chunk.return_value = (
            ["wrong"],
            TranslationContext(summary="s", scene="sc", guideline="g"),
        )

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = "guideline"

        translator = self._make_translator(chunk_size=30)
        translator.atomic_translate = MagicMock(return_value=["你好", "世界"])

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        self.assertEqual(result, ["你好", "世界"])
        translator.atomic_translate.assert_called_once()

    @patch("openlrc.translate.ContextReviewerAgent")
    @patch("openlrc.translate.ChunkedTranslatorAgent")
    def test_resume_from_compare_file(self, mock_agent_cls, mock_reviewer_cls):
        """Translation resumes from saved compare file, skipping already-translated chunks."""
        texts = [f"text{i}" for i in range(6)]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        # Only chunk 2 should be translated (chunk 1 already done)
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(
            [f"trans{i}" for i in range(3, 6)], summary="summary_2"
        )

        mock_reviewer = mock_reviewer_cls.return_value

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / "compare.json"

            # Pre-populate compare file as if chunk 1 was already translated
            saved_state = {
                "compare": [
                    {
                        "chunk": 1,
                        "idx": i + 1,
                        "method": "chunked",
                        "model": "None",
                        "input": f"text{i}",
                        "output": f"trans{i}",
                    }
                    for i in range(3)
                ],
                "summaries": ["summary_1"],
                "scene": "scene_1",
                "guideline": "saved guideline",
            }
            with open(compare_path, "w") as f:
                json.dump(saved_state, f)

            translator = self._make_translator(chunk_size=3)
            result = translator.translate(texts, "en", "zh", compare_path=compare_path)

        # Should have 6 translations: 3 resumed + 3 newly translated
        self.assertEqual(len(result), 6)
        self.assertEqual(result[:3], ["trans0", "trans1", "trans2"])
        # build_context should NOT be called (guideline loaded from file)
        mock_reviewer.build_context.assert_not_called()
        # translate_chunk called only once (for chunk 2)
        mock_agent.translate_chunk.assert_called_once()


class TestCheckpoint(unittest.TestCase):
    """Unit tests for BaseLLMTranslator._save_checkpoint / _load_checkpoint."""

    def _make_translator(self):
        return LLMTranslator(chatbot=_make_mock_chatbot(), chunk_size=30)

    def test_save_checkpoint_produces_expected_json(self):
        """_save_checkpoint writes JSON with 'compare' key plus context keys."""
        translator = self._make_translator()
        compare_list = [
            {"chunk": 1, "idx": 1, "method": "chunked", "model": "gpt-4", "input": "hello", "output": "你好"},
            {"chunk": 1, "idx": 2, "method": "chunked", "model": "gpt-4", "input": "world", "output": "世界"},
        ]
        context = {"summaries": ["A greeting."], "scene": "office", "guideline": "Translate naturally."}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            translator._save_checkpoint(path, compare_list, context)

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

        self.assertEqual(data["compare"], compare_list)
        self.assertEqual(data["summaries"], ["A greeting."])
        self.assertEqual(data["scene"], "office")
        self.assertEqual(data["guideline"], "Translate naturally.")

    def test_load_checkpoint_restores_state(self):
        """_load_checkpoint correctly restores translations, compare_list, start_chunk, and context."""
        translator = self._make_translator()

        saved = {
            "compare": [
                {"chunk": 1, "idx": 1, "method": "chunked", "model": "gpt-4", "input": "hello", "output": "你好"},
                {"chunk": 1, "idx": 2, "method": "chunked", "model": "gpt-4", "input": "world", "output": "世界"},
                {"chunk": 2, "idx": 3, "method": "atomic", "model": "gpt-4", "input": "foo", "output": "富"},
            ],
            "summaries": ["sum1", "sum2"],
            "scene": "park",
            "guideline": "Be concise.",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(saved, f)

            translations, compare_list, start_chunk, ctx = translator._load_checkpoint(path)

        self.assertEqual(translations, ["你好", "世界", "富"])
        self.assertEqual(len(compare_list), 3)
        self.assertEqual(start_chunk, 2)
        self.assertEqual(ctx["summaries"], ["sum1", "sum2"])
        self.assertEqual(ctx["scene"], "park")
        self.assertEqual(ctx["guideline"], "Be concise.")

    def test_load_checkpoint_file_not_exists(self):
        """_load_checkpoint returns empty defaults when file does not exist."""
        translator = self._make_translator()
        translations, compare_list, start_chunk, ctx = translator._load_checkpoint(Path("/nonexistent/path.json"))

        self.assertEqual(translations, [])
        self.assertEqual(compare_list, [])
        self.assertEqual(start_chunk, 0)
        self.assertEqual(ctx, {})

    def test_load_checkpoint_empty_compare_list(self):
        """_load_checkpoint handles empty compare list without IndexError."""
        translator = self._make_translator()

        saved = {"compare": [], "summaries": [], "guideline": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(saved, f)

            translations, compare_list, start_chunk, ctx = translator._load_checkpoint(path)

        self.assertEqual(translations, [])
        self.assertEqual(compare_list, [])
        self.assertEqual(start_chunk, 0)
        self.assertEqual(ctx["guideline"], "test")

    def test_save_load_roundtrip(self):
        """Data survives a save -> load roundtrip with identical semantics."""
        translator = self._make_translator()
        compare_list = [{"chunk": 3, "idx": 7, "method": "chunked", "model": "gpt-4", "input": "hi", "output": "嗨"}]
        context = {"summaries": ["s1", "s2", "s3"], "scene": "beach", "guideline": "Keep it casual."}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            translator._save_checkpoint(path, compare_list, context)
            translations, loaded_list, start_chunk, ctx = translator._load_checkpoint(path)

        self.assertEqual(translations, ["嗨"])
        self.assertEqual(loaded_list, compare_list)
        self.assertEqual(start_chunk, 3)
        self.assertEqual(ctx["summaries"], ["s1", "s2", "s3"])
        self.assertEqual(ctx["scene"], "beach")
        self.assertEqual(ctx["guideline"], "Keep it casual.")
