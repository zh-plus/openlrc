#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from openlrc.context import TranslationContext
from openlrc.translate import LLMTranslator


class TestMakeChunks(unittest.TestCase):
    """Unit tests for LLMTranslator.make_chunks — pure logic, no mocks needed."""

    def test_basic(self):
        """10 texts, chunk_size=5 -> 2 chunks of 5."""
        texts = [f'text{i}' for i in range(10)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)

    def test_exact_fit(self):
        """30 texts, chunk_size=30 -> 1 chunk."""
        texts = [f'text{i}' for i in range(30)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 30)

    def test_merge_small_tail(self):
        """35 texts, chunk_size=30 -> tail (5) < 30/2, merged into previous -> 1 chunk of 35."""
        texts = [f'text{i}' for i in range(35)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 35)

    def test_no_merge_large_tail(self):
        """46 texts, chunk_size=30 -> tail (16) >= 30/2, not merged -> 2 chunks."""
        texts = [f'text{i}' for i in range(46)]
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
        chunks = LLMTranslator.make_chunks(['hello'], chunk_size=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(chunks[0][0], (1, 'hello'))

    def test_line_numbers(self):
        """Line numbers start at 1 and increment continuously across chunks."""
        texts = [f'text{i}' for i in range(8)]
        chunks = LLMTranslator.make_chunks(texts, chunk_size=3)
        # 8 items, chunk_size=3 -> [3, 3, 2], tail 2 >= 1.5 so 3 chunks
        all_line_numbers = [num for chunk in chunks for num, _ in chunk]
        self.assertEqual(all_line_numbers, list(range(1, 9)))


@patch.dict(os.environ, {'OPENAI_API_KEY': 'test-dummy'})
class TestLLMTranslatorTranslate(unittest.TestCase):
    """Mock tests for LLMTranslator.translate() — no real API calls."""

    def _make_translator(self, chunk_size=30, retry_model=None):
        return LLMTranslator(
            chatbot_model='gpt-4.1-nano', fee_limit=0.8,
            chunk_size=chunk_size, retry_model=retry_model,
        )

    def _mock_translate_chunk(self, translations, summary='summary', scene='scene'):
        """Return a side_effect function that returns translations matching chunk length."""
        offset = 0

        def side_effect(chunk_id, chunk, context, use_glossary=True):
            nonlocal offset
            ctx = TranslationContext(
                summary=summary, scene=scene, guideline=context.guideline,
                previous_summaries=context.previous_summaries,
            )
            result = translations[offset:offset + len(chunk)]
            offset += len(chunk)
            return result, ctx
        return side_effect

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_single_chunk(self, mock_agent_cls, mock_reviewer_cls):
        """Texts fitting in one chunk -> translate_chunk called once, correct result."""
        texts = ['hello', 'world']
        expected = ['你好', '世界']

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0.001
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(expected)

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = 'test guideline'

        translator = self._make_translator(chunk_size=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / 'compare.json'
            result = translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        self.assertEqual(result, expected)
        mock_agent.translate_chunk.assert_called_once()
        mock_reviewer.build_context.assert_called_once()

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_multiple_chunks(self, mock_agent_cls, mock_reviewer_cls):
        """Texts spanning 2 chunks -> translate_chunk called twice.

        With chunk_size=3 and 6 texts, translate() produces 2 chunks.
        The mock side_effect advances an internal offset so that each
        chunk receives its own slice of the translations list:
          chunk 1 -> ['trans0', 'trans1', 'trans2']
          chunk 2 -> ['trans3', 'trans4', 'trans5']
        The final result should be the full list in order.
        """
        texts = [f'text{i}' for i in range(6)]
        translations = [f'trans{i}' for i in range(6)]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0.002
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(translations)

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = 'test guideline'

        translator = self._make_translator(chunk_size=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / 'compare.json'
            result = translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        self.assertEqual(result, translations)
        self.assertEqual(mock_agent.translate_chunk.call_count, 2)

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_context_passing_between_chunks(self, mock_agent_cls, mock_reviewer_cls):
        """Context (summary, scene) from chunk N is passed to chunk N+1."""
        texts = [f'text{i}' for i in range(6)]
        call_contexts = []

        def capture_context(chunk_id, chunk, context, use_glossary=True):
            call_contexts.append({
                'chunk_id': chunk_id,
                'previous_summaries': list(context.previous_summaries or []),
            })
            ctx = TranslationContext(
                summary=f'summary_{chunk_id}', scene=f'scene_{chunk_id}',
                guideline=context.guideline,
                previous_summaries=context.previous_summaries,
            )
            # Return one translation per source line, using the line number from chunk
            return [f'trans{line_num}' for line_num, _ in chunk], ctx

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.translate_chunk.side_effect = capture_context

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = 'guideline'

        translator = self._make_translator(chunk_size=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / 'compare.json'
            translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        # Chunk 1: no previous summaries
        self.assertEqual(call_contexts[0]['previous_summaries'], [])
        # Chunk 2: has summary from chunk 1
        self.assertEqual(call_contexts[1]['previous_summaries'], ['summary_1'])

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_length_mismatch_triggers_atomic(self, mock_agent_cls, mock_reviewer_cls):
        """When translate_chunk returns wrong length, atomic_translate is used as fallback."""
        texts = ['hello', 'world']

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        mock_agent.info.glossary = None
        # Return 1 translation for 2 texts -> length mismatch
        mock_agent.translate_chunk.return_value = (
            ['only_one'],
            TranslationContext(summary='s', scene='sc', guideline='g'),
        )

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = 'guideline'

        translator = self._make_translator(chunk_size=30)
        with patch.object(translator, 'atomic_translate', return_value=['你好', '世界']) as mock_atomic:
            with tempfile.TemporaryDirectory() as tmpdir:
                compare_path = Path(tmpdir) / 'compare.json'
                result = translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        self.assertEqual(result, ['你好', '世界'])
        mock_atomic.assert_called_once()

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_retry_agent_used_on_primary_failure(self, mock_agent_cls, mock_reviewer_cls):
        """When primary agent returns wrong length, retry agent is activated."""
        texts = ['hello', 'world']

        # Two ChunkedTranslatorAgent instances: primary (call 1) and retry (call 2)
        primary_agent = MagicMock()
        primary_agent.cost = 0
        primary_agent.info.glossary = None
        # Primary returns wrong length
        primary_agent.translate_chunk.return_value = (
            ['only_one'],
            TranslationContext(summary='s', scene='sc', guideline='g'),
        )

        retry_agent = MagicMock()
        retry_agent.cost = 0
        retry_agent.info.glossary = None
        # Retry returns correct length
        retry_agent.translate_chunk.return_value = (
            ['你好', '世界'],
            TranslationContext(summary='s', scene='sc', guideline='g'),
        )

        mock_agent_cls.side_effect = [primary_agent, retry_agent]

        mock_reviewer = mock_reviewer_cls.return_value
        mock_reviewer.build_context.return_value = 'guideline'

        translator = self._make_translator(chunk_size=30, retry_model='gpt-4.1-nano')
        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / 'compare.json'
            result = translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        self.assertEqual(result, ['你好', '世界'])
        primary_agent.translate_chunk.assert_called_once()
        retry_agent.translate_chunk.assert_called_once()

    @patch('openlrc.translate.ContextReviewerAgent')
    @patch('openlrc.translate.ChunkedTranslatorAgent')
    def test_resume_from_compare_file(self, mock_agent_cls, mock_reviewer_cls):
        """Translation resumes from saved compare file, skipping already-translated chunks."""
        texts = [f'text{i}' for i in range(6)]

        mock_agent = mock_agent_cls.return_value
        mock_agent.cost = 0
        # Only chunk 2 should be translated (chunk 1 already done)
        mock_agent.translate_chunk.side_effect = self._mock_translate_chunk(
            [f'trans{i}' for i in range(3, 6)], summary='summary_2',
        )

        mock_reviewer = mock_reviewer_cls.return_value

        with tempfile.TemporaryDirectory() as tmpdir:
            compare_path = Path(tmpdir) / 'compare.json'

            # Pre-populate compare file as if chunk 1 was already translated
            saved_state = {
                'compare': [
                    {'chunk': 1, 'idx': i + 1, 'method': 'chunked', 'model': 'None',
                     'input': f'text{i}', 'output': f'trans{i}'}
                    for i in range(3)
                ],
                'summaries': ['summary_1'],
                'scene': 'scene_1',
                'guideline': 'saved guideline',
            }
            with open(compare_path, 'w') as f:
                json.dump(saved_state, f)

            translator = self._make_translator(chunk_size=3)
            result = translator.translate(texts, 'en', 'zh', compare_path=compare_path)

        # Should have 6 translations: 3 resumed + 3 newly translated
        self.assertEqual(len(result), 6)
        self.assertEqual(result[:3], ['trans0', 'trans1', 'trans2'])
        # build_context should NOT be called (guideline loaded from file)
        mock_reviewer.build_context.assert_not_called()
        # translate_chunk called only once (for chunk 2)
        mock_agent.translate_chunk.assert_called_once()
