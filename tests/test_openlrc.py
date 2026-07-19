#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from openlrc.chunking import TranslationChunkPlan
from openlrc.config import ContextLLMConfig, HyMT2Mode, LocalLLMConfig, SubtitleOptimizationMode
from openlrc.context import ContextTimeline, ProChunkContext, TranslateInfo, TranslationBrief, TranslationBriefInput
from openlrc.hymt2_pipeline import HyMT2RiskReviewAgent
from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_ALIAS,
    HY_MT2_7B_MODEL_ALIAS,
    HY_MT2_7B_MODEL_FILE,
    HY_MT2_PROMPT_PROFILE,
)
from openlrc.models import ModelProvider
from openlrc.openlrc import LRCer, TranscriptionConfig, TranslationConfig
from openlrc.subtitle import Subtitle
from openlrc.transcribe import TranscriptionInfo
from openlrc.utils import extend_filename
from openlrc.whisper_resources import DEFAULT_MODEL_NAME
from openlrc.whisper_types import Segment, Word

TEST_DATA_DIR = Path(__file__).parent / "data"

# Shared test config — avoids repeating these in every test method.
_TEST_TRANSCRIPTION = TranscriptionConfig(whisper_model=sys.executable, cli_path=sys.executable, vad_model="")


def _mock_create_chatbot(*args, **kwargs):
    """Return a lightweight mock ChatBot for tests that hit LRCer._translate()."""
    bot = MagicMock()
    bot.model_name = "gpt-4.1-nano"
    bot.close = MagicMock()
    return bot


@patch(
    "openlrc.transcribe.Transcriber.transcribe",
    MagicMock(
        return_value=(
            [
                Segment(
                    0,
                    0,
                    0,
                    3,
                    "hello world1",
                    [],
                    0.8,
                    0,
                    0,
                    words=[Word(0, 1.5, "hello", probability=0.8), Word(1.6, 3, " world1", probability=0.8)],
                    temperature=0,
                ),
                Segment(
                    0,
                    0,
                    3,
                    6,
                    "hello world2",
                    [],
                    0.8,
                    0,
                    0,
                    words=[Word(3, 4.5, "hello", probability=0.8), Word(4.6, 6, " world2", probability=0.8)],
                    temperature=0,
                ),
            ],
            TranscriptionInfo("en", 6.0, 6.0),
        )
    ),
)
@patch("openlrc.agents.create_chatbot", side_effect=_mock_create_chatbot)
class TestLRCer(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = TEST_DATA_DIR / "test_audio.wav"
        self.video_path = TEST_DATA_DIR / "test_video.mp4"
        self.nospeech_video_path = TEST_DATA_DIR / "test_nospeech_video.mp4"
        self._clear_artifacts()

    def _clear_artifacts(self) -> None:
        def clear_paths(input_path):
            transcribed = extend_filename(input_path, "_transcribed").with_suffix(".json")
            optimized = extend_filename(transcribed, "_optimized")
            translated = extend_filename(optimized, "_translated")
            relaxed_optimized = extend_filename(transcribed, "_optimized_relaxed")
            relaxed_translated = extend_filename(relaxed_optimized, "_translated")
            compare_path = extend_filename(input_path, "_compare").with_suffix(".json")

            json_path = input_path.with_suffix(".json")
            lrc_path = input_path.with_suffix(".lrc")
            srt_path = input_path.with_suffix(".srt")

            [
                p.unlink(missing_ok=True)
                for p in [
                    transcribed,
                    optimized,
                    translated,
                    relaxed_optimized,
                    relaxed_translated,
                    compare_path,
                    json_path,
                    lrc_path,
                    srt_path,
                ]
            ]

        clear_paths(self.audio_path)
        clear_paths(self.video_path)

        self.video_path.with_suffix(".wav").unlink(missing_ok=True)

        shutil.rmtree(TEST_DATA_DIR / "preprocessed", ignore_errors=True)

    def tearDown(self) -> None:
        self._clear_artifacts()

    # ------------------------------------------------------------------
    # Pipeline tests (using new config API)
    # ------------------------------------------------------------------

    def test_subtitle_optimization_defaults_and_factories(self, _mock_chatbot):
        self.assertIs(LRCer().subtitle_optimization, SubtitleOptimizationMode.AGGRESSIVE)
        self.assertIs(
            LRCer.local(subtitle_optimization="relaxed").subtitle_optimization, SubtitleOptimizationMode.RELAXED
        )
        self.assertIs(
            LRCer.local_hy_mt2(subtitle_optimization=SubtitleOptimizationMode.RELAXED).subtitle_optimization,
            SubtitleOptimizationMode.RELAXED,
        )

    def test_relaxed_cache_path_is_isolated_and_existing_final_is_rebuilt(self, _mock_chatbot):
        lrcer = LRCer(subtitle_optimization="relaxed")
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            optimized_path = folder / "sample_transcribed_optimized_relaxed.json"
            optimized = Subtitle(
                language="en", filename=optimized_path, segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}]
            )
            (folder / "sample.json").write_text("{}", encoding="utf-8")
            translated = MagicMock()
            with patch.object(lrcer, "_translate", return_value=translated) as translate:
                result = lrcer._build_final_subtitle("sample", "zh-cn", optimized, skip_trans=False)

        self.assertIs(result, translated)
        self.assertEqual(translate.call_args.args[3].name, "sample_transcribed_optimized_relaxed_translated.json")

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_single_audio_transcription_translation(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.audio_path)
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_multiple_audio_transcription_translation(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run([self.audio_path, self.video_path])
        self.assertTrue(result)
        self.assertEqual(len(result), 2)

    def test_audio_file_not_found(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        with self.assertRaises(FileNotFoundError):
            lrcer.run(TEST_DATA_DIR / "invalid.mp3")

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_video_file_transcription_translation(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.video_path)
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_nospeech_video_file_transcription_translation(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.nospeech_video_path)
        self.assertTrue(result)

    @patch("openlrc.translate.LLMTranslator.translate", MagicMock(side_effect=Exception("test exception")))
    def test_translation_error(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        with self.assertRaises(Exception):
            lrcer.run(self.audio_path)

    @patch("openlrc.translate.LLMTranslator.translate", MagicMock(side_effect=Exception("test exception")))
    def test_skip_translation(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.video_path, skip_trans=True)
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_skip_preprocess(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)

        # Stage 1: Run preprocessing only
        lrcer.pre_process([self.audio_path])

        # Verify preprocessed file exists
        from openlrc.utils import get_preprocessed_path

        preprocessed_path = get_preprocessed_path(self.audio_path)
        self.assertTrue(preprocessed_path.exists())

        # Stage 2: Run transcription with skip_preprocess=True
        result = lrcer.run(self.audio_path, skip_preprocess=True)
        self.assertTrue(result)

    def test_skip_preprocess_file_not_found(self, _mock_chatbot):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)

        # Ensure no preprocessed file exists
        from openlrc.utils import get_preprocessed_path

        preprocessed_path = get_preprocessed_path(self.audio_path)
        preprocessed_path.unlink(missing_ok=True)

        # Should raise FileNotFoundError when skip_preprocess=True but file doesn't exist
        with self.assertRaises(FileNotFoundError):
            lrcer.run(self.audio_path, skip_preprocess=True)

    @patch("openlrc.translate.LLMTranslator.translate")
    @patch("openlrc.openlrc.LRCer.post_process", wraps=LRCer.post_process)
    def test_skip_trans_skips_translate_but_calls_post_process(self, mock_post_process, mock_translate, _mock_chatbot):
        """skip_trans=True should skip translation but still post-process transcription."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        lrcer.run(self.audio_path, skip_trans=True)
        mock_translate.assert_not_called()
        mock_post_process.assert_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_normal_run_calls_translate(self, mock_translate, _mock_chatbot):
        """skip_trans=False (default) should invoke LLMTranslator.translate."""
        mock_translate.return_value = ["test translation1", "test translation2"]
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        lrcer.run(self.audio_path)
        mock_translate.assert_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_transcribe_returns_json_paths(self, mock_translate, _mock_chatbot):
        """transcribe() should return transcribed JSON paths without triggering translation."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.transcribe(self.audio_path)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].exists())
        self.assertTrue(result[0].name.endswith("_transcribed.json"))
        mock_translate.assert_not_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_translate_processes_transcribed_json(self, mock_translate, _mock_chatbot):
        """translate() should process transcribed JSON files and produce subtitle output."""
        mock_translate.return_value = ["test translation1", "test translation2"]
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)

        # Stage 1: transcribe to get JSON paths
        transcribed = lrcer.transcribe(self.audio_path)
        self.assertEqual(len(transcribed), 1)
        mock_translate.assert_not_called()

        # Stage 2: translate the transcribed JSON
        result = lrcer.translate(transcribed, target_lang="zh-cn")
        self.assertTrue(result)
        self.assertEqual(len(result), 1)
        mock_translate.assert_called_once()

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_translate_independent_video_transcription_outputs_srt(self, _mock_chatbot):
        """translate() should keep video-origin output as .srt even on a fresh LRCer."""
        lrcer_transcribe = LRCer(transcription=_TEST_TRANSCRIPTION)
        transcribed = lrcer_transcribe.transcribe(self.video_path)
        self.assertEqual(len(transcribed), 1)

        lrcer_translate = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer_translate.translate(transcribed, target_lang="zh-cn")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].suffix, ".srt")

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_run_skip_trans_deduplicates_duplicate_inputs(self, mock_translate, _mock_chatbot):
        """run(skip_trans=True) should transcribe duplicate input paths only once."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run([self.audio_path, self.audio_path], skip_trans=True)
        self.assertTrue(result)
        self.assertEqual(len(result), 1)
        mock_translate.assert_not_called()

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_run_full_pipeline(self, _mock_chatbot):
        """run() with default args should produce subtitle output (full pipeline)."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.audio_path)
        self.assertTrue(result)
        self.assertEqual(len(result), 1)

    # ------------------------------------------------------------------
    # Config and constructor tests
    # ------------------------------------------------------------------

    def test_translate_only_does_not_instantiate_transcriber(self, _mock_chatbot):
        """LRCer with only TranslationConfig should not create a Transcriber."""
        lrcer = LRCer(translation=TranslationConfig())
        # Access the private attribute directly — should be None (lazy init).
        self.assertIsNone(lrcer._transcriber)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_transcriber_created_on_first_transcribe_call(self, _mock_chatbot):
        """Transcriber should be lazily created when transcribe() is first called."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        self.assertIsNone(lrcer._transcriber)
        lrcer.transcribe(self.audio_path)
        self.assertIsNotNone(lrcer._transcriber)

    def test_default_construction_without_arguments(self, _mock_chatbot):
        """LRCer() with no arguments should use default configs."""
        lrcer = LRCer()
        self.assertEqual(lrcer._transcription_config.whisper_model, DEFAULT_MODEL_NAME)
        self.assertIsNone(lrcer._translation_config.chatbot)
        self.assertIsNone(lrcer._transcriber)


class TestLRCerLocalLLM(unittest.TestCase):
    def test_translate_clears_completed_checkpoint_but_keeps_incomplete_review(self):
        lrcer = LRCer.local()
        with tempfile.TemporaryDirectory() as tmpdir:
            transcribed = Path(tmpdir) / "sample_preprocessed_transcribed.json"
            transcribed.write_text("{}", encoding="utf-8")
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text("{}", encoding="utf-8")

            with patch.object(lrcer, "_process_transcribed_file"):
                lrcer.translate(transcribed)
            self.assertFalse(checkpoint.exists())

            checkpoint.write_text("{}", encoding="utf-8")

            def mark_incomplete(*args, **kwargs):
                lrcer.review_statuses["sample"] = {"incomplete": True, "failed_chunks": [1]}

            with patch.object(lrcer, "_process_transcribed_file", side_effect=mark_incomplete):
                lrcer.translate(transcribed)
            self.assertTrue(checkpoint.exists())

    def test_clear_temp_files_removes_only_current_input_artifacts(self):
        lrcer = LRCer.local()
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "preprocessed"
            folder.mkdir()
            audio = folder / "sample_preprocessed.wav"
            derivative = folder / "sample_preprocessed_transcribed.json"
            checkpoint = folder / "sample_compare.json"
            unrelated = folder / "other_preprocessed.wav"
            for path in (audio, derivative, checkpoint, unrelated):
                path.write_text("data", encoding="utf-8")

            lrcer.clear_temp_files([audio])

            self.assertFalse(audio.exists())
            self.assertFalse(derivative.exists())
            self.assertFalse(checkpoint.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(folder.exists())

    def test_local_constructor_uses_recommended_local_config(self):
        lrcer = LRCer.local(idle_timeout=12, port=9090)

        self.assertIsNotNone(lrcer._translation_config.local_llm)
        self.assertEqual(lrcer._translation_config.local_llm.idle_timeout, 12)
        self.assertEqual(lrcer._translation_config.local_llm.port, 9090)
        self.assertEqual(lrcer._translation_config._translator_engine, "classic")
        self.assertTrue(lrcer._translation_config.enable_cr)
        self.assertEqual(lrcer._translation_config.consumer_thread, 1)

    def test_local_hy_mt2_constructor_uses_profile_config(self):
        lrcer = LRCer.local_hy_mt2(idle_timeout=12, port=9090)

        self.assertIsNotNone(lrcer._translation_config.local_llm)
        self.assertEqual(lrcer._translation_config.local_llm.model_path, HY_MT2_7B_MODEL_FILE)
        self.assertEqual(lrcer._translation_config.local_llm.alias, HY_MT2_7B_MODEL_ALIAS)
        self.assertEqual(lrcer._translation_config.prompt_profile, HY_MT2_PROMPT_PROFILE)
        self.assertEqual(lrcer._translation_config.chatbot.temperature, 0.7)
        self.assertEqual(lrcer._translation_config.chatbot.top_p, 0.6)

    def test_context_plus_stage_order(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        events: list[str] = []
        translator = MagicMock()
        translator.translate.return_value = ["草稿"]
        translator.metrics = {"retries": 0, "splits": 0, "atomic_ids": []}

        @contextmanager
        def local_session():
            events.append("hy-start")
            try:
                yield
            finally:
                events.append("hy-session-end")

        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]

        def prepare(*args, **kwargs):
            events.append("context-cr")
            return TranslationBrief(summary="summary"), {"schema_version": 2}

        def close_primary():
            events.append("hy-close")

        def review(*args, **kwargs):
            events.append("context-review")
            return ["修订"]

        with (
            patch.object(lrcer, "_prepare_hymt2_brief", side_effect=prepare),
            patch.object(lrcer, "_local_llm_session", side_effect=local_session),
            patch.object(lrcer, "_create_translator", return_value=translator),
            patch.object(lrcer, "_close_primary_local_stage", side_effect=close_primary),
            patch.object(lrcer, "_review_hymt2_translations", side_effect=review),
            patch("openlrc.openlrc.Subtitle.from_json", return_value=MagicMock()),
            patch.object(lrcer, "post_process", return_value=MagicMock()),
            patch("openlrc.openlrc.deepcopy", side_effect=lambda value: value),
        ):
            with self.subTest("staged order"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output = Path(tmpdir) / "context-plus-output.json"
                    lrcer._translate("sample", "zh-cn", transcribed, output)

        self.assertEqual(events, ["context-cr", "hy-start", "hy-session-end", "hy-close", "context-review"])

    def test_pro_stage_order_and_timeline_forwarding(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm)
        events: list[str] = []
        translator = MagicMock()
        translator.translate.return_value = ["草稿"]
        translator.metrics = {"retries": 0, "splits": 0, "atomic_ids": []}
        plan = TranslationChunkPlan(chunk_id=1, segment_ids=[1], texts=["source"])
        timeline = ContextTimeline(
            chunk_signature="signature",
            chunks=[ProChunkContext(chunk_id=1, segment_ids=[1], story_so_far="story", current_scene="scene")],
        )

        @contextmanager
        def local_session():
            events.append("hy-start")
            try:
                yield
            finally:
                events.append("hy-session-end")

        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]

        def prepare(*args, **kwargs):
            events.append("context-plan")
            return TranslationBrief(summary="summary"), timeline, [plan], {"schema_version": 3, "hymt2_mode": "pro"}

        def close_primary():
            events.append("hy-close")

        def review(*args, **kwargs):
            events.append("context-review")
            self.assertIs(kwargs["context_timeline"], timeline)
            self.assertEqual(kwargs["chunk_plans"], [plan])
            return ["修订"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample_translated.json"
            with (
                patch.object(lrcer, "_prepare_hymt2_pro_context", side_effect=prepare),
                patch.object(lrcer, "_local_llm_session", side_effect=local_session),
                patch.object(lrcer, "_create_translator", return_value=translator),
                patch.object(lrcer, "_close_primary_local_stage", side_effect=close_primary),
                patch.object(lrcer, "_review_hymt2_translations", side_effect=review),
                patch("openlrc.openlrc.Subtitle.from_json", return_value=MagicMock()),
                patch.object(lrcer, "post_process", return_value=MagicMock()),
                patch("openlrc.openlrc.deepcopy", side_effect=lambda value: value),
            ):
                lrcer._translate("sample", "zh-cn", transcribed, output)

        self.assertEqual(events, ["context-plan", "hy-start", "hy-session-end", "hy-close", "context-review"])
        translate_kwargs = translator.translate.call_args.kwargs
        self.assertIs(translate_kwargs["context_timeline"], timeline)
        self.assertEqual(translate_kwargs["chunk_plans"], [plan])

    def test_pro_context_plan_replaces_stale_checkpoint_and_closes_context_stage(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm)
        brief_payload = {"summary": "A meeting.", "characters": [], "glossary": [], "tone_style": "neutral"}
        bot = MagicMock()
        bot.api_fees = []
        bot.message.side_effect = [[MagicMock()], [MagicMock()]]
        bot.get_content.side_effect = [
            json.dumps(brief_payload),
            json.dumps(
                {
                    "chunk_id": 1,
                    "segment_ids": [1],
                    "story_so_far": "The meeting begins.",
                    "current_scene": "Two people are talking.",
                }
            ),
        ]
        server = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "source_fingerprint": "stale",
                        "chunk_signature": "stale",
                        "hymt2_mode": "pro",
                        "compare": [{"chunk": 1, "idx": 1, "output": "stale translation"}],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, server)):
                brief, timeline, plans, metadata = lrcer._prepare_hymt2_pro_context(
                    ["Hello."],
                    [(0.0, 1.0)],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=checkpoint_path,
                )
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(brief.summary, "A meeting.")
        self.assertEqual(timeline.chunks[0].segment_ids, [1])
        self.assertEqual(plans[0].segment_ids, [1])
        self.assertEqual(metadata["pipeline_stage"], "translation")
        self.assertEqual(saved["compare"], [])
        self.assertEqual(saved["schema_version"], 4)
        self.assertEqual(saved["checkpoint_kind"], "hymt2_pipeline")
        self.assertEqual(saved["pipeline_stage"], "translation")
        self.assertNotEqual(saved["source_fingerprint"], "stale")
        bot.close.assert_called_once()
        server.close.assert_called_once()

    def test_context_brief_prompt_version_invalidates_derived_checkpoint(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {
                "summary": "A new source-language meeting summary for the translation task.",
                "characters": [],
                "glossary": [],
                "tone_style": "formal",
            }
        )
        server = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "source_fingerprint": "fingerprint",
                        "brief_prompt_version": 1,
                        "translation_brief": {"summary": "stale target-language summary"},
                        "compare": [{"chunk": 1, "output": "stale translation"}],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("openlrc.hymt2_pipeline.source_fingerprint", return_value="fingerprint"),
                patch.object(lrcer, "_create_context_chatbot", return_value=(bot, server)),
            ):
                brief, metadata = lrcer._prepare_hymt2_brief(
                    ["The meeting begins."],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=checkpoint_path,
                )
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(brief.summary, "A new source-language meeting summary for the translation task.")
        self.assertEqual(metadata["brief_prompt_version"], 3)
        self.assertEqual(saved["brief_prompt_version"], 3)
        self.assertEqual(saved["compare"], [])
        bot.close.assert_called_once()
        server.close.assert_called_once()

    def test_complete_manual_brief_skips_context_model_and_checkpoints_provenance(self):
        manual = TranslationBriefInput(
            summary="A complete user-authored summary of the detective story.",
            characters=[],
            tone_style="restrained noir",
        )
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.NORMAL, translation_brief=manual, glossary={"case": "案件"})

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            with patch.object(lrcer, "_create_context_chatbot") as create_context:
                brief, metadata = lrcer._prepare_hymt2_brief(
                    ["The detective opened the case."],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample", glossary=lrcer.glossary),
                    compare_path=checkpoint_path,
                )
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        create_context.assert_not_called()
        self.assertEqual(brief.summary, manual.summary)
        self.assertEqual(brief.glossary[0].target, "案件")
        self.assertEqual(metadata["brief_origin"], "manual")
        self.assertEqual(metadata["brief_prompt_version"], 0)
        self.assertEqual(metadata["brief_input_fields"], ["summary", "characters", "tone_style"])
        self.assertEqual(saved["brief_input_fingerprint"], manual.fingerprint())

    def test_changed_manual_brief_invalidates_saved_effective_brief(self):
        first = TranslationBriefInput(summary="The first complete story summary.", characters=[], tone_style="")
        second = TranslationBriefInput(summary="The revised complete story summary.", characters=[], tone_style="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            first_lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.NORMAL, translation_brief=first)
            first_lrcer._prepare_hymt2_brief(
                ["The story begins."],
                src_lang="en",
                target_lang="zh-cn",
                info=TranslateInfo(title="sample"),
                compare_path=checkpoint_path,
            )
            first_saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            second_lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.NORMAL, translation_brief=second)
            brief, _ = second_lrcer._prepare_hymt2_brief(
                ["The story begins."],
                src_lang="en",
                target_lang="zh-cn",
                info=TranslateInfo(title="sample"),
                compare_path=checkpoint_path,
            )
            second_saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(brief.summary, second.summary)
        self.assertNotEqual(first_saved["source_fingerprint"], second_saved["source_fingerprint"])
        self.assertEqual(second_saved["brief_input_fingerprint"], second.fingerprint())

    def test_partial_brief_resume_reuses_completed_effective_brief(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        partial = TranslationBriefInput(summary="The user's fixed meeting summary.")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.NORMAL, context_llm=context_llm, translation_brief=partial)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {
                "summary": "Generated summary that must be replaced.",
                "characters": [],
                "glossary": [],
                "tone_style": "formal",
            }
        )
        server = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, server)):
                first, _ = lrcer._prepare_hymt2_brief(
                    ["The meeting begins."],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=checkpoint_path,
                )
            with patch.object(lrcer, "_create_context_chatbot") as create_context:
                resumed, metadata = lrcer._prepare_hymt2_brief(
                    ["The meeting begins."],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=checkpoint_path,
                )

        create_context.assert_not_called()
        self.assertEqual(first.summary, partial.summary)
        self.assertEqual(resumed, first)
        self.assertEqual(metadata["brief_origin"], "partial")

    def test_complete_manual_pro_brief_skips_brief_call_but_builds_timeline(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        manual = TranslationBriefInput(
            summary="A complete user-authored summary of the scene.", characters=[], tone_style="cinematic"
        )
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm, translation_brief=manual)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {"chunk_id": 1, "segment_ids": [1], "story_so_far": "", "current_scene": "A greeting."}
        )
        server = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, server)):
                brief, timeline, _, metadata = lrcer._prepare_hymt2_pro_context(
                    ["Hello."],
                    [(0.0, 1.0)],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=Path(tmpdir) / "sample_compare.json",
                )

        self.assertEqual(bot.message.call_count, 1)
        self.assertEqual(brief.summary, manual.summary)
        self.assertEqual(timeline.chunks[0].segment_ids, [1])
        self.assertEqual(metadata["brief_origin"], "manual")
        self.assertEqual(metadata["brief_prompt_version"], 0)
        bot.close.assert_called_once()
        server.close.assert_called_once()

    def test_pro_context_plan_failure_closes_context_stage_and_keeps_checkpoint(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.side_effect = RuntimeError("context failure")
        server = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sample_compare.json"
            with (
                patch.object(lrcer, "_create_context_chatbot", return_value=(bot, server)),
                self.assertRaisesRegex(RuntimeError, "context failure"),
            ):
                lrcer._prepare_hymt2_pro_context(
                    ["Hello."],
                    [(0.0, 1.0)],
                    src_lang="en",
                    target_lang="zh-cn",
                    info=TranslateInfo(title="sample"),
                    compare_path=checkpoint_path,
                )
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["pipeline_stage"], "context_plan")
        self.assertEqual(saved["compare"], [])
        bot.close.assert_called_once()
        server.close.assert_called_once()

    def test_pro_resumes_review_without_loading_hy_mt2(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm)
        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]
        translated_sub = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            translated_path = Path(tmpdir) / "sample_translated.json"
            translated_path.write_text("{}", encoding="utf-8")
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "hymt2_mode": "pro",
                        "source_fingerprint": "fingerprint",
                        "chunk_signature": "signature",
                        "chunk_planner_version": 2,
                        "timeline_schema_version": 1,
                        "timeline_prompt_version": 3,
                        "brief_prompt_version": 3,
                        "pipeline_stage": "review",
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "translation_brief": {"summary": "summary"},
                        "context_timeline": {
                            "schema_version": 1,
                            "chunk_signature": "signature",
                            "chunks": [
                                {"chunk_id": 1, "segment_ids": [1], "story_so_far": "story", "current_scene": "scene"}
                            ],
                        },
                        "raw_hymt2_translations": ["草稿"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch("openlrc.hymt2_pipeline.source_fingerprint", return_value="fingerprint"),
                patch("openlrc.chunking.chunk_plan_signature", return_value="signature"),
                patch.object(lrcer, "_review_hymt2_translations", return_value=["修订"]) as review,
                patch.object(lrcer, "_create_translator") as create_translator,
                patch("openlrc.openlrc.deepcopy", return_value=translated_sub),
                patch("openlrc.openlrc.Subtitle.from_json", return_value=translated_sub),
                patch.object(lrcer, "post_process", return_value=translated_sub),
            ):
                lrcer._translate("sample", "zh-cn", transcribed, translated_path)

        create_translator.assert_not_called()
        review.assert_called_once()
        self.assertIsInstance(review.call_args.kwargs["context_timeline"], ContextTimeline)
        translated_sub.set_texts.assert_called_once_with(["修订"], lang="zh-cn")

    def test_pro_completed_translation_checkpoint_runs_missing_deterministic_stage(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.PRO, context_llm=context_llm)
        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]
        translated_sub = MagicMock()
        plan = TranslationChunkPlan(chunk_id=1, segment_ids=[1], texts=["source"])
        timeline = ContextTimeline(
            chunk_signature="signature",
            chunks=[ProChunkContext(chunk_id=1, segment_ids=[1], story_so_far="story", current_scene="scene")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translated_path = Path(tmpdir) / "sample_translated.json"
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "hymt2_mode": "pro",
                        "pipeline_stage": "translation",
                        "compare": [{"chunk": 1, "idx": 1, "input": "source", "output": "草稿"}],
                        "translation_brief": {"summary": "summary"},
                        "context_timeline": timeline.model_dump(),
                        "raw_hymt2_translations": ["草稿"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    lrcer,
                    "_prepare_hymt2_pro_context",
                    return_value=(
                        TranslationBrief(summary="summary"),
                        timeline,
                        [plan],
                        {"schema_version": 3, "hymt2_mode": "pro"},
                    ),
                ),
                patch.object(lrcer, "_local_llm_session") as local_session,
                patch.object(lrcer, "_create_translator") as create_translator,
                patch.object(lrcer, "_close_primary_local_stage") as close_primary,
                patch.object(lrcer, "_run_deterministic_hymt2_edit", return_value=["草稿"]) as deterministic,
                patch.object(lrcer, "_review_hymt2_translations", return_value=["修订"]) as review,
                patch("openlrc.openlrc.deepcopy", return_value=translated_sub),
                patch("openlrc.openlrc.Subtitle.from_json", return_value=translated_sub),
                patch.object(lrcer, "post_process", return_value=translated_sub),
            ):
                lrcer._translate("sample", "zh-cn", transcribed, translated_path)

        local_session.assert_called_once()
        create_translator.assert_called_once()
        close_primary.assert_called_once()
        deterministic.assert_called_once()
        self.assertIs(review.call_args.kwargs["translator"], create_translator.return_value)
        translated_sub.set_texts.assert_called_once_with(["修订"], lang="zh-cn")

    def test_context_plus_only_applies_high_risk_revisions(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {
                "items": [
                    {"id": 1, "risk": "low", "issues": [], "revised_translation": None},
                    {"id": 2, "risk": "high", "issues": ["meaning"], "revised_translation": "修订二"},
                ]
            },
            ensure_ascii=False,
        )
        translator = MagicMock()
        translator.metrics = {"retries": 0, "splits": 0, "atomic_ids": []}
        translator.make_chunks_by_tokens.return_value = [[(1, "one"), (2, "two")]]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps({"compare": [{"idx": 1, "output": "草稿一"}, {"idx": 2, "output": "草稿二"}]})
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)):
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["one", "two"],
                    ["草稿一", "草稿二"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                    translator=translator,
                )

            saved = json.loads(checkpoint.read_text())

        self.assertEqual(output, ["草稿一", "修订二"])
        self.assertEqual(saved["pipeline_stage"], "complete")
        self.assertFalse(saved["review_incomplete"])
        self.assertEqual(saved["raw_hymt2_translations"], ["草稿一", "草稿二"])
        self.assertEqual(saved["final_translations"], ["草稿一", "修订二"])
        self.assertEqual([item["output"] for item in saved["compare"]], ["草稿一", "草稿二"])

    def test_context_plus_review_failure_keeps_hymt2_draft(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = ""
        translator = MagicMock()
        translator.metrics = {"retries": 1, "splits": 0, "atomic_ids": []}
        translator.make_chunks_by_tokens.return_value = [[(1, "one")]]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(json.dumps({"compare": [{"idx": 1, "output": "草稿"}]}))
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)):
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["one"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                    translator=translator,
                )
            saved = json.loads(checkpoint.read_text())

        self.assertEqual(output, ["草稿"])
        self.assertEqual(saved["pipeline_stage"], "review_incomplete")
        self.assertTrue(saved["review_incomplete"])
        self.assertEqual(saved["review_failed_chunks"], [1])
        self.assertTrue(lrcer.review_statuses["sample"]["incomplete"])

    def test_completed_review_checkpoint_does_not_reload_context_model(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "source_fingerprint": "fingerprint",
                        "hymt2_mode": "context-plus",
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "raw_hymt2_translations": ["草稿"],
                        "review_chunks": [[1]],
                        "reviewed_chunks": [1],
                        "review_failed_chunks": [],
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "review_results": {
                            "1": [{"id": 1, "risk": "high", "issues": ["meaning"], "revised_translation": "修订"}]
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot") as create_context_chatbot:
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["source"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(output, ["修订"])
        self.assertEqual(saved["pipeline_stage"], "complete")
        create_context_chatbot.assert_not_called()

    def test_corrupt_reviewed_chunk_index_is_reprocessed(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {"items": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "raw_hymt2_translations": ["草稿"],
                        "review_chunks": [[1]],
                        "reviewed_chunks": [99],
                        "review_failed_chunks": [],
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "review_results": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)) as create_context_chatbot:
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["source"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(output, ["草稿"])
        self.assertEqual(saved["reviewed_chunks"], [1])
        self.assertEqual(saved["pipeline_stage"], "complete")
        create_context_chatbot.assert_called_once()

    def test_polluted_saved_revision_is_discarded_and_reprocessed(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {"items": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "raw_hymt2_translations": ["草稿"],
                        "review_chunks": [[1]],
                        "reviewed_chunks": [1],
                        "review_failed_chunks": [],
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "review_results": {
                            "1": [
                                {
                                    "id": 1,
                                    "risk": "high",
                                    "issues": ["meaning"],
                                    "revised_translation": '<seg id="1">污染</seg>',
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)):
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["source"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )

        self.assertEqual(output, ["草稿"])

    def test_mismatched_saved_review_result_ids_are_reprocessed(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {"items": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "raw_hymt2_translations": ["草稿"],
                        "review_chunks": [[1]],
                        "reviewed_chunks": [1],
                        "review_failed_chunks": [],
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "review_results": {"1": [{"id": 99, "risk": "low", "issues": [], "revised_translation": None}]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)) as create_context_chatbot:
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["source"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )

        self.assertEqual(output, ["草稿"])
        create_context_chatbot.assert_called_once()

    def test_mismatched_saved_final_translation_count_is_reprocessed(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        bot = MagicMock()
        bot.api_fees = []
        bot.message.return_value = [MagicMock()]
        bot.get_content.return_value = json.dumps(
            {"items": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "raw_hymt2_translations": ["草稿"],
                        "review_chunks": [[1]],
                        "reviewed_chunks": [1],
                        "review_failed_chunks": [],
                        "review_protocol_version": HyMT2RiskReviewAgent.REVIEW_PROTOCOL_VERSION,
                        "review_results": {"1": [{"id": 1, "risk": "low", "issues": [], "revised_translation": None}]},
                        "final_translations": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(lrcer, "_create_context_chatbot", return_value=(bot, None)) as create_context_chatbot:
                output = lrcer._review_hymt2_translations(
                    "sample",
                    ["source"],
                    ["草稿"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )

        self.assertEqual(output, ["草稿"])
        create_context_chatbot.assert_called_once()

    def test_review_rejects_mismatched_raw_translation_count(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "compare.json"
            checkpoint.write_text(json.dumps({"compare": [], "raw_hymt2_translations": ["只有一条"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw Hy-MT2 translations"):
                lrcer._review_hymt2_translations(
                    "sample",
                    ["one", "two"],
                    ["草稿一", "草稿二"],
                    src_lang="en",
                    target_lang="zh-cn",
                    brief=TranslationBrief(summary="summary"),
                    compare_path=checkpoint,
                )

    def test_context_plus_resumes_incomplete_review_without_reloading_hymt2(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]
        translated_sub = MagicMock()

        from openlrc.chunking import CHUNK_PLANNER_VERSION, chunk_plan_signature, plan_translation_chunks

        signature = chunk_plan_signature(
            plan_translation_chunks(transcribed.texts, timestamps=[(0.0, 1.0)]), timestamps=[(0.0, 1.0)]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translated_path = Path(tmpdir) / "sample_translated.json"
            translated_path.write_text("{}", encoding="utf-8")
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "source_fingerprint": "fingerprint",
                        "hymt2_mode": "context-plus",
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "translation_brief": {"summary": "summary"},
                        "raw_hymt2_translations": ["草稿"],
                        "brief_prompt_version": 3,
                        "chunk_planner_version": CHUNK_PLANNER_VERSION,
                        "chunk_signature": signature,
                        "review_incomplete": True,
                        "review_failed_chunks": [1],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch("openlrc.hymt2_pipeline.source_fingerprint", return_value="fingerprint"),
                patch.object(lrcer, "_review_hymt2_translations", return_value=["修订"]) as review,
                patch.object(lrcer, "_create_translator") as create_translator,
                patch("openlrc.openlrc.deepcopy", return_value=translated_sub),
                patch("openlrc.openlrc.Subtitle.from_json", return_value=translated_sub),
                patch.object(lrcer, "post_process", return_value=translated_sub),
            ):
                lrcer._translate("sample", "zh-cn", transcribed, translated_path)

        create_translator.assert_not_called()
        review.assert_called_once()
        self.assertIsNone(review.call_args.kwargs.get("translator"))
        translated_sub.set_texts.assert_called_once_with(["修订"], lang="zh-cn")

    def test_context_plus_stale_planner_checkpoint_retranslates_before_review(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        transcribed.segments = [MagicMock(start=0.0, end=1.0)]
        translated_sub = MagicMock()
        translator = MagicMock()
        translator.translate.return_value = ["新草稿"]
        translator.api_fee = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            translated_path = Path(tmpdir) / "sample_translated.json"
            translated_path.write_text("{}", encoding="utf-8")
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "旧草稿"}],
                        "translation_brief": {"summary": "summary"},
                        "raw_hymt2_translations": ["旧草稿"],
                        "brief_prompt_version": 3,
                        "chunk_planner_version": 1,
                        "chunk_signature": "stale",
                        "review_incomplete": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    lrcer,
                    "_prepare_hymt2_brief",
                    return_value=(TranslationBrief(summary="summary"), {"schema_version": 2}),
                ),
                patch.object(lrcer, "_local_llm_session"),
                patch.object(lrcer, "_create_translator", return_value=translator),
                patch.object(lrcer, "_close_primary_local_stage"),
                patch.object(lrcer, "_review_hymt2_translations", return_value=["修订"]) as review,
                patch("openlrc.openlrc.deepcopy", return_value=translated_sub),
                patch("openlrc.openlrc.Subtitle.from_json", return_value=translated_sub),
                patch.object(lrcer, "post_process", return_value=translated_sub),
            ):
                lrcer._translate("sample", "zh-cn", transcribed, translated_path)

        translator.translate.assert_called_once()
        self.assertEqual(review.call_args.args[2], ["新草稿"])

    @patch("openlrc.agents.create_chatbot", side_effect=_mock_create_chatbot)
    @patch("openlrc.local_llm_server.LocalLLMServer")
    def test_local_chatbot_starts_server_and_injects_base_url(self, mock_server_cls, mock_create_chatbot):
        server = mock_server_cls.return_value
        server.ensure_running.return_value = "http://127.0.0.1:8088/v1"

        lrcer = LRCer.local()
        _ = lrcer.chatbot

        model_config = mock_create_chatbot.call_args.args[0]
        self.assertEqual(model_config.name, DEFAULT_LLAMA_MODEL_ALIAS)
        self.assertEqual(model_config.base_url, "http://127.0.0.1:8088/v1")
        self.assertEqual(model_config.api_key, "openlrc-local")
        server.ensure_running.assert_called_once()

    @patch("openlrc.agents.create_chatbot", side_effect=_mock_create_chatbot)
    @patch("openlrc.local_llm_server.LocalLLMServer")
    def test_close_stops_owned_local_server(self, mock_server_cls, _mock_create_chatbot):
        server = mock_server_cls.return_value
        server.ensure_running.return_value = "http://127.0.0.1:8088/v1"

        lrcer = LRCer.local()
        _ = lrcer.chatbot
        lrcer.close()

        server.close.assert_called_once()

    @patch("openlrc.agents.create_chatbot", side_effect=_mock_create_chatbot)
    @patch("openlrc.local_llm_server.LocalLLMServer")
    def test_local_llm_config_without_chatbot_defaults_to_local_provider(self, mock_server_cls, mock_create_chatbot):
        server = mock_server_cls.return_value
        server.ensure_running.return_value = "http://127.0.0.1:8088/v1"

        lrcer = LRCer(translation=TranslationConfig(local_llm=LocalLLMConfig()))
        _ = lrcer.chatbot

        model_config = mock_create_chatbot.call_args.args[0]
        self.assertEqual(model_config.name, DEFAULT_LLAMA_MODEL_ALIAS)
        self.assertEqual(model_config.base_url, "http://127.0.0.1:8088/v1")
