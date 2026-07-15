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

from openlrc.config import ContextLLMConfig, HyMT2Mode, LocalLLMConfig
from openlrc.context import TranslationBrief
from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_ALIAS,
    HY_MT2_7B_MODEL_ALIAS,
    HY_MT2_7B_MODEL_FILE,
    HY_MT2_PROMPT_PROFILE,
)
from openlrc.models import ModelProvider
from openlrc.openlrc import LRCer, TranscriptionConfig, TranslationConfig
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
            compare_path = extend_filename(input_path, "_compare").with_suffix(".json")

            json_path = input_path.with_suffix(".json")
            lrc_path = input_path.with_suffix(".lrc")
            srt_path = input_path.with_suffix(".srt")

            [
                p.unlink(missing_ok=True)
                for p in [transcribed, optimized, translated, compare_path, json_path, lrc_path, srt_path]
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
                output = TEST_DATA_DIR / "context-plus-output.json"
                output.unlink(missing_ok=True)
                lrcer._translate("sample", "zh-cn", transcribed, output)

        self.assertEqual(events, ["context-cr", "hy-start", "hy-session-end", "hy-close", "context-review"])

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

    def test_context_plus_resumes_incomplete_review_without_reloading_hymt2(self):
        context_llm = ContextLLMConfig.online(provider=ModelProvider.OPENAI, model="gpt-4.1-nano")
        lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT_PLUS, context_llm=context_llm)
        transcribed = MagicMock()
        transcribed.texts = ["source"]
        transcribed.lang = "en"
        translated_sub = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            translated_path = Path(tmpdir) / "sample_translated.json"
            translated_path.write_text("{}", encoding="utf-8")
            checkpoint = Path(tmpdir) / "sample_compare.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "compare": [{"chunk": 1, "idx": 1, "output": "草稿"}],
                        "translation_brief": {"summary": "summary"},
                        "raw_hymt2_translations": ["草稿"],
                        "review_incomplete": True,
                        "review_failed_chunks": [1],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
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
