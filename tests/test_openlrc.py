#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
