#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import shutil
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from faster_whisper.transcribe import Segment, Word

from openlrc.openlrc import LRCer, TranscriptionConfig, TranslationConfig
from openlrc.transcribe import TranscriptionInfo
from openlrc.utils import extend_filename

# Shared test config — avoids repeating these in every test method.
_TEST_TRANSCRIPTION = TranscriptionConfig(whisper_model="tiny", compute_type="default", device="cpu")


@patch("openlrc.transcribe.BatchedInferencePipeline", MagicMock())
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
class TestLRCer(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path("data/test_audio.wav")
        self.video_path = Path("data/test_video.mp4")
        self.nospeech_video_path = Path("data/test_nospeech_video.mp4")

    def tearDown(self) -> None:
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

        shutil.rmtree("data/preprocessed", ignore_errors=True)

    # ------------------------------------------------------------------
    # Pipeline tests (using new config API)
    # ------------------------------------------------------------------

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_single_audio_transcription_translation(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.audio_path)
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_multiple_audio_transcription_translation(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run([self.audio_path, self.video_path])
        self.assertTrue(result)
        self.assertEqual(len(result), 2)

    def test_audio_file_not_found(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        with self.assertRaises(FileNotFoundError):
            lrcer.run("data/invalid.mp3")

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_video_file_transcription_translation(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run("data/test_video.mp4")
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_nospeech_video_file_transcription_translation(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run("data/test_nospeech_video.mp4")
        self.assertTrue(result)

    @patch("openlrc.translate.LLMTranslator.translate", MagicMock(side_effect=Exception("test exception")))
    def test_translation_error(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        with self.assertRaises(Exception):
            lrcer.run(self.audio_path)

    @patch("openlrc.translate.LLMTranslator.translate", MagicMock(side_effect=Exception("test exception")))
    def test_skip_translation(self):
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run("data/test_video.mp4", skip_trans=True)
        self.assertTrue(result)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_skip_preprocess(self):
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

    def test_skip_preprocess_file_not_found(self):
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
    def test_skip_trans_skips_translate_but_calls_post_process(self, mock_post_process, mock_translate):
        """skip_trans=True should skip translation but still post-process transcription."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        lrcer.run(self.audio_path, skip_trans=True)
        mock_translate.assert_not_called()
        mock_post_process.assert_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_normal_run_calls_translate(self, mock_translate):
        """skip_trans=False (default) should invoke LLMTranslator.translate."""
        mock_translate.return_value = ["test translation1", "test translation2"]
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        lrcer.run(self.audio_path)
        mock_translate.assert_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_transcribe_returns_json_paths(self, mock_translate):
        """transcribe() should return transcribed JSON paths without triggering translation."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.transcribe(self.audio_path)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].exists())
        self.assertTrue(result[0].name.endswith("_transcribed.json"))
        mock_translate.assert_not_called()

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_translate_processes_transcribed_json(self, mock_translate):
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
    def test_translate_independent_video_transcription_outputs_srt(self):
        """translate() should keep video-origin output as .srt even on a fresh LRCer."""
        lrcer_transcribe = LRCer(transcription=_TEST_TRANSCRIPTION)
        transcribed = lrcer_transcribe.transcribe(self.video_path)
        self.assertEqual(len(transcribed), 1)

        lrcer_translate = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer_translate.translate(transcribed, target_lang="zh-cn")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].suffix, ".srt")

    @patch("openlrc.translate.LLMTranslator.translate")
    def test_run_skip_trans_deduplicates_duplicate_inputs(self, mock_translate):
        """run(skip_trans=True) should transcribe duplicate input paths only once."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run([self.audio_path, self.audio_path], skip_trans=True)
        self.assertTrue(result)
        self.assertEqual(len(result), 1)
        mock_translate.assert_not_called()

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_run_full_pipeline(self):
        """run() with default args should produce subtitle output (full pipeline)."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        result = lrcer.run(self.audio_path)
        self.assertTrue(result)
        self.assertEqual(len(result), 1)

    # ------------------------------------------------------------------
    # Config and constructor tests
    # ------------------------------------------------------------------

    def test_translate_only_does_not_instantiate_transcriber(self):
        """LRCer with only TranslationConfig should not create a Transcriber."""
        lrcer = LRCer(translation=TranslationConfig())
        # Access the private attribute directly — should be None (lazy init).
        self.assertIsNone(lrcer._transcriber)

    @patch(
        "openlrc.translate.LLMTranslator.translate", MagicMock(return_value=["test translation1", "test translation2"])
    )
    def test_transcriber_created_on_first_transcribe_call(self):
        """Transcriber should be lazily created when transcribe() is first called."""
        lrcer = LRCer(transcription=_TEST_TRANSCRIPTION)
        self.assertIsNone(lrcer._transcriber)
        lrcer.transcribe(self.audio_path)
        self.assertIsNotNone(lrcer._transcriber)

    def test_default_option_merging_parity(self):
        """Config-based construction should merge default options the same as legacy kwargs."""
        from openlrc.defaults import default_asr_options

        custom_asr = {"beam_size": 3}
        custom_vad = {"threshold": 0.7}
        custom_preprocess = {"atten_lim_db": 20}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = LRCer(
                whisper_model="tiny",
                device="cpu",
                compute_type="default",
                asr_options=custom_asr,
                vad_options=custom_vad,
                preprocess_options=custom_preprocess,
            )

        config_based = LRCer(
            transcription=TranscriptionConfig(
                whisper_model="tiny",
                device="cpu",
                compute_type="default",
                asr_options=custom_asr,
                vad_options=custom_vad,
                preprocess_options=custom_preprocess,
            )
        )

        self.assertEqual(legacy.asr_options, config_based.asr_options)
        self.assertEqual(legacy.vad_options, config_based.vad_options)
        self.assertEqual(legacy.preprocess_options, config_based.preprocess_options)

        # Verify custom values are merged on top of defaults
        self.assertEqual(config_based.asr_options["beam_size"], 3)
        self.assertEqual(config_based.asr_options["best_of"], default_asr_options["best_of"])
        self.assertEqual(config_based.vad_options["threshold"], 0.7)
        self.assertEqual(config_based.preprocess_options["atten_lim_db"], 20)

    def test_legacy_and_config_equivalent_runtime_behavior(self):
        """Legacy kwargs and config objects should produce equivalent LRCer state."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = LRCer(
                whisper_model="tiny",
                device="cpu",
                compute_type="default",
                chatbot_model="gpt-4.1-nano",
                fee_limit=1.5,
                consumer_thread=2,
            )

        config_based = LRCer(
            transcription=TranscriptionConfig(whisper_model="tiny", device="cpu", compute_type="default"),
            translation=TranslationConfig(chatbot_model="gpt-4.1-nano", fee_limit=1.5, consumer_thread=2),
        )

        # Transcription config
        self.assertEqual(legacy._transcription_config.whisper_model, config_based._transcription_config.whisper_model)
        self.assertEqual(legacy._transcription_config.device, config_based._transcription_config.device)
        self.assertEqual(legacy._transcription_config.compute_type, config_based._transcription_config.compute_type)

        # Translation config
        self.assertEqual(legacy.chatbot_model, config_based.chatbot_model)
        self.assertEqual(legacy.fee_limit, config_based.fee_limit)
        self.assertEqual(legacy.consumer_thread, config_based.consumer_thread)

    def test_legacy_kwargs_emit_deprecation_warning(self):
        """Legacy keyword arguments should emit a DeprecationWarning."""
        with self.assertWarns(DeprecationWarning):
            LRCer(whisper_model="tiny", device="cpu", compute_type="default")

    def test_mixing_legacy_and_config_raises_error(self):
        """Passing both legacy kwargs and config objects should raise ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError):
                LRCer(whisper_model="tiny", transcription=TranscriptionConfig(whisper_model="tiny"))

    def test_default_construction_without_arguments(self):
        """LRCer() with no arguments should use default configs."""
        lrcer = LRCer()
        self.assertEqual(lrcer._transcription_config.whisper_model, "large-v3")
        self.assertEqual(lrcer._translation_config.chatbot_model, "gpt-4.1-nano")
        self.assertIsNone(lrcer._transcriber)
