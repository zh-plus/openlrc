#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from openlrc.preprocess import Preprocessor

TEST_DATA_DIR = Path(__file__).parent / "data"


class TestPreprocessor(unittest.TestCase):
    def tearDown(self) -> None:
        preprocessed_path = TEST_DATA_DIR / "preprocessed"
        shutil.rmtree(preprocessed_path, ignore_errors=True)

    @patch("openlrc.preprocess.FFmpegNormalize")
    def test_loudness_normalization_returns_path_objects(self, mock_norm):
        mock_norm.return_value.run_normalization.return_value = None
        preprocessor = Preprocessor(TEST_DATA_DIR / "test_audio.wav")
        ln_paths = preprocessor.loudness_normalization(preprocessor.audio_paths)
        self.assertIsInstance(ln_paths, list)
        self.assertIsInstance(ln_paths[0], Path)

    @patch("openlrc.preprocess.Path.rename")
    @patch("openlrc.preprocess.Preprocessor.loudness_normalization")
    def test_run_normalizes_original_audio_and_returns_path_objects(self, mock_loudness_normalization, mock_rename):
        mock_rename.return_value = Path("audio_processed.wav")
        mock_loudness_normalization.return_value = [Path("audio_ln.wav")]
        audio_path = TEST_DATA_DIR / "test_audio.wav"
        preprocessor = Preprocessor(audio_path)
        final_processed = preprocessor.run()
        mock_loudness_normalization.assert_called_once_with([audio_path])
        self.assertIsInstance(final_processed, list)
        self.assertIsInstance(final_processed[0], Path)

    def test_preprocessor_raises_exception_when_audio_paths_is_not_a_list_or_a_string(self):
        with self.assertRaises(TypeError):
            Preprocessor(123)

    def test_preprocessor_options_are_keyword_only(self):
        with self.assertRaises(TypeError):
            Preprocessor(TEST_DATA_DIR / "test_audio.wav", "custom")  # type: ignore[misc]
