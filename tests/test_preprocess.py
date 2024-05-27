#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import torch

from openlrc.preprocess import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def tearDown(self) -> None:
        preprocessed_path = Path('data/preprocessed')
        shutil.rmtree(preprocessed_path, ignore_errors=True)

    @patch('openlrc.preprocess.enhance')
    @patch('openlrc.preprocess.init_df')
    @patch('openlrc.preprocess.load_audio')
    @patch('openlrc.preprocess.save_audio')
    @patch('openlrc.preprocess.release_memory')
    def test_noise_suppression_returns_path_objects(self, mock_release_memory, mock_save_audio, mock_load_audio,
                                                    mock_init_df, mock_enhance):
        mock_enhance.return_value = torch.zeros((2, 300 * 16000))
        mock_init_df.return_value = (Mock(), Mock(), Mock())

        mock_info = Mock()
        mock_info.sample_rate = 16000
        mock_load_audio.return_value = (torch.zeros((2, 1200 * 16000)), mock_info)

        mock_save_audio.return_value = None
        mock_release_memory.return_value = None
        preprocessor = Preprocessor('audio.wav')
        ns_paths = preprocessor.noise_suppression(preprocessor.audio_paths)
        self.assertIsInstance(ns_paths, list)
        self.assertIsInstance(ns_paths[0], Path)

    @patch('openlrc.preprocess.FFmpegNormalize')
    def test_loudness_normalization_returns_path_objects(self, mock_norm):
        mock_norm.return_value.run_normalization.return_value = None
        preprocessor = Preprocessor('data/test_audio.wav')
        ln_paths = preprocessor.loudness_normalization(preprocessor.audio_paths)
        self.assertIsInstance(ln_paths, list)
        self.assertIsInstance(ln_paths[0], Path)

    @patch('openlrc.preprocess.Path.rename')
    @patch('openlrc.preprocess.Preprocessor.noise_suppression')
    @patch('openlrc.preprocess.Preprocessor.loudness_normalization')
    def test_run_returns_path_objects(self, mock_loudness_normalization, mock_noise_suppression, mock_rename):
        mock_rename.return_value = Path('audio_processed.wav')
        mock_noise_suppression.return_value = [Path('audio_ns.wav')]
        mock_loudness_normalization.return_value = [Path('audio_ln.wav')]
        preprocessor = Preprocessor('audio.wav')
        final_processed = preprocessor.run()
        self.assertIsInstance(final_processed, list)
        self.assertIsInstance(final_processed[0], Path)

    def test_preprocessor_raises_exception_when_audio_paths_is_not_a_list_or_a_string(self):
        with self.assertRaises(TypeError):
            Preprocessor(123)
