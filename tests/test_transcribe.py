#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import patch

from faster_whisper.transcribe import Segment, Word

from openlrc.transcribe import Transcriber, TranscriptionInfo

return_tuple = ([
                    Segment(
                        0, 0, 0, 3, 'hello world', [], 0, 0.8, 0, 0, words=[
                            Word(0, 1.5, 'hello', probability=0.8),
                            Word(1.6, 3, ' world', probability=0.8)
                        ]),
                    Segment(
                        0, 0, 3, 6, 'hello world', [], 0, 0.8, 0, 0, words=[
                            Word(3, 4.5, 'hello', probability=0.8),
                            Word(4.6, 6, ' world', probability=0.8)
                        ]),
                ], TranscriptionInfo('en', 30))


class TestTranscriber(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path('data/test_audio.wav')

    def test_transcribe_success(self):
        with patch('openlrc.transcribe.WhisperModel') as MockModel:
            MockModel.return_value.transcribe.return_value = return_tuple

            transcriber = Transcriber(model_name='tiny')
            result, info = transcriber.transcribe(self.audio_path)
            self.assertIsNotNone(result)
            self.assertEqual(round(info.duration), 30)

    def test_audio_file_not_found(self):
        with patch('openlrc.transcribe.WhisperModel') as MockModel:
            MockModel.return_value.transcribe.return_value = return_tuple

            transcriber = Transcriber(model_name='tiny')
            with self.assertRaises(FileNotFoundError):
                transcriber.transcribe('audio.wav')
