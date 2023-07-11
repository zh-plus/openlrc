#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from faster_whisper.transcribe import Segment, Word

from openlrc.transcribe import Transcriber, TranscriptionInfo


@patch('faster_whisper.WhisperModel', MagicMock())
@patch('faster_whisper.WhisperModel.transcribe', MagicMock(
    return_value=([
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
                  ], TranscriptionInfo('en', 30))))
class TestTranscriber(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path('data/test_audio.wav')
        self.transcriber = Transcriber(model_name='tiny')

    def test_transcribe_success(self):
        result, info = self.transcriber.transcribe(self.audio_path)
        self.assertIsNotNone(result)
        self.assertEqual(round(info.duration), 30)

    #  Tests that an audio file not found raises an error
    def test_audio_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.transcriber.transcribe('audio.wav')
