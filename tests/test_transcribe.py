#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from openlrc.transcribe import Transcriber


@patch('whisperx.load_model', MagicMock())
@patch('whisperx.asr.FasterWhisperPipeline.transcribe', MagicMock(return_value={'language': 'en', 'segments': []}))
@patch('whisperx.load_align_model', MagicMock(return_value=(None, None)))
@patch('whisperx.align', MagicMock(return_value={
    'segments': [
        {'text': 'hello world', 'start': 0, 'end': 3, 'words': [
            {'text': 'hello', 'start': 0, 'end': 1.5}, {'text': 'world', 'start': 1.6, 'end': 3}
        ]},
        {'text': 'hello world', 'start': 3, 'end': 6, 'words': [
            {'word': 'hello', 'start': 3, 'end': 4.5}, {'word': 'world', 'start': 4.6, 'end': 6}
        ]}
    ],
    'word_segments': [{'word': 'hello', 'start': 0, 'end': 1.5}, {'word': 'world', 'start': 1.6, 'end': 3},
                      {'word': 'hello', 'start': 3, 'end': 4.5}, {'word': 'world', 'start': 4.6, 'end': 6}]
}))
class TestTranscriber(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path('data/test_audio.wav')
        self.transcriber = Transcriber(model_name='tiny')

    def test_transcribe_success(self):
        result, info = self.transcriber.transcribe(self.audio_path)
        assert result['sentences']
        assert info.duration == 24347.887142857144

    #  Tests that an audio file not found raises an error
    def test_audio_file_not_found(self):
        with self.assertRaises(RuntimeError):
            self.transcriber.transcribe('audio.wav')
