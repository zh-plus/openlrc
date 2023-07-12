#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from faster_whisper.transcribe import Segment, Word

from openlrc.openlrc import LRCer
from openlrc.transcribe import TranscriptionInfo
from openlrc.utils import extend_filename


@patch('faster_whisper.WhisperModel', MagicMock())
@patch('openlrc.transcribe.Transcriber.transcribe',
       MagicMock(return_value=(
               [
                   Segment(
                       0, 0, 0, 3, 'hello world1', [], 0, 0.8, 0, 0, words=[
                           Word(0, 1.5, 'hello', probability=0.8), Word(1.6, 3, ' world1', probability=0.8)
                       ]),
                   Segment(
                       0, 0, 3, 6, 'hello world2', [], 0, 0.8, 0, 0, words=[
                           Word(3, 4.5, 'hello', probability=0.8), Word(4.6, 6, ' world2', probability=0.8)
                       ])
               ],
               TranscriptionInfo('en', 6.0))))
class TestLRCer(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_path = Path('data/test_audio.wav')
        self.video_path = Path('data/test_video.mp4')

    def tearDown(self) -> None:
        def clear_paths(input_path):
            transcribed = extend_filename(input_path, '_transcribed').with_suffix('.json')
            optimized = extend_filename(transcribed, '_optimized')
            translated = extend_filename(optimized, '_translated')
            compare_path = extend_filename(input_path, '_compare').with_suffix('.json')

            json_path = input_path.with_suffix('.json')
            lrc_path = input_path.with_suffix('.lrc')
            srt_path = input_path.with_suffix('.srt')

            [p.unlink(missing_ok=True) for p in
             [transcribed, optimized, translated, compare_path, json_path, lrc_path, srt_path]]

        clear_paths(self.audio_path)
        clear_paths(self.video_path)

        self.video_path.with_suffix('.wav').unlink(missing_ok=True)

    @patch('openlrc.translate.GPTTranslator.translate',
           MagicMock(return_value=['test translation1', 'test translation2']))
    def test_single_audio_transcription_translation(self):
        lrcer = LRCer()
        lrcer.run(self.audio_path)

    @patch('openlrc.translate.GPTTranslator.translate',
           MagicMock(return_value=['test translation1', 'test translation2']))
    def test_multiple_audio_transcription_translation(self):
        lrcer = LRCer()
        lrcer.run([self.audio_path, self.video_path])

    #  Tests that an error is raised when an audio file is not found
    def test_audio_file_not_found(self):
        lrcer = LRCer()
        with self.assertRaises(FileNotFoundError):
            lrcer.run('data/invalid.mp3')

    #  Tests that a video file can be transcribed and translated
    def test_video_file_transcription_translation(self):
        lrcer = LRCer()
        lrcer.run('data/test_video.mp4')

    @patch('openlrc.translate.GPTTranslator.translate', MagicMock(side_effect=Exception('test exception')))
    def test_translation_error(self):
        lrcer = LRCer()
        with self.assertRaises(Exception):
            lrcer.run(self.audio_path)

    @patch('openlrc.translate.GPTTranslator.translate', MagicMock(side_effect=Exception('test exception')))
    def test_skip_translation(self):
        lrcer = LRCer()
        lrcer.run('data/test_video.mp4', skip_trans=True)
