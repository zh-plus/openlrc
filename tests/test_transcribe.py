#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import pytest

from openlrc.transcribe import Transcriber


class TestTranscriber:
    #  Tests that audio file is transcribed successfully
    def test_transcribe_success(self, mocker):
        mocker.patch('whisperx.load_model')
        mocker.patch('whisperx.load_audio')
        mocker.patch('whisperx.load_align_model', return_value=[None, None])
        mocker.patch('whisperx.align')
        mocker.patch('whisperx.asr.FasterWhisperPipeline.transcribe', return_value={'language': '', 'segments': []})
        mocker.patch('openlrc.transcribe.Transcriber.sentence_align',
                     return_value={'sentences': [], 'word_segments': []})
        mocker.patch('json.dump')
        transcriber = Transcriber()
        result, info = transcriber.transcribe('data/test_video.wav')
        assert result['sentences']
        assert info.duration == 24347.887142857144

    #  Tests that an empty audio file raises an error
    def test_empty_audio_file(self, mocker):
        mocker.patch('whisperx.load_model')
        mocker.patch('whisperx.load_audio', return_value=b'')
        transcriber = Transcriber()
        with pytest.raises(ValueError):
            transcriber.transcribe('audio.wav')

    #  Tests that an audio file not found raises an error
    def test_audio_file_not_found(self, mocker):
        mocker.patch('whisperx.load_model')
        mocker.patch('whisperx.load_audio', side_effect=FileNotFoundError)
        transcriber = Transcriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe('audio.wav')

    #  Tests that an invalid model name raises an error
    def test_invalid_model_name(self, mocker):
        mocker.patch('whisperx.load_model', side_effect=ValueError)
        transcriber = Transcriber()
        with pytest.raises(ValueError):
            transcriber.transcribe('audio.wav')

    #  Tests that an invalid compute type raises an error
    def test_invalid_compute_type(self, mocker):
        mocker.patch('whisperx.load_model', side_effect=ValueError)
        transcriber = Transcriber(compute_type='invalid')
        with pytest.raises(ValueError):
            transcriber.transcribe('audio.wav')
