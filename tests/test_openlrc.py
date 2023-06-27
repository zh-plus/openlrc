#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

from pathlib import Path

import pytest

from openlrc.openlrc import LRCer


#  Tests that a single audio file can be transcribed and translated
def test_single_audio_transcription_translation(mocker):
    mocker.patch.object(LRCer, 'pre_process', return_value=[Path('tests/data/test.mp3')])
    mocker.patch.object(LRCer, 'to_json', return_value=None)
    mocker.patch.object(LRCer, 'post_process', return_value=None)
    mocker.patch.object(LRCer, 'transcription_producer', return_value=None)
    mocker.patch.object(LRCer, 'transcription_consumer', return_value=None)
    mocker.patch.object(LRCer, 'translation_worker', return_value=None)
    lrcer = LRCer()
    lrcer.run('tests/data/test.mp3')


#  Tests that multiple audio files can be transcribed and translated
def test_multiple_audio_transcription_translation(mocker):
    mocker.patch.object(LRCer, 'pre_process', return_value=[Path('tests/data/test.mp3'), Path('tests/data/test2.mp3')])
    mocker.patch.object(LRCer, 'to_json', return_value=None)
    mocker.patch.object(LRCer, 'post_process', return_value=None)
    mocker.patch.object(LRCer, 'transcription_producer', return_value=None)
    mocker.patch.object(LRCer, 'transcription_consumer', return_value=None)
    mocker.patch.object(LRCer, 'translation_worker', return_value=None)
    lrcer = LRCer()
    lrcer.run(['tests/data/test.mp3', 'tests/data/test2.mp3'])


#  Tests that an error is raised when an audio file is not found
def test_audio_file_not_found(mocker):
    lrcer = LRCer()
    with pytest.raises(FileNotFoundError):
        lrcer.run('tests/data/invalid.mp3')


#  Tests that an error is raised when there is an error in the translation process
# def test_translation_error(mocker):
#     mocker.patch.object(LRCer, 'pre_process', return_value=['tests/data/test.mp3'])
#     mocker.patch.object(LRCer, 'to_json', return_value=None)
#     mocker.patch.object(LRCer, 'post_process', return_value=None)
#     mocker.patch.object(LRCer, 'transcription_producer', return_value=None)
#     # mocker.patch.object(LRCer, 'transcription_consumer', return_value=None)
#     mocker.patch.object(LRCer, 'translation_worker', side_effect=Exception('Test error'))
#     lrcer = LRCer()
#     with pytest.raises(Exception):
#         lrcer.run('tests/data/test.mp3')

#  Tests that a video file can be transcribed and translated
def test_video_file_transcription_translation(mocker):
    mocker.patch.object(LRCer, 'pre_process', return_value=[Path('tests/data/test.mp4')])
    mocker.patch.object(LRCer, 'to_json', return_value=None)
    mocker.patch.object(LRCer, 'post_process', return_value=None)
    mocker.patch.object(LRCer, 'transcription_producer', return_value=None)
    mocker.patch.object(LRCer, 'transcription_consumer', return_value=None)
    mocker.patch.object(LRCer, 'translation_worker', return_value=None)
    lrcer = LRCer()
    lrcer.run('tests/data/test.mp4')
