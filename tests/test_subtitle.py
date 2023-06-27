#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import json
from pathlib import Path

from openlrc.subtitle import Subtitle


class TestSubtitle:
    #  Tests that a Subtitle object can be created with valid language, generator, segments, and filename
    def test_create_subtitle_valid(self):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = Path('test.json')
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        assert subtitle.lang == 'en'
        assert subtitle.generator == 'test'
        assert len(subtitle.segments) == 2
        assert subtitle.segments[0].start == 0.0
        assert subtitle.segments[0].end == 1.0
        assert subtitle.segments[0].text == 'Hello'
        assert subtitle.segments[1].start == 1.0
        assert subtitle.segments[1].end == 2.0
        assert subtitle.segments[1].text == 'World'
        assert subtitle.filename == filename

    #  Tests that the length of a Subtitle object can be obtained
    def test_get_subtitle_length(self):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        assert len(subtitle) == 2

    #  Tests that the texts of a Subtitle object can be obtained
    def test_get_subtitle_texts(self):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        assert subtitle.texts == ['Hello', 'World']

    #  Tests that the texts of a Subtitle object can be set with a list of valid texts
    def test_set_subtitle_texts_valid(self):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        new_texts = ['Hi', 'Planet']
        subtitle.set_texts(new_texts)
        assert subtitle.texts == new_texts

    #  Tests that a Subtitle object can be saved to a valid JSON file
    def test_save_subtitle_valid(self, tmp_path):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = tmp_path / 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        subtitle.save(filename)
        with open(filename, 'r') as f:
            content = json.load(f)
        assert content['language'] == 'en'
        assert content['generator'] == 'test'
        assert len(content['segments']) == 2
        assert content['segments'][0]['start'] == 0.0
        assert content['segments'][0]['end'] == 1.0
        assert content['segments'][0]['text'] == 'Hello'
        assert content['segments'][1]['start'] == 1.0
        assert content['segments'][1]['end'] == 2.0
        assert content['segments'][1]['text'] == 'World'

    #  Tests that a Subtitle object can be converted to LRC format
    def test_convert_subtitle_to_lrc_valid(self, tmp_path):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = tmp_path / 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        subtitle.to_lrc()
        lrc_filename = filename.with_suffix('.lrc')
        with open(lrc_filename, 'r') as f:
            content = f.read()
        assert '[00:00.00] Hello' in content
        assert '[00:01.00] World' in content

    def test_test_convert_subtitle_to_srt_valid(self, tmp_path):
        segments = [{'start': 0.0, 'end': 1.0, 'text': 'Hello'}, {'start': 1.0, 'end': 2.0, 'text': 'World'}]
        filename = tmp_path / 'test.json'
        subtitle = Subtitle(language='en', generator='test', segments=segments, filename=filename)
        subtitle.to_srt()
        lrc_filename = filename.with_suffix('.srt')
        with open(lrc_filename, 'r') as f:
            content = f.read()
        assert '1\n00:00:00,000 --> 00:00:01,000\nHello' in content
        assert '2\n00:00:01,000 --> 00:00:02,000\nWorld' in content
