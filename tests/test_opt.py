#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import json
import os
import unittest

from openlrc.opt import SubtitleOptimizer
from openlrc.subtitle import Subtitle


class TestSubtitleOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitle = Subtitle.from_json('data/test_valid_subtitle.json')

    def test_merge_same(self):
        subtitle = self.subtitle
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.merge_same()
        assert len(optimizer.subtitle.segments) == original_len - 1

    def test_merge_short(self):
        subtitle = self.subtitle
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.merge_short()
        assert len(optimizer.subtitle.segments) == original_len - 1

    def test_merge_repeat(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.merge_repeat()
        assert optimizer.subtitle.segments[2].text == '好好...'

    def test_cut_long(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.cut_long(keep=2)
        assert optimizer.subtitle.segments[4].text == '这太(Cut to 2)'

    def test_traditional2mandarin(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.traditional2mandarin()
        assert optimizer.subtitle.segments[5].text == '繁体的字'

    def test_remove_unk(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_unk()
        assert optimizer.subtitle.segments[6].text == 'unk'

    def test_remove_empty(self):
        subtitle = self.subtitle
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_empty()
        assert len(optimizer.subtitle.segments) == original_len - 1

    def test_save(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.perform_all()
        optimizer.save(output_name='data/test_subtitle_optimized.json')

        with open('data/test_subtitle_optimized.json', 'r', encoding='utf-8') as f:
            optimized_subtitle = json.load(f)

        assert optimized_subtitle['language'] == 'zh'
        assert optimized_subtitle['generator'] == 'test'
        assert len(optimized_subtitle['segments']) == 6

        os.remove('data/test_subtitle_optimized.json')
