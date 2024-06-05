#  Copyright (C) 2024. Hao Zheng
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
        self.assertEqual(len(optimizer.subtitle.segments), original_len - 1)

    def test_merge_short(self):
        subtitle = self.subtitle
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.merge_short()
        self.assertEqual(len(optimizer.subtitle.segments), original_len - 1)

    def test_merge_repeat(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.merge_repeat()
        self.assertEqual(optimizer.subtitle.segments[2].text, '好好...')

    def test_cut_long(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.cut_long(max_length=2)
        self.assertEqual(optimizer.subtitle.segments[4].text, '这太')

    def test_traditional2mandarin(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.traditional2mandarin()
        self.assertEqual(optimizer.subtitle.segments[5].text, '繁体的字')

    def test_punctuation_optimization(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.punctuation_optimization()
        self.assertEqual(optimizer.subtitle.segments[0].text, '你好，你好...你好！你好。')

    def test_punctuation_optimization_with_dots(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.punctuation_optimization()
        self.assertEqual(optimizer.subtitle.segments[9].text, '1. 测试。这是1.2节。')

    def test_remove_unk(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_unk()
        self.assertEqual(optimizer.subtitle.segments[6].text, 'unk')

    def test_remove_empty(self):
        subtitle = self.subtitle
        subtitle.segments[0].text = ''
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_empty()
        self.assertEqual(len(optimizer.subtitle.segments), original_len - 1)

    def test_save(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.perform_all()
        optimizer.save(output_name='data/test_subtitle_optimized.json')

        with open('data/test_subtitle_optimized.json', 'r', encoding='utf-8') as f:
            optimized_subtitle = json.load(f)

        self.assertEqual(optimized_subtitle['language'], 'zh')
        self.assertEqual(len(optimized_subtitle['segments']), 8)

        os.remove('data/test_subtitle_optimized.json')
