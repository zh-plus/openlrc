#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import os
import unittest
from pathlib import Path

from openlrc.config import SubtitleOptimizationMode
from openlrc.opt import SubtitleOptimizer
from openlrc.subtitle import Subtitle

TEST_DATA_DIR = Path(__file__).parent / "data"


class TestSubtitleOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitle = Subtitle.from_json(TEST_DATA_DIR / "test_valid_subtitle.json")

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
        self.assertEqual(optimizer.subtitle.segments[2].text, "好好...")

    def test_cut_long(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.cut_long(max_length=2)
        self.assertEqual(optimizer.subtitle.segments[4].text, "这太")

    def test_traditional2mandarin(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.traditional2mandarin()
        self.assertEqual(optimizer.subtitle.segments[5].text, "繁体的字")

    def test_punctuation_optimization(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.punctuation_optimization()
        self.assertEqual(optimizer.subtitle.segments[0].text, "你好，你好……你好！你好。")

    def test_punctuation_optimization_with_dots(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.punctuation_optimization()
        self.assertEqual(optimizer.subtitle.segments[9].text, "1. 测试。这是1.2节。")

    def test_remove_unk(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_unk()
        self.assertEqual(optimizer.subtitle.segments[6].text, "unk")

    def test_remove_empty(self):
        subtitle = self.subtitle
        subtitle.segments[0].text = ""
        original_len = len(subtitle)
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.remove_empty()
        self.assertEqual(len(optimizer.subtitle.segments), original_len - 1)

    def test_save(self):
        subtitle = self.subtitle
        optimizer = SubtitleOptimizer(subtitle)
        optimizer.perform_all()
        output_path = TEST_DATA_DIR / "test_subtitle_optimized.json"
        optimizer.save(output_name=output_path)

        with open(output_path, encoding="utf-8") as f:
            optimized_subtitle = json.load(f)

        self.assertEqual(optimized_subtitle["language"], "zh")
        self.assertEqual(len(optimized_subtitle["segments"]), 8)

        os.remove(output_path)

    def test_relaxed_source_preserves_segments_text_and_timing(self):
        long_repeat = "verylong" * 60
        subtitle = Subtitle(
            language="en",
            filename="relaxed.json",
            segments=[
                {"start": 0.0, "end": 0.3, "text": "  One  "},
                {"start": 0.4, "end": 0.7, "text": "One"},
                {"start": 0.8, "end": 1.1, "text": "haaaaa"},
                {"start": 1.2, "end": 2.0, "text": long_repeat},
                {"start": 2.1, "end": 2.4, "text": "<unk>"},
                {"start": 2.5, "end": 2.8, "text": "   "},
            ],
        )
        timings = [(segment.start, segment.end) for segment in subtitle.segments]

        optimizer = SubtitleOptimizer(subtitle)
        optimizer.perform_all(mode=SubtitleOptimizationMode.RELAXED, stage="source", extend_time=True)

        self.assertEqual(len(optimizer.subtitle.segments), 6)
        self.assertEqual(optimizer.subtitle.texts, ["One", "One", "haaaaa", long_repeat, "<unk>", ""])
        self.assertEqual([(segment.start, segment.end) for segment in optimizer.subtitle.segments], timings)

    def test_relaxed_target_only_normalizes_target_text(self):
        subtitle = Subtitle(
            language="zh-cn", filename="target.json", segments=[{"start": 0.0, "end": 0.3, "text": "  繁體字幕!  "}]
        )

        optimizer = SubtitleOptimizer(subtitle)
        optimizer.perform_all(mode="relaxed", stage="target", extend_time=True)

        self.assertEqual(len(optimizer.subtitle.segments), 1)
        self.assertEqual(optimizer.subtitle.texts, ["繁体字幕！"])
        self.assertEqual((optimizer.subtitle.segments[0].start, optimizer.subtitle.segments[0].end), (0.0, 0.3))
