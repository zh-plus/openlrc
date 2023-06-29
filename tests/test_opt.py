#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import json
import os

import pytest

from openlrc.opt import SubtitleOptimizer
from openlrc.subtitle import Subtitle


@pytest.fixture
def subtitle():
    return Subtitle.from_json('data/test_subtitle.json')


def test_merge_same(subtitle):
    original_len = len(subtitle)
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.merge_same()
    assert len(optimizer.subtitle.segments) == original_len - 1


def test_merge_short(subtitle):
    original_len = len(subtitle)
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.merge_short()
    assert len(optimizer.subtitle.segments) == original_len - 1


def test_merge_repeat(subtitle):
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.merge_repeat()
    assert optimizer.subtitle.segments[2].text == '好好...'


def test_cut_long(subtitle):
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.cut_long(keep=2)
    assert optimizer.subtitle.segments[4].text == '这太(Cut to 2)'


def test_traditional2mandarin(subtitle):
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.traditional2mandarin()
    assert optimizer.subtitle.segments[5].text == '繁体的字'


def test_remove_unk(subtitle):
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.remove_unk()
    assert optimizer.subtitle.segments[6].text == 'unk'


def test_remove_empty(subtitle):
    original_len = len(subtitle)
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.remove_empty()
    assert len(optimizer.subtitle.segments) == original_len - 1


def test_save(subtitle):
    optimizer = SubtitleOptimizer(subtitle)
    optimizer.perform_all()
    optimizer.save(output_name='test_subtitle_optimized.json')

    with open('test_subtitle_optimized.json', 'r', encoding='utf-8') as f:
        optimized_subtitle = json.load(f)

    assert optimized_subtitle['language'] == 'cn'
    assert optimized_subtitle['generator'] == 'test'
    assert len(optimized_subtitle['segments']) == 5

    os.remove('test_subtitle_optimized.json')
