#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import unittest

from openlrc.subtitle import Subtitle


class TestSubtitle(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitle = Subtitle.from_json('data/test_valid_subtitle.json')

    def check_content(self, subtitle, length=8):
        self.assertEqual(subtitle.lang, 'zh')
        self.assertEqual(len(subtitle), length)
        self.assertEqual(subtitle.segments[0].start, 0.0)
        self.assertEqual(subtitle.segments[0].end, 3.0)
        self.assertEqual(subtitle.segments[0].text, '你好')
        self.assertEqual(subtitle.segments[2].start, 6.0)
        self.assertEqual(subtitle.segments[2].end, 9.0)
        self.assertEqual(subtitle.segments[2].text, '好好好好好好好好好好好好好好好好好好好好好好好好')

    #  Tests that a valid JSON subtitle file can be loaded
    def test_load_valid_json(self):
        self.check_content(self.subtitle)

    #  Tests that a valid LRC subtitle file can be loaded
    def test_load_lrc(self):
        subtitle = Subtitle.from_file('data/test_subtitle.lrc')
        self.check_content(subtitle, length=7)

    #  Tests that a subtitle can be saved to a valid JSON file
    def test_save_json(self):
        subtitle = self.subtitle
        subtitle.save('data/saved.json')
        loaded_subtitle = Subtitle.from_file('data/saved.json')
        self.check_content(loaded_subtitle)
        loaded_subtitle.filename.unlink()

    #  Tests that a subtitle can be saved to a valid LRC file
    def test_save_to_lrc(self):
        subtitle = self.subtitle
        subtitle.to_lrc()
        loaded_subtitle = Subtitle.from_file(subtitle.filename.with_suffix('.lrc'))
        self.check_content(loaded_subtitle, length=7)
        loaded_subtitle.filename.unlink()

    def test_save_to_srt(self):
        subtitle = self.subtitle
        subtitle.to_srt()
        loaded_subtitle = Subtitle.from_file(subtitle.filename.with_suffix('.srt'))
        self.check_content(loaded_subtitle, length=8)
        loaded_subtitle.filename.unlink()

    #  Tests that the length of the subtitle can be retrieved
    def test_get_length(self):
        subtitle = self.subtitle
        self.assertEqual(len(subtitle), 8)

    #  Tests that the texts of the subtitle can be retrieved
    def test_get_texts(self):
        subtitle = self.subtitle
        self.assertEqual(subtitle.texts, ['你好', '你好', '好好好好好好好好好好好好好好好好好好好好好好好好', '',
                                          '这太长打发螺丝扣搭街坊拉克斯酱豆腐垃圾啊阿里山扩大飞机拉克斯基的flak涉及到了反馈啊螺丝扣搭街坊拉啊手动阀手动阀阿斯顿发射点发射点发生发射点发射点发萨看见对方这太长打发螺丝扣搭街坊拉克斯酱豆腐垃圾啊阿里山扩大飞机拉克斯基的flak涉及到了反馈啊螺丝扣搭街坊拉啊手动阀手动阀阿斯顿发射点发射点发生发射点发射点发萨看见对方这太长打发螺丝扣搭街坊拉克斯酱豆腐垃圾啊阿里山扩大飞机拉克斯基的flak涉及到了反馈啊螺丝扣搭街坊拉啊手动阀手动阀阿斯顿发射点发射点发生发射点发射点发萨看见对方',
                                          '繁體的字', '<unk>unk<unk>', '123'])

    #  Tests that texts can be set with a list of different length than the subtitle
    def test_set_texts_edge_case(self):
        subtitle = self.subtitle
        with self.assertRaises(AssertionError):
            subtitle.set_texts(['Hello'])

    #  Tests that texts can be set with a list containing empty strings
    def test_set_texts_edge_case_2(self):
        subtitle = self.subtitle
        subtitle.set_texts([''] * 8, lang='en')
        self.assertEqual(subtitle.lang, 'en')
        self.assertEqual(subtitle.segments[0].text, '')
        self.assertEqual(subtitle.segments[1].text, '')
