#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

import torch

from openlrc.utils import format_timestamp, parse_timestamp, get_text_token_number, get_messages_token_number, \
    extend_filename, release_memory, extract_audio, get_file_type, normalize


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_file = Path('data/test_audio.wav')
        self.video_file = Path('data/test_video.mp4')
        self.unsupported = Path('data/unsupported_file.xyz')

    def tearDown(self) -> None:
        self.video_file.with_suffix('.wav').unlink(missing_ok=True)

    def test_extract_audio(self):
        # Test extracting audio from a video file
        extracted_audio_file = extract_audio(self.video_file)
        assert extracted_audio_file == self.video_file.with_suffix('.wav')

        # Test extracting audio from an audio file
        extracted_audio_file = extract_audio(self.audio_file)
        assert extracted_audio_file == self.audio_file

        # Test extracting audio from an unsupported file type
        with self.assertRaises(RuntimeError):
            extract_audio(self.unsupported)

    def test_get_file_type(self):
        # Test getting the file type of video file
        file_type = get_file_type(self.video_file)
        assert file_type == 'video'

        # Test getting the file type of audio file
        file_type = get_file_type(self.audio_file)
        assert file_type == 'audio'

        # Test getting the file type of unsupported file type
        with self.assertRaises(RuntimeError):
            get_file_type(self.unsupported)

    def test_lrc_format(self):
        assert format_timestamp(1.2345, 'lrc') == '00:01.23'
        assert format_timestamp(61.2345, 'lrc') == '01:01.23'
        assert format_timestamp(3661.2345, 'lrc') == '01:01.23'

        assert parse_timestamp('1:23.45', 'lrc') == 83.45
        assert parse_timestamp('0:00.01', 'lrc') == 0.01
        assert parse_timestamp('10:00.00', 'lrc') == 600.0

    def test_srt_format(self):
        assert format_timestamp(1.2345, 'srt') == '00:00:01,234'
        assert format_timestamp(61.2345, 'srt') == '00:01:01,234'
        assert format_timestamp(3661.2345, 'srt') == '01:01:01,234'

        assert parse_timestamp('01:23:45,678', 'srt') == 5025.678
        assert parse_timestamp('00:00:01,000', 'srt') == 1.0
        assert parse_timestamp('01:00:00,000', 'srt') == 3600.0

    def test_negative_timestamp(self):
        with self.assertRaises(AssertionError):
            format_timestamp(-1.2345, 'lrc')

        with self.assertRaises(ValueError):
            parse_timestamp('-1:23.45', 'lrc')

        with self.assertRaises(AssertionError):
            format_timestamp(-1.2345, 'srt')

        with self.assertRaises(ValueError):
            parse_timestamp('-01:23:45,678', 'srt')

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            format_timestamp(1.2345, 'invalid')

        with self.assertRaises(ValueError):
            parse_timestamp('1:23.45', 'invalid')

    def test_get_text_token_number(self):
        assert get_text_token_number('Hello, world!') == 4
        assert get_text_token_number('This is a longer sentence.') == 6
        assert get_text_token_number('') == 0

    def test_get_messages_token_number(self):
        messages = [
            {'content': 'Hello, world!'},
            {'content': 'This is a longer sentence.'},
            {'content': ''}
        ]
        assert get_messages_token_number(messages) == 10

        messages = [
            {'content': 'Hello, world!'},
            {'content': 'This is a longer sentence.'},
            {'content': ''},
            {'content': 'Another message.'}
        ]
        assert get_messages_token_number(messages) == 13

    def test_extend_filename(self):
        assert extend_filename(Path('file.txt'), '_new') == Path('file_new.txt')
        assert extend_filename(Path('file.txt'), '') == Path('file.txt')

    def test_release_memory(self):
        model = torch.nn.Module()
        if torch.cuda.is_available():
            model.cuda()
        release_memory(model)
        assert torch.cuda.memory_allocated() == 0

    def test_normalize(self):
        # Full width to half width & lower case
        alphabet_fw = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        alphabet_hw = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz'
        assert normalize(alphabet_fw) == alphabet_hw

        number_fw = '０１２３４５６７８９'
        number_hw = '0123456789'
        assert normalize(number_fw) == number_hw

        sign_fw = '！＃＄％＆（）＊＋，－．／：；＜＝＞？＠［］＾＿｀｛｜｝”’￥～'
        sign_hw = '!#$%&()*+,-./:;<=>?@[]^_`{|}"\'¥~'
        assert normalize(sign_fw) == sign_hw

        kana_fw = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴァィゥェォッャュョ・ー、。・「」'
        kana_hw = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝｶﾞｷﾞｸﾞｹﾞｺﾞｻﾞｼﾞｽﾞｾﾞｿﾞﾀﾞﾁﾞﾂﾞﾃﾞﾄﾞﾊﾞﾋﾞﾌﾞﾍﾞﾎﾞﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟｳﾞｧｨｩｪｫｯｬｭｮ･ｰ､｡･｢｣'
        assert normalize(kana_fw) == kana_hw

        space_fw = '　'  # Full-width space (U+3000)
        space_hw = ' '  # Half-width space (U+0020)
        assert normalize(space_fw) == space_hw
