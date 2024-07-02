#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import unittest

from openlrc.context import TranslateInfo
from openlrc.prompter import ChunkedTranslatePrompter, Prompter

formatted_user_input = '''Translation guidelines from context reviewer:
This is a guidline.

Previews summaries:
Chunk 1: test chunk1 summary
Chunk 2: test chunk2 summary

<chunk_id> Scene 1 Chunk 1 <chunk_id>

Please translate these subtitles for movie from Japanese to Chinese (China).

#1
Original>
変わりゆく時代において、
Translation>

#2
Original>
生き残る秘訣は、進化し続けることです。
Translation>
<summary></summary>
<scene></scene>'''


class TestPrompter(unittest.TestCase):
    def setUp(self) -> None:
        context = TranslateInfo(title='Title', audio_type='movie')
        self.prompter = ChunkedTranslatePrompter('ja', 'zh-cn', context)
        self.formatted_user_input = formatted_user_input

    def test_user_prompt(self):
        user_input = '''#1
Original>
変わりゆく時代において、
Translation>

#2
Original>
生き残る秘訣は、進化し続けることです。
Translation>'''
        self.assertEqual(
            self.prompter.user(1, user_input, ['test chunk1 summary', 'test chunk2 summary'],
                               guideline='This is a guidline.'),
            self.formatted_user_input
        )

    def test_format_texts(self):
        texts = [(1, '変わりゆく時代において、'), (2, '生き残る秘訣は、進化し続けることです。')]
        expected_output = '#1\nOriginal>\n変わりゆく時代において、\nTranslation>\n\n#2\nOriginal>\n' \
                          '生き残る秘訣は、進化し続けることです。\nTranslation>\n'
        self.assertEqual(ChunkedTranslatePrompter.format_texts(texts), expected_output)

    def test_check_format(self):
        messages = [{'role': 'system', 'content': 'system content'},
                    {'role': 'user', 'content': formatted_user_input}]
        content = '''<title>Title</title>
<context>
<scene>Scene</scene>
<chunk> Chunk 1:  </chunk>
</context>
<chunk_id> Scene 1 Chunk 1 <chunk_id>

#1
Original>
変わりゆく時代において、
Translation>
在不断变化的时代里，

#2
Original>
生き残る秘訣は、進化し続けることです。
Translation>
生存的秘诀是不断进化。

<summary>Summary</summary>
<scene>Scene</scene>
'''
        self.assertTrue(self.prompter.check_format(formatted_user_input, content))

    def test_default_check_format(self):
        class TMPPrompter(Prompter):
            pass

        self.assertTrue(TMPPrompter().check_format('content', 'content'))
