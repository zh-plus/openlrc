#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import pytest

from openlrc.prompter import BaseTranslatePrompter


@pytest.fixture
def prompter():
    return BaseTranslatePrompter('ja', 'zh-cn', 'movie', 'Title', 'Synopsis')


def test_user_prompt(prompter):
    user_input = '''#1
Original>
変わりゆく時代において、
Translation>'''
    expected_output = '''<title>Title</title>
<synopsis>Synopsis</synopsis>
<context>
<scene>test scene content</scene>
<chunk> Chunk 1: test chunk1 summary
Chunk 2: test chunk2 summary </chunk>
</context>
<chunk_id> Scene 1 Chunk 1 <chunk_id>

Please translate these subtitles for movie named Title from Japanese to Mandarin Chinese.

#1
Original>
変わりゆく時代において、
Translation>

<summary/>
<scene/>'''
    assert prompter.user(1, user_input, ['test chunk1 summary', 'test chunk2 summary'],
                         'test scene content') == expected_output


def test_format_texts():
    texts = [(1, '変わりゆく時代において、'), (2, '生き残る秘訣は、進化し続けることです。')]
    expected_output = '#1\nOriginal>\n変わりゆく時代において、\nTranslation>\n\n#2\nOriginal>\n生き残る秘訣は、進化し続けることです。\nTranslation>\n'
    assert BaseTranslatePrompter.format_texts(texts) == expected_output


def test_check_format():
    prompter = BaseTranslatePrompter('ja', 'zh-cn', 'movie', 'Title', 'Synopsis')
    messages = []
    content = '''<title>Title</title>
<synopsis>Synopsis</synopsis>
<context>
<scene>Scene</scene>
<chunk> Chunk 1:  </chunk>
</context>
<chunk_id> Scene 1 Chunk 1 <chunk_id>

#1
Original>
変わりゆく時代において、
Translation>
#2
Original>
生き残る秘訣は、進化し続けることです。
Translation>

<summary>Summary</summary>
<scene>Scene</scene>
'''
    assert prompter.check_format(messages, content) is True
