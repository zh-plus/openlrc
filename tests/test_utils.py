#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import pytest
import torch

from openlrc.utils import format_timestamp, parse_timestamp, get_text_token_number, get_messages_token_number, \
    change_ext, extend_filename, release_memory


def test_lrc_format():
    assert format_timestamp(1.2345, 'lrc') == '00:01.23'
    assert format_timestamp(61.2345, 'lrc') == '01:01.23'
    assert format_timestamp(3661.2345, 'lrc') == '01:01.23'

    assert parse_timestamp('1:23.45', 'lrc') == 83.45
    assert parse_timestamp('0:00.01', 'lrc') == 0.01
    assert parse_timestamp('10:00.00', 'lrc') == 600.0


def test_srt_format():
    assert format_timestamp(1.2345, 'srt') == '00:00:01,234'
    assert format_timestamp(61.2345, 'srt') == '00:01:01,234'
    assert format_timestamp(3661.2345, 'srt') == '01:01:01,234'

    assert parse_timestamp('01:23:45,678', 'srt') == 5025.678
    assert parse_timestamp('00:00:01,000', 'srt') == 1.0
    assert parse_timestamp('01:00:00,000', 'srt') == 3600.0


def test_negative_timestamp():
    with pytest.raises(AssertionError):
        format_timestamp(-1.2345, 'lrc')

    with pytest.raises(ValueError):
        parse_timestamp('-1:23.45', 'lrc')

    with pytest.raises(AssertionError):
        format_timestamp(-1.2345, 'srt')

    with pytest.raises(ValueError):
        parse_timestamp('-01:23:45,678', 'srt')


def test_invalid_format():
    with pytest.raises(ValueError):
        format_timestamp(1.2345, 'invalid')

    with pytest.raises(ValueError):
        parse_timestamp('1:23.45', 'invalid')


def test_get_text_token_number():
    assert get_text_token_number('Hello, world!') == 4
    assert get_text_token_number('This is a longer sentence.') == 6
    assert get_text_token_number('') == 0


def test_get_messages_token_number():
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


def test_change_ext():
    assert change_ext('file.txt', 'csv') == 'file.csv'
    assert change_ext('file.txt', '') == 'file.'
    assert change_ext('file', 'csv') == 'file.csv'


def test_extend_filename():
    assert extend_filename('file.txt', '_new') == 'file_new.txt'
    assert extend_filename('file', '_new') == 'file_new'
    assert extend_filename('file.txt', '') == 'file.txt'


def test_release_memory():
    model = torch.nn.Module()
    if torch.cuda.is_available():
        model.cuda()
    release_memory(model)
    assert torch.cuda.memory_allocated() == 0
