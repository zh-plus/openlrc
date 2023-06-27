#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import tempfile
from pathlib import Path

import pytest

from openlrc.context import Context


@pytest.fixture
def context():
    return Context(background='test background', audio_type='test audio type',
                   synopsis_map={'test audio name': 'synopsis'})


def test_init(context):
    assert context.background == 'test background'
    assert context.audio_type == 'test audio type'
    assert context.synopsis_map == {'test audio name': 'synopsis'}
    assert context.config_path is None


def test_init_with_config_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(
            'background: config background\naudio_type: config audio type\nsynopsis_map:\n  config: config synopsis\n')
        config_path = Path(f.name)

    context = Context(config_path=config_path)

    assert context.background == 'config background'
    assert context.audio_type == 'config audio type'
    assert context.synopsis_map == {'config': 'config synopsis'}
    assert context.config_path == config_path

    config_path.unlink()


def test_init_with_invalid_config_file():
    with pytest.raises(FileNotFoundError):
        Context(config_path='invalid_path')


def test_load_config(context):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(
            'background: config background\naudio_type: config audio type\nsynopsis_map:\n  config: config synopsis\n')
        config_path = Path(f.name)

    context.load_config(config_path)

    assert context.background == 'config background'
    assert context.audio_type == 'config audio type'
    assert context.synopsis_map == {'config': 'config synopsis'}
    assert context.config_path == config_path

    config_path.unlink()


def test_save_config(context):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        config_path = Path(f.name)

    context.config_path = config_path
    context.save_config()

    with open(config_path, 'r') as file:
        config = file.read()

    assert 'background: test background' in config
    assert 'audio_type: test audio type' in config
    assert 'synopsis_map:\n  test audio name: synopsis' in config

    config_path.unlink()


def test_get_synopsis(context):
    assert context.get_synopsis('test audio name') == 'synopsis'
    assert context.get_synopsis('audio name without synopsis') == ''


def test_str(context):
    assert str(context) == \
           'Context(background=test background, audio_type=test audio type, synopsis_map={\'test audio name\': \'synopsis\'})'
