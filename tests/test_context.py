#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import tempfile
import unittest
from pathlib import Path

from openlrc.context import Context


class TestContext(unittest.TestCase):
    def setUp(self) -> None:
        self.context = Context(background='test background', audio_type='test audio type',
                               description_map={'test audio name': 'description'})

    def test_init(self):
        context = self.context
        assert context.background == 'test background'
        assert context.audio_type == 'test audio type'
        assert context.description_map == {'test audio name': 'description'}
        assert context.config_path is None

    def test_init_with_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(
                'background: config background\naudio_type: config audio type\ndescription_map:\n  config: config description\n')
            config_path = Path(f.name)

        context = Context(config_path=config_path)

        assert context.background == 'config background'
        assert context.audio_type == 'config audio type'
        assert context.description_map == {'config': 'config description'}
        assert context.config_path == config_path

        config_path.unlink()

    def test_init_with_invalid_config_file(self):
        with self.assertRaises(FileNotFoundError):
            Context(config_path='invalid_path')

    def test_load_config(self):
        context = self.context
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(
                'background: config background\naudio_type: config audio type\ndescription_map:\n  config: config description\n')
            config_path = Path(f.name)

        context.load_config(config_path)

        assert context.background == 'config background'
        assert context.audio_type == 'config audio type'
        assert context.description_map == {'config': 'config description'}
        assert context.config_path == config_path

        config_path.unlink()

    def test_save_config(self):
        context = self.context
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            config_path = Path(f.name)

        context.config_path = config_path
        context.save_config()

        with open(config_path, 'r') as file:
            config = file.read()

        assert 'background: test background' in config
        assert 'audio_type: test audio type' in config
        assert 'description_map:\n  test audio name: description' in config

        config_path.unlink()

    def test_get_description(self):
        context = self.context
        assert context.get_description('test audio name') == 'description'
        assert context.get_description('audio name without description') == ''

    def test_str(self):
        assert str(self.context) == \
               'Context(background=test background, audio_type=test audio type, description_map={\'test audio name\': \'description\'})'
