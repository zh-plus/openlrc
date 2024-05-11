#  Copyright (C) 2024. Hao Zheng
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
        self.assertEqual(context.background, 'test background')
        self.assertEqual(context.audio_type, 'test audio type')
        self.assertEqual(context.description_map, {'test audio name': 'description'})
        self.assertIsNone(context.config_path)

    def test_init_with_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(
                'background: config background\naudio_type: config audio type\ndescription_map:\n  config: config description\n')
            config_path = Path(f.name)

        context = Context(config_path=config_path)

        self.assertEqual(context.background, 'config background')
        self.assertEqual(context.audio_type, 'config audio type')
        self.assertEqual(context.description_map, {'config': 'config description'})
        self.assertEqual(context.config_path, config_path)

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

        self.assertEqual(context.background, 'config background')
        self.assertEqual(context.audio_type, 'config audio type')
        self.assertEqual(context.description_map, {'config': 'config description'})
        self.assertEqual(context.config_path, config_path)

        config_path.unlink()

    def test_save_config(self):
        context = self.context
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            config_path = Path(f.name)

        context.config_path = config_path
        context.save_config()

        with open(config_path, 'r') as file:
            config = file.read()

        self.assertIn('background: test background', config)
        self.assertIn('audio_type: test audio type', config)
        self.assertIn('description_map:\n  test audio name: description', config)

        config_path.unlink()

    def test_get_description(self):
        context = self.context
        self.assertEqual(context.get_description('test audio name'), 'description')
        self.assertEqual(context.get_description('audio name without description'), '')

    def test_str(self):
        self.assertEqual(str(self.context),
                         'Context(background=test background, audio_type=test audio type, description_map={\'test audio name\': \'description\'})')
