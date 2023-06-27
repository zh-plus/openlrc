#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.
from difflib import get_close_matches
from pathlib import Path
from typing import Union

import yaml

from openlrc.logger import logger


class Context:
    def __init__(self, background='', synopsis_map=None, audio_type='Anime', config_path=None):
        """
        Context(optional) for translation.

        :param background: Providing background information for establishing context for the translation.
        :param synopsis_map: {"name(without extension)": "synopsis", ...}
        :param audio_type: Audio type, default to Anime.
        :param config_path: Path to config file.
        """
        self.config_path = None
        self.background = background
        self.audio_type = audio_type
        self.synopsis_map = synopsis_map if synopsis_map else dict()

        # if config_path exist, load yaml file
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                self.load_config(config_path)
            else:
                raise FileNotFoundError(f'Config file {config_path} not found.')

    def load_config(self, config_path: Union[str, Path]):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file {config_path} not found.')

        with open(config_path, 'r', encoding='utf-8') as f:
            config: dict = yaml.safe_load(f)

        if config.get('background'):
            self.background = config['background']

        if config.get('audio_type'):
            self.audio_type = config['audio_type']

        if config.get('synopsis_map'):
            self.synopsis_map = config['synopsis_map']

        self.config_path = config_path

    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump({
                'background': self.background,
                'audio_type': self.audio_type,
                'synopsis_map': self.synopsis_map,
            }, f)

    def get_synopsis(self, audio_name):
        value = ''
        if self.synopsis_map:
            matches = get_close_matches(audio_name, self.synopsis_map.keys())
            if matches:
                key = matches[0]
                value = self.synopsis_map.get(key)
                logger.info(f'Found synopsis map: {key} -> {value}')
            else:
                logger.info(f'No synopsis map for {audio_name} found.')

        return value

    def __str__(self):
        return f'Context(background={self.background}, audio_type={self.audio_type}, synopsis_map={self.synopsis_map})'
