#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
from difflib import get_close_matches
from pathlib import Path
from typing import Union

import yaml

from openlrc.logger import logger


class Context:
    def __init__(self, background='', description_map=None, audio_type='Anime', config_path=None):
        """
        Context(optional) for translation.

        Args:
            background (str): Providing background information for establishing context for the translation.
            description_map (dict, optional): {"name(without extension)": "description", ...}
            audio_type (str, optional): Audio type, default to Anime.
            config_path (str, optional): Path to config file.

        Raises:
            FileNotFoundError: If the config file specified by config_path does not exist.

        """
        self.config_path = None
        self.background = background
        self.audio_type = audio_type
        self.description_map = description_map if description_map else dict()

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

        if config.get('description_map'):
            self.description_map = config['description_map']

        self.config_path = config_path

    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump({
                'background': self.background,
                'audio_type': self.audio_type,
                'description_map': self.description_map,
            }, f)

    def get_description(self, audio_name):
        value = ''
        if self.description_map:
            matches = get_close_matches(audio_name, self.description_map.keys())
            if matches:
                key = matches[0]
                value = self.description_map.get(key)
                logger.info(f'Found description map: {key} -> {value}')
            else:
                logger.info(f'No description map for {audio_name} found.')

        return value

    def __str__(self):
        return f'Context(background={self.background}, audio_type={self.audio_type}, description_map={self.description_map})'
