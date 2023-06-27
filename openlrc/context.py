#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import os

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
        self.config_path = config_path
        self.background = background
        self.audio_type = audio_type
        self.synopsis_map = synopsis_map if synopsis_map else dict()

        # if config_path exist, load yaml file, else create a new one
        if config_path and os.path.exists(config_path):
            self.load_config(self.config_path)

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file {config_path} not found.')

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, yaml.CLoader)

        if config['background']:
            self.background = config['background']

        if config['audio_type']:
            self.audio_type = config['audio_type']

        if config['synopsis_map']:
            self.synopsis_map = config['synopsis_map']

    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump({
                'background': self.background,
                'audio_type': self.audio_type,
                'synopsis_map': self.synopsis_map,
            }, f)

    def get_synopsis(self, audio_name):
        if self.synopsis_map and any(k in audio_name for k in self.synopsis_map.keys()):
            for k, v in self.synopsis_map.items():
                if k in audio_name:
                    logger.info(f'Found synopsis map: {k} -> {v}')
                    return v

        return ''

    def __str__(self):
        return f'Context(background={self.background}, audio_type={self.audio_type}, synopsis_map={self.synopsis_map})'
