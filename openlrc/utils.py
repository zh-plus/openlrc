import time
import json

from os.path import splitext
from openlrc.logger import logger

import tiktoken


def json2dict(json_str):
    """ Convert json string to python dict. """

    try:
        result = json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        logger.warning(f'Error: Trying to convert into json: \n {json_str}\n')

        # Try to fix the json string, only keep the content from first '{' to last '}'
        result = json_str[json_str.find('{'):json_str.rfind('}') + 1]
        logger.warning(f'Error: Trying to fix the json string: \n {result}\n Into: \n {result}\n')
        try:
            result = json.loads(result)
        except json.decoder.JSONDecodeError:
            logger.error(f'Error: Failed to convert into json: \n {json_str}\n')
            raise e

    return result


def get_token_number(text):
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    return len(encoder.encode(text))


def change_ext(filename, ext):
    return f'{splitext(filename)[0]}.{ext}'


def extend_filename(filename, extend):
    name, ext = splitext(filename)
    return f'{name}{extend}{ext}'


class Timer:
    def __init__(self, task=""):
        self._start = None
        self._stop = None
        self.task = task

    def start(self):
        if self.task:
            logger.info(f'Start {self.task}')
        self._start = time.perf_counter()

    def stop(self):
        self._stop = time.perf_counter()
        logger.info(f'{self.task} Elapsed: {self.elapsed:.2f}s')

    @property
    def elapsed(self):
        if self._start is None:
            raise RuntimeError("Timer not started")
        if self._stop is None:
            raise RuntimeError("Timer not stopped")
        return self._stop - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
