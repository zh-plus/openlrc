import json
import time
from os.path import splitext

import tiktoken

from openlrc.logger import logger


def json2dict(json_str):
    """ Convert json string to python dict. """

    try:
        result = json.loads(json_str)
        return result
    except json.decoder.JSONDecodeError as e:
        logger.warning(f'Fail to convert into json: \n {json_str}\n')

    # Try to fix the json string, only keep the content from first '{' to last '}'
    fixed_json_str1 = json_str[json_str.find('{'):json_str.rfind('}') + 1]
    logger.warning(
        f'Trying to fix the json string by keep only "{{content}}": \n {json_str}\n Into: \n {fixed_json_str1}\n')
    try:
        result = json.loads(fixed_json_str1)
        return result
    except json.decoder.JSONDecodeError:
        logger.warning(f'Failed to convert into json: \n {json_str}\n')

    # Try to replace chinese "，" with english ","
    fixed_json_str2 = fixed_json_str1.replace('，', ',')
    logger.warning(
        f'Trying to fix the json string by replace chinese quote with eng quote: \n {fixed_json_str1}\n Into: \n {fixed_json_str2}\n'
    )
    try:
        result = json.loads(fixed_json_str2)
        return result
    except json.decoder.JSONDecodeError:
        logger.warning(f'Failed to convert into json: \n {fixed_json_str2}\n')

    # The content after last found " should be "}]"
    fixed_json_str3 = fixed_json_str2[:fixed_json_str2.rfind('"') + 1] + ']}'
    try:
        result = json.loads(fixed_json_str3)
        return result
    except json.decoder.JSONDecodeError as e:
        logger.error(f'Failed to convert returned content into json: \n {fixed_json_str3}\n\n')

        # Save the json string to file
        with open('test_return.json', 'w', encoding='utf-8') as f:
            f.write(fixed_json_str3)

        logger.info(f'The json file is saved to test_return.json')

        raise e


def get_token_number(text, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)

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
