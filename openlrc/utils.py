import gc
import json
import time
from os.path import splitext

import audioread
import tiktoken
import torch

from openlrc.logger import logger


def check_json(json_str):
    """ Check if the json string is valid. """

    try:
        json.loads(json_str)
        return True
    except json.decoder.JSONDecodeError:
        return False


def get_audio_duration(path):
    return format_timestamp(audioread.audio_open(path).duration)


def release_memory(model):
    gc.collect()
    torch.cuda.empty_cache()
    del model


def get_text_token_number(text, model="gpt-3.5-turbo"):
    encoder = tiktoken.encoding_for_model(model)

    return len(encoder.encode(text))


def get_messages_token_number(messages, model="gpt-3.5-turbo"):
    total = sum([get_text_token_number(element['content'], model=model) for element in messages])

    return total


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


def parse_timestamp(time_stamp, fmt='lrc'):
    if fmt == 'lrc':
        minutes, seconds = time_stamp.split(':')
        seconds, hundredths_of_sec = seconds.split('.')
        return int(minutes) * 60 + int(seconds) + int(hundredths_of_sec) / 100.0
    elif fmt == 'srt':
        hours, minutes, seconds = time_stamp.split(':')
        seconds, milliseconds = seconds.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")


def format_timestamp(seconds: float, fmt='lrc'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    seconds = milliseconds // 1000
    milliseconds %= 1000

    if fmt == 'lrc':
        # [<minutes>:<seconds>.<hundredths of a second>] for lrc
        return f"{minutes:02d}:{seconds:02d}.{milliseconds // 10:02d}"
    elif fmt == 'srt':
        # [<hours>:<minutes>:<seconds>,<milliseconds>] for srt
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")
