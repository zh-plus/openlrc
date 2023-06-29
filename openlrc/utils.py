#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

import gc
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import audioread
import ffmpeg
import tiktoken
import torch
from langcodes import Language
from lingua import LanguageDetectorBuilder

from openlrc.exceptions import FfmpegException
from openlrc.logger import logger


def extract_audio(path: Path) -> Path:
    """
    Extract audio from video.
    :return: Audio path
    """
    file_type = get_file_type(path)
    if file_type == 'audio':
        return path

    probe = ffmpeg.probe(path)
    audio_streams = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    sample_rate = audio_streams['sample_rate']
    logger.info(f'File {path}: Audio sample rate: {sample_rate}')

    audio, err = (
        ffmpeg.input(path).
        output("pipe:", format='wav', acodec='pcm_s16le', ar=sample_rate, loglevel='quiet').
        run(capture_stdout=True)
    )

    if err:
        raise RuntimeError(f'ffmpeg error: {err}')

    audio_path = path.with_suffix('.wav')
    with open(audio_path, 'wb') as f:
        f.write(audio)

    return audio_path


def get_file_type(path: Path) -> str:
    try:
        video_stream = ffmpeg.probe(path, select_streams='v')['streams']
    except Exception as e:
        raise FfmpegException(f'ffmpeg error: {e}')

    return ['audio', 'video'][len(video_stream) > 0]


def get_audio_duration(path: str) -> float:
    return audioread.audio_open(path).duration


def release_memory(model: torch.nn.Module) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    del model


def get_text_token_number(text: str, model: str = "gpt-3.5-turbo") -> int:
    tokens = tiktoken.encoding_for_model(model).encode(text)

    return len(tokens)


def get_messages_token_number(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
    total = sum([get_text_token_number(element['content'], model=model) for element in messages])

    return total


def extend_filename(filename: Path, extend: str) -> Path:
    """Extend a filename with some string."""
    return filename.with_stem(filename.stem + extend)


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


def parse_timestamp(time_stamp: str, fmt: str) -> float:
    """
    Parse a timestamp from a subtitle file.

    :param time_stamp: Timestamp to parse.
    :param fmt: Format of `time_stamp`. Supported values are:
        1. 'lrc' for LRC format, e.g. '1:23.45'
        2. 'srt' for SRT format, e.g. '01:23:45,678'
    :return: Time stamp in seconds.
    """

    if fmt == 'lrc':
        if not re.match(r'^\d+:\d+\.\d+$', time_stamp):
            raise ValueError(f"Invalid timestamp format for LRC: {time_stamp}")
        minutes, seconds = time_stamp.split(':')
        seconds, hundredths_of_sec = seconds.split('.')
        return int(minutes) * 60 + int(seconds) + int(hundredths_of_sec) / 100.0
    elif fmt == 'srt':
        if not re.match(r'^\d+:\d+:\d+,\d+$', time_stamp):
            raise ValueError(f"Invalid timestamp format for SRT: {time_stamp}")
        hours, minutes, seconds = time_stamp.split(':')
        seconds, milliseconds = seconds.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")


def format_timestamp(seconds: float, fmt: str = 'lrc') -> str:
    """
    Convert a timestamp in seconds into a string.

    :param seconds: Timestamp in seconds.
    :param fmt: Format of the output string. Supported values are:
        1. 'lrc' for LRC format, e.g. '1:23.45'
        2. 'srt' for SRT format, e.g. '01:23:45,678'
    :return: A string representation of the timestamp in the specified format.
    """

    # Ensure that the timestamp is non-negative.
    assert seconds >= 0, "non-negative timestamp expected"

    # Convert seconds into milliseconds.
    milliseconds = round(seconds * 1000.0)

    # Extract hours, minutes, seconds, and milliseconds from milliseconds.
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    # Return the timestamp in the specified format.
    if fmt == 'lrc':
        return f"{minutes:02d}:{seconds:02d}.{milliseconds // 10:02d}"
    elif fmt == 'srt':
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    else:
        raise ValueError(f"Unsupported timestamp format: {fmt}")


def detect_lang(text):
    detector = LanguageDetectorBuilder.from_all_languages().build()
    name = detector.detect_language_of(' '.join(text)).name.lower()
    lang_code = Language.find(name).language
    return lang_code
