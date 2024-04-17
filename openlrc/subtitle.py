#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import json
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Union, Dict

from openlrc.logger import logger
from openlrc.utils import format_timestamp, parse_timestamp, detect_lang


@dataclass
class Element:
    """
    Save a LRC format element.
    """
    start: float
    end: Union[float, None]
    text: str

    @property
    def duration(self):
        if self.end:
            return self.end - self.start
        else:
            return sys.maxsize  # Fake int infinity

    def to_json(self):
        return {'start': self.start, 'end': self.end, 'text': self.text}


class Subtitle:
    """
    Save a sequence of Element, and meta data.
    """

    def __init__(self, language: str, segments: List[Dict], filename: Union[str, Path]):
        self.lang = language
        self.segments: List[Element] = [Element(**seg) for seg in segments]
        self.filename = Path(filename)

    @staticmethod
    def from_json(filename):
        with open(filename, encoding='utf-8') as f:
            content = json.loads(f.read())
        return Subtitle(filename=filename, **content)

    @staticmethod
    def from_file(filename):
        filename = Path(filename)
        suffix = filename.suffix
        if suffix == '.json':
            return Subtitle.from_json(filename)
        elif suffix == '.lrc':
            return Subtitle.from_lrc(filename)
        elif suffix == '.srt':
            return Subtitle.from_srt(filename)

    def __len__(self):
        return len(self.segments)

    @property
    def texts(self):
        return [e.text for e in self.segments]

    def set_texts(self, texts, lang=None):
        # Check length
        assert len(texts) == len(self.segments)

        for i, text in enumerate(texts):
            self.segments[i].text = text

        if lang:
            self.lang = lang

    def save(self, filename: Union[str, Path], update_name=False):
        results = {
            'language': self.lang,
            'segments': [seg.to_json() for seg in self.segments]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        if update_name:
            self.filename = filename

        return filename

    def exists(self):
        """
        Check if the .lrc/.srt/.json file exist.
        :return:
        """
        lrc_path = self.filename.with_suffix('.lrc')
        srt_path = self.filename.with_suffix('.srt')

        return lrc_path.exists() or srt_path.exists() or self.filename.exists()

    def to_lrc(self):
        # If duration larger than 1 hour, use srt file instead
        if self.segments[-1].end >= 3600:
            logger.warning('Duration larger than 1 hour, use srt file instead')
            self.to_srt()
            return self.filename.with_suffix('.srt')

        lrc_path = self.filename.with_suffix('.lrc')
        fmt = partial(format_timestamp, fmt='lrc')
        with open(lrc_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments):
                print(
                    f'[{fmt(segment.start)}] {segment.text}',
                    file=f,
                    flush=True,
                )
                if i == len(self.segments) - 1 or segment.end != self.segments[i + 1].start:
                    print(f'[{fmt(segment.end)}]', file=f, flush=True)

        logger.info(f'File saved to {lrc_path}')

        return lrc_path

    def to_srt(self):
        srt_path = self.filename.with_suffix('.srt')
        fmt = partial(format_timestamp, fmt='srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments, start=1):
                print(f'{i}\n'
                      f'{fmt(segment.start)} --> {fmt(segment.end)}\n'
                      f'{segment.text}\n', file=f, flush=True)

        logger.info(f'File saved to {srt_path}')

        return srt_path

    @classmethod
    def from_lrc(cls, filename):
        filename = Path(filename)
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()

        # Remove comments
        lines = [line for line in lines if not line.startswith('#')]
        # Only include valid lines
        lines = [line for line in lines if line.startswith('[')]

        segments = []
        for i, line in enumerate(lines):
            # get time stamp
            start_str, text = re.search(r'\[(\d+:\d+(?:\.\d+)?)](.*)', line).group(1, 2)
            start = parse_timestamp(start_str, fmt='lrc')

            if i != len(lines) - 1:
                end_str = re.search(r'\[(\d+:\d+(?:\.\d+)?)]', lines[i + 1]).group(1)
                end = parse_timestamp(end_str, fmt='lrc')
            else:
                end = None

            if not text.strip():
                continue

            segments.append({'start': start, 'end': end, 'text': text.strip()})

        lang = detect_lang(' '.join(segment['text'] for segment in segments[:10]))

        return cls(language=lang, segments=segments, filename=filename)

    @classmethod
    def from_srt(cls, filename):
        """
        Processes an SRT (SubRip Subtitle) file according to the SRT specifications outlined
        at http://www.textfiles.com/uploads/kds-srt.txt.

        The SRT format consists of:
            - A sequential subtitle number.
            - The start time and end time of the subtitle display, separated by '-->'.
            - The subtitle text, which can span one or more lines.
            - A blank line indicating the end of a subtitle entry.

        This function is designed to read or manipulate an SRT file based on the provided filename.

        Args:
            filename (str): The path to the SRT file to be processed.

        Returns:
            The return value is not specified in the provided docstring. Depending on the implementation,
            this function could return a data structure representing the parsed SRT file, a success status,
            or possibly nothing.
        """
        filename = Path(filename)
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()

        # # Remove comments
        # lines = [line for line in lines if not line.startswith('#')]

        segments = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().isdigit():
                start_str, end_str = re.search(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[i + 1]).group(1, 2)
                start = parse_timestamp(start_str, fmt='srt')
                end = parse_timestamp(end_str, fmt='srt')
                i += 2

                # Multi-line subtitle
                text = []
                while lines[i].strip():
                    text.append(lines[i].strip())
                    i += 1

                text = '\n'.join(text)
                segments.append({'start': start, 'end': end, 'text': text})

                i += 1
            else:
                raise ValueError(f'Invalid srt file {filename}')

        lang = detect_lang(' '.join(segment['text'] for segment in segments[:10]))

        return cls(language=lang, segments=segments, filename=filename)


@dataclass
class BilingualElement:
    """
    Save a LRC format element.
    """
    start: float
    end: Union[float, None]
    src_text: str
    target_text: str

    @property
    def duration(self):
        if self.end:
            return self.end - self.start
        else:
            return sys.maxsize  # Fake int infinity

    def to_json(self):
        return {'start': self.start, 'end': self.end, 'src': self.src_text, 'target': self.target_text}


class BilingualSubtitle:
    def __init__(self, src: Subtitle, target: Subtitle, filename: Union[str, Path]):
        if len(src) != len(target):
            raise ValueError(f'Source and target subtitle length not equal: {len(src)} vs {len(target)}')

        self.segments = []
        for src_seg, target_seg in zip(src.segments, target.segments):
            if src_seg.start != target_seg.start or src_seg.end != target_seg.end:
                raise ValueError(
                    f'Source and target subtitle start time not equal: {src_seg.start} vs {target_seg.start}')
            self.segments.append(BilingualElement(src_seg.start, src_seg.end, src_seg.text, target_seg.text))

        self.filename = Path(filename)
        self.suffix = '.lrc'

    @classmethod
    def from_preprocessed(cls, preprocessed_folder, audio_name):
        src_file = preprocessed_folder / f'{audio_name}_preprocessed_transcribed_optimized.json'
        target_file = preprocessed_folder / f'{audio_name}_preprocessed_transcribed_optimized_translated.json'

        if not src_file.exists() or not target_file.exists():
            raise ValueError(f'Preprocessed file not found for {audio_name}')

        src_sub = Subtitle.from_json(src_file)
        target_sub = Subtitle.from_json(target_file)

        bilingual_sub_name = preprocessed_folder / f'{audio_name}_bilingual.json'

        return cls(src_sub, target_sub, filename=bilingual_sub_name)

    def to_lrc(self):
        # If duration larger than 1 hour, use srt file instead
        if self.segments[-1].end >= 3600:
            logger.warning('Duration larger than 1 hour, use srt file instead')
            self.to_srt()
            return self.filename.with_suffix('.srt')

        lrc_path = self.filename.with_suffix('.lrc')
        fmt = partial(format_timestamp, fmt='lrc')
        with open(lrc_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments):
                print(
                    f'[{fmt(segment.start)}] {segment.target_text.strip()}\n[{fmt(segment.start)}] {segment.src_text.strip()}',
                    file=f,
                    flush=True,
                )
                if i == len(self.segments) - 1 or segment.end != self.segments[i + 1].start:
                    print(f'[{fmt(segment.end)}]', file=f, flush=True)

        logger.info(f'File saved to {lrc_path}')

    def to_srt(self):
        srt_path = self.filename.with_suffix('.srt')
        fmt = partial(format_timestamp, fmt='srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments, start=1):
                print(f'{i}\n'
                      f'{fmt(segment.start)} --> {fmt(segment.end)}\n'
                      f'{segment.target_text.strip()}\n'
                      f'{segment.src_text.strip()}\n', file=f, flush=True)

        logger.info(f'File saved to {srt_path}')
        self.suffix = '.srt'
