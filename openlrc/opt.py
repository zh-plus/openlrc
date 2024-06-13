#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import re
from pathlib import Path
from typing import Union, Optional, List

import zhconv

from openlrc.logger import logger
from openlrc.subtitle import Subtitle
from openlrc.utils import extend_filename, format_timestamp

# Thresholds for different languages
CUT_LONG_THRESHOLD = {
    'en': 350,
    'cn': 125,
    'ja': 125
}

# Punctuation mapping
PUNCTUATION_MAPPING = {
    ',': '，',
    '.': '。',
    '?': '？',
    '!': '！',
    ':': '：',
    ';': '；',
    '"': '”',
    "'": '’',
    '(': '（',
    ')': '）',
    '[': '【',
    ']': '】',
    '{': '｛',
    '}': '｝'
}


class SubtitleOptimizer:
    """
    SubtitleOptimizer class is used to optimize subtitles by performing various operations.
    """

    def __init__(self, subtitle: Union[Path, Subtitle]):
        if isinstance(subtitle, Path):
            subtitle = Subtitle.from_json(subtitle)

        self.subtitle = subtitle
        self.lang = self.subtitle.lang

    @property
    def filename(self):
        return self.subtitle.filename

    def merge_same(self):
        """
        Merge consecutive segments with the same text.
        """
        new_elements = []

        for i, element in enumerate(self.subtitle.segments):
            if i == 0 or element.text != new_elements[-1].text:
                new_elements.append(element)
            else:
                new_elements[-1].end = element.end

        self.subtitle.segments = new_elements

    def merge_short(self, threshold=1.2):
        """
        Merge short duration subtitles.
        """
        new_elements = []
        merged_element = None

        for i, element in enumerate(self.subtitle.segments):
            if i == 0 or element.duration >= threshold:
                if merged_element:
                    self._finalize_merge(new_elements, merged_element, element)
                    merged_element = None
                new_elements.append(element)
            else:
                if not merged_element:
                    merged_element = element
                    continue

                # Merge to previous element if closer to pre-element and gap > 3s
                previous_gap = merged_element.start - new_elements[-1].start
                next_gap = element.start - merged_element.end
                if previous_gap <= next_gap and previous_gap <= 3:
                    previous_element = new_elements.pop()
                    merged_element.text = previous_element.text + merged_element.text
                    merged_element.start = previous_element.start
                    new_elements.append(merged_element)
                    merged_element = element
                elif next_gap <= previous_gap and next_gap <= 3:
                    merged_element.text += element.text
                    merged_element.end = element.end
                    new_elements.append(merged_element)
                    merged_element = None
                else:
                    new_elements.append(merged_element)
                    merged_element = element

        self.subtitle.segments = new_elements

    def _finalize_merge(self, new_elements, merged_element, element):
        if merged_element.duration < 1.5:
            previous_gap = merged_element.start - new_elements[-1].end
            next_gap = element.start - merged_element.end
            if previous_gap <= next_gap and previous_gap <= 3:
                new_elements[-1].text += merged_element.text
                new_elements[-1].end = merged_element.end
            elif next_gap <= previous_gap and next_gap <= 3:
                element.text = merged_element.text + element.text
                element.start = merged_element.start
            else:
                new_elements.append(merged_element)
        else:
            new_elements.append(merged_element)

    def _merge_elements(self, merged_element, element):
        if not merged_element:
            return element
        merged_element.text += element.text
        merged_element.end = element.end
        return merged_element

    def merge_repeat(self):
        """
        Merge repeated patterns in the text.
        """
        for element in self.subtitle.segments:
            element.text = re.sub(r'(.)\1{4,}', r'\1\1...', element.text)
            element.text = re.sub(r'(.+)\1{4,}', r'\1\1...', element.text)

    def cut_long(self, max_length=20):
        """
        Cut long texts based on language-specific thresholds.
        """
        threshold = CUT_LONG_THRESHOLD.get(self.lang.lower(), 150)

        for element in self.subtitle.segments:
            if len(element.text) > threshold and len(element.text) / len(set(element.text)) > 3.0:
                logger.warning(f'Cut long text: {element.text}\nInto: {element.text[:max_length]}...')
                element.text = element.text[:max_length]

    def traditional2mandarin(self):
        """
        Convert traditional Chinese characters to simplified Chinese.
        """
        for element in self.subtitle.segments:
            element.text = zhconv.convert(element.text, locale='zh-cn')

    def punctuation_optimization(self):
        """
        Replace English punctuation with Chinese punctuation.
        """
        for element in self.subtitle.segments:
            element.text = self._replace_punctuation_with_chinese(element.text)

    def _replace_punctuation_with_chinese(self, text):
        pattern = re.compile("|".join(map(re.escape, PUNCTUATION_MAPPING.keys())))
        result = pattern.sub(lambda match: PUNCTUATION_MAPPING[match.group(0)], text)

        result = re.sub(r'(。){3,}', '...', result)
        result = re.sub(r'(\d)。', r'\1.', result)

        return result

    def remove_unk(self):
        """
        Remove '<unk>' tags from the text.
        """
        for element in self.subtitle.segments:
            element.text = element.text.replace('<unk>', '')

    def remove_empty(self):
        """
        Remove empty subtitle segments.
        """
        self.subtitle.segments = [element for element in self.subtitle.segments if element.text]

    def strip(self):
        """
        Strip whitespace from the text of each subtitle segment.
        """
        for element in self.subtitle.segments:
            element.text = element.text.strip()

    def extend_time(self):
        """
        Extend the subtitle time for each element to 0.5s.
        """
        for i, element in enumerate(self.subtitle.segments):
            if i == len(self.subtitle.segments) - 1 or self.subtitle.segments[i + 1].start - element.end > 0.5:
                element.end += 0.5

    def perform_all(self, steps: Optional[List[str]] = None, extend_time=False):
        """
        Perform all or specified optimization operations.

        Args:
            steps (list of str): List of method names to be executed in order.
                                 If None, a default sequence of operations will be executed.
            extend_time (bool): Whether to extend the subtitle time for each element.
        """
        # Check steps is valid
        if steps and any(step not in dir(self) for step in steps):
            invalid_steps = ', '.join(s for s in steps if s not in dir(self))
            raise ValueError(f'Invalid steps: {invalid_steps}')

        if steps is None:
            steps = [
                'merge_same', 'merge_short', 'merge_repeat', 'cut_long', 'remove_unk', 'remove_empty', 'strip'
            ]
            if self.lang.lower() in ['zh-cn', 'zh']:
                steps.append('traditional2mandarin')
            if self.lang.lower() in ['zh-cn', 'zh', 'zh-tw']:
                steps.append('punctuation_optimization')

        for step in steps:
            method = getattr(self, step, None)
            if method:
                method()

        if extend_time:
            self.extend_time()

        # Finally check to notify users
        self.check()

    def save(self, output_name: Optional[str] = None, update_name=False):
        """
        Save the optimized subtitle to a file.
        """
        optimized_name = extend_filename(self.filename, '_optimized') if not output_name else output_name
        self.subtitle.save(optimized_name, update_name=update_name)
        logger.info(f'Optimized json file saved to {optimized_name}')

    def check(self):
        for element in self.subtitle.segments:
            if element.duration >= 10:
                logger.warning(f'Duration of text "{element.text}" at {format_timestamp(element.start)} exceeds 10')
