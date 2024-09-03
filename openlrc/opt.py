#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import re
from pathlib import Path
from typing import Union, Optional, List

import zhconv

from openlrc.logger import logger
from openlrc.subtitle import Subtitle, BilingualSubtitle
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

    def __init__(self, subtitle: Union[Path, Subtitle, BilingualSubtitle]):
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

    def merge_short(self, duration_threshold=1.2):
        """
        Merge short duration subtitles.
        """
        optimized_segments = []
        current_segment = None

        for i, element in enumerate(self.subtitle.segments):
            if i == 0 or element.duration >= duration_threshold:
                if current_segment:
                    self._finalize_merge(optimized_segments, current_segment, element)
                    current_segment = None
                optimized_segments.append(element)
            else:
                if not current_segment:
                    current_segment = element
                    continue

                # Merge to previous element if closer to pre-element and gap > 3s
                previous_gap = current_segment.start - optimized_segments[-1].start
                next_gap = element.start - current_segment.end
                if previous_gap <= next_gap and previous_gap <= 3:
                    previous_element = optimized_segments.pop()
                    current_segment.text = previous_element.text + current_segment.text
                    current_segment.start = previous_element.start
                    optimized_segments.append(current_segment)
                    current_segment = element
                elif next_gap <= previous_gap and next_gap <= 3:
                    current_segment.text += element.text
                    current_segment.end = element.end
                    optimized_segments.append(current_segment)
                    current_segment = None
                else:
                    optimized_segments.append(current_segment)
                    current_segment = element

        # Handle the last merged_element if it exists
        if current_segment:
            self._finalize_merge(optimized_segments, current_segment, None)

        self.subtitle.segments = optimized_segments

    def _finalize_merge(self, optimized_segments, current_segment, next_segment):
        if next_segment and current_segment.duration < 1.5:
            previous_gap = current_segment.start - optimized_segments[-1].end
            next_gap = next_segment.start - current_segment.end
            if previous_gap <= next_gap and previous_gap <= 3:
                optimized_segments[-1].text += current_segment.text
                optimized_segments[-1].end = current_segment.end
            elif next_gap <= previous_gap and next_gap <= 3:
                next_segment.text = current_segment.text + next_segment.text
                next_segment.start = current_segment.start
            else:
                optimized_segments.append(current_segment)
        else:
            optimized_segments.append(current_segment)

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
        if isinstance(self.subtitle, BilingualSubtitle):
            logger.warning('Bilingual subtitle is not supported for cut_long operation.')
            return

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
        Replace English punctuation with Chinese punctuation where appropriate.
        """
        if isinstance(self.subtitle, BilingualSubtitle):
            logger.warning('Bilingual subtitle is not supported for punctuation_optimization operation.')
            return

        for element in self.subtitle.segments:
            element.text = self._replace_punctuation_with_chinese(element.text)

    def _replace_punctuation_with_chinese(self, text):
        # Define a regex pattern to match URLs
        url_pattern = r'https?://\S+'

        # Find all URLs in the text
        urls = re.findall(url_pattern, text)

        # Replace URLs with placeholders
        for i, url in enumerate(urls):
            text = text.replace(url, f'URL_PLACEHOLDER_{i}')

        # Replace "..." with "……"
        text = re.sub(r'\.{3,}', '……', text)

        # Replace consecutive Chinese full stops "。。。。" with "……"
        text = re.sub(r'。{3,}', '……', text)

        # Replace punctuation
        for eng, chn in PUNCTUATION_MAPPING.items():
            # Avoid replacing dots in abbreviations like "Mr.", "Mrs.", or in decimal numbers
            if eng == '.':
                text = re.sub(r'(?<!\w)\.(?!\w)|(?<=[^A-Za-z0-9])\.(?=[^A-Za-z0-9])|(?<=[^A-Za-z0-9])\.(?=$)', chn,
                              text)
            else:
                text = text.replace(eng, chn)

        # Restore URLs
        for i, url in enumerate(urls):
            text = text.replace(f'URL_PLACEHOLDER_{i}', url)

        return text

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
