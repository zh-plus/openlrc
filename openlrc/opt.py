#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import re
from pathlib import Path
from typing import Union

import zhconv

from openlrc.logger import logger
from openlrc.subtitle import Subtitle
from openlrc.utils import extend_filename

# Different language may need different threshold
cut_long_threshold = {
    350: 'en',
    125: 'cn, ja'
}


class SubtitleOptimizer:
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
        Merge the same text.
        """
        new_elements = []

        for i, element in enumerate(self.subtitle.segments):
            if i == 0 or element.text != new_elements[-1].text:
                new_elements.append(element)
            else:
                new_elements[-1].end = element.end

        self.subtitle.segments = new_elements

    def merge_short(self, threshold=1.5):
        """
        Merge the short duration subtitle.
        """
        new_elements = []

        merged_element = None
        for i, element, in enumerate(self.subtitle.segments):
            if i == 0 or element.duration >= threshold:
                if merged_element:
                    if merged_element.duration < 1:
                        # If the merged elements is still too small, find the smaller element nearby and merge it
                        if new_elements[-1].duration > element.duration:
                            # merge to the next
                            element.text = merged_element.text + element.text
                            element.start = merged_element.start
                        else:
                            # merge to the last
                            new_elements[-1].text += merged_element.text
                            new_elements[-1].end = merged_element.end
                    else:
                        new_elements.append(merged_element)
                    new_elements.append(element)
                    merged_element = None
                else:
                    new_elements.append(element)
            else:
                if not merged_element:
                    merged_element = element
                else:
                    merged_element.text += element.text
                    merged_element.end = element.end

        self.subtitle.segments = new_elements

    def merge_repeat(self):
        """
        Merge the same pattern in one lyric.
        :return:
        """
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            text = element.text
            text = re.sub(r'(.)\1{4,}', r'\1\1...', text)
            text = re.sub(r'(.+)\1{4,}', r'\1\1...', text)
            new_elements[i].text = text

        self.subtitle.segments = new_elements

    def cut_long(self, keep=20):
        threshold = 150
        for threshold, lang in cut_long_threshold.items():
            if self.lang.lower() in lang:
                threshold = threshold
                break

        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            if len(element.text) > threshold and len(element.text) / len(set(element.text)) > 3.0:
                # text length larger than threshold and text density is larger than 3.0
                logger.warning(f'Cut long text: {element.text}\nInto: {element.text[:keep]}...')
                new_elements[i].text = element.text[:keep]

        self.subtitle.segments = new_elements

    def traditional2mandarin(self):
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            new_elements[i].text = zhconv.convert(element.text, locale='zh-cn')

        self.subtitle.segments = new_elements

    def remove_unk(self):
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            new_elements[i].text = element.text.replace('<unk>', '')

        self.subtitle.segments = new_elements

    def remove_empty(self):
        self.subtitle.segments = [element for element in self.subtitle.segments if element.text]

    def strip(self):
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            new_elements[i].text = element.text.strip()

        self.subtitle.segments = new_elements

    def perform_all(self):
        for _ in range(2):
            self.merge_same()
            self.merge_short()
            self.merge_repeat()
            self.cut_long()
            self.remove_unk()
            self.remove_empty()
            self.strip()

            if self.subtitle.lang.lower() in ['zh-cn', 'zh']:
                self.traditional2mandarin()

    def save(self, output_name=None, update_name=False):
        optimized_name = extend_filename(self.filename, '_optimized') if not output_name else output_name
        self.subtitle.save(optimized_name, update_name=update_name)
        logger.info(f'Optimized json file saved to {optimized_name}')
