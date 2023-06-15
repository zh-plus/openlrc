import math
from typing import Union

import opencc

from openlrc.logger import logger
from openlrc.subtitle import Subtitle
from openlrc.utils import extend_filename


class SubtitleOptimizer:
    def __init__(self, subtitle: Union[str, Subtitle]):
        if isinstance(subtitle, str):
            subtitle = Subtitle(subtitle)

        self.subtitle = subtitle

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

        logger.debug(f'Merge same text: {len(self.subtitle.segments)} -> {len(new_elements)}')

        self.subtitle.segments = new_elements

    def merge_short(self, threshold=2):
        """
        Merge the short text.
        """
        new_elements = []

        for i, element, in enumerate(self.subtitle.segments):
            if i == 0 or element.duration >= threshold:
                new_elements.append(element)
            else:
                new_elements[-1].text += ' ' + element.text
                new_elements[-1].end = element.end

        logger.debug(f'Merge short text: {len(self.subtitle.segments)} -> {len(new_elements)}')

        self.subtitle.segments = new_elements

    def merge_repeat(self):
        """
        Merge the same pattern in one lyric.
        :return:
        """
        new_elements = self.subtitle.segments

        def get_repeat(text):
            """
            Check if the text is repeated for [1-4] words.
            """
            for j in range(1, 5):
                repeating_num = math.floor(len(text) / float(j))
                if text[:j] * repeating_num == text[:j * repeating_num]:
                    return text[:j]
            return None

        for i in range(len(new_elements)):
            repeat_text = get_repeat(new_elements[i].text)
            if repeat_text:
                new_elements[i].text = repeat_text + '...(Repeat)'
                logger.debug(f'Merge same words: {repeat_text}')

        logger.debug('Merge same words done.')

        self.subtitle.segments = new_elements

    def cut_long(self, threshold=125, keep=20):
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            if len(element.text) > threshold:
                logger.warning(f'Cut long text: {element.text}\nInto: {element.text[:keep]}...')
                new_elements[i].text = element.text[:keep] + f'(Cut to {keep})'

        logger.debug('Cut long text done.')

        self.subtitle.segments = new_elements

    def traditional2mandarin(self):
        new_elements = self.subtitle.segments

        converter = opencc.OpenCC('t2s.json')
        for i, element in enumerate(new_elements):
            new_elements[i].text = converter.convert(element.text)

        logger.debug('Traditional Chinese to Mandarin done.')

        self.subtitle.segments = new_elements

    def remove_unk(self):
        new_elements = self.subtitle.segments

        for i, element in enumerate(new_elements):
            new_elements[i].text = element.text.replace('<unk>', ' ')

        logger.debug('Remove <unk> done.')

        self.subtitle.segments = new_elements

    def remove_empty(self):
        self.subtitle.segments = [element for element in self.subtitle.segments if element.text]

        logger.debug('Remove empty done.')

    def perform_all(self, t2m=False):
        for _ in range(2):
            self.merge_same()
            self.merge_short()
            # self.merge_same_words()
            self.cut_long()
            self.remove_unk()
            self.remove_empty()

            if t2m or self.subtitle.lang.lower() == 'zh-cn':
                self.traditional2mandarin()

    def save(self, output_name=None, update_name=False):
        optimized_name = extend_filename(self.filename, '_optimized') if not output_name else output_name
        self.subtitle.save(optimized_name, update_name=update_name)
        logger.info(f'Optimized json file saved to {optimized_name}')
