#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import re
from typing import List

from langcodes import Language
from lingua import LanguageDetectorBuilder

from openlrc.logger import logger

ORIGINAL_PREFIX = 'Original>'
TRANSLATION_PREFIX = 'Translation>'
PROOFREAD_PREFIX = 'Proofread>'

POTENTIAL_PREFIX_COMBOS = [
    [ORIGINAL_PREFIX, TRANSLATION_PREFIX],
    ['原文>', '翻译>'],
    ['原文>', '译文>'],
    ['原文>', '翻譯>'],
    ['原文>', '譯文>'],
    ['Original>', 'Translation>'],
    ['Original>', 'Traducción>']
]


class BaseValidator:
    def __init__(self, target_lang):
        self.target_lang = target_lang
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()


class ChunkedTranslateValidator(BaseValidator):
    def _extract_translation(self, content: str) -> List[str]:
        for potential_ori_prefix, potential_trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translation = re.findall(f'{potential_trans_prefix}\n*(.*?)(?:#\\d+|<summary>|\\n*$)', content, re.DOTALL)
            if translation:
                return translation
        return []

    def _is_translation_in_target_language(self, translation: List[str]) -> bool:
        if len(translation) >= 3:
            chunk_size = len(translation) // 3
            translation_chunks = [translation[i:i + chunk_size] for i in range(0, len(translation), chunk_size)]
            if len(translation_chunks) > 3:
                translation_chunks[-2].extend(translation_chunks[-1])
                translation_chunks.pop()

            translated_langs = [self.lan_detector.detect_language_of(' '.join(chunk)) for chunk in translation_chunks]
            translated_langs = [lang.name.lower() for lang in translated_langs if lang]

            if not translated_langs:
                return True

            translated_lang = max(set(translated_langs), key=translated_langs.count)
        else:
            detected_lang = self.lan_detector.detect_language_of(' '.join(translation))
            if not detected_lang:
                return True
            translated_lang = detected_lang.name.lower()

        target_lang = Language.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated language is {translated_lang}, not {target_lang}.')
            return False

        return True

    def validate(self, user_input, generated_content):
        summary = re.search(r'<summary>(.*)</summary>', generated_content)
        scene = re.search(r'<scene>(.*)</scene>', generated_content)

        original = re.findall(ORIGINAL_PREFIX + r'\n(.*?)\n' + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error(f'Fail to extract original text.')
            return False

        translation = self._extract_translation(generated_content)
        if not translation:
            logger.warning(f'Fail to extract translation.')
            logger.debug(f'Content: {generated_content}')
            return False

        if len(original) != len(translation):
            logger.warning(
                f'Fail to ensure length consistent: original is {len(original)}, translation is {len(translation)}')
            logger.debug(f'original: {original}')
            logger.debug(f'translation: {translation}')
            return False

        if not self._is_translation_in_target_language(translation):
            return False

        if not summary or not summary.group(1):
            logger.warning(f'Fail to extract summary.')
        if not scene or not scene.group(1):
            logger.warning(f'Fail to extract scene.')

        return True


class AtomicTranslateValidator(BaseValidator):
    def validate(self, user_input, generated_content):
        detected_lang = self.lan_detector.detect_language_of(generated_content)
        if not detected_lang:
            return True

        translated_lang = detected_lang.name.lower()
        target_lang = Language.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated text: "{generated_content}" is {translated_lang}, not {target_lang}.')
            return False

        return True


class ProofreaderValidator(BaseValidator):
    def validate(self, user_input, generated_content):
        original = re.findall(ORIGINAL_PREFIX + r'\n(.*?)\n' + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error(f'Fail to extract original text.')
            return False

        localized = re.findall(PROOFREAD_PREFIX + r'\s*(.*)', generated_content, re.MULTILINE)

        if not localized:
            logger.warning(f'Fail to extract translation.')
            logger.debug(f'Content: {generated_content}')
            return False

        if len(original) != len(localized):
            logger.warning(
                f'Fail to ensure length consistent: original is {len(original)}, translation is {len(localized)}')
            logger.debug(f'original: {original}')
            logger.debug(f'translation: {localized}')
            return False

        return True


class ContextReviewerValidateValidator(BaseValidator):
    def validate(self, user_input: str, generated_content: str) -> bool:
        """
        Validate the generated content based on user input.

        Args:
            user_input (str): The user input to compare against.
            generated_content (str): The content generated for validation.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        if re.search(r'\b(?:true|false)\b', generated_content, re.IGNORECASE):
            return True
        else:
            logger.warning(f'Context reviewer validation failed: {generated_content}.')

        return False
