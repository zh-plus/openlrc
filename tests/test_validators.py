#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import unittest

from openlrc.validators import ChunkedTranslateValidator, AtomicTranslateValidator, ContextReviewerValidateValidator


class TestChunkedTranslateValidator(unittest.TestCase):

    #  Validate correctly extracts original text and translation
    def test_validate_correctly_extracts_original_and_translation(self):
        user_input = "Original>\nThis is a test.\nTranslation>\nEsto es una prueba."
        generated_content = "<summary>Test Summary</summary><scene>Test Scene</scene>\nOriginal>\nThis is a test.\nTranslation>\nEsto es una prueba."
        validator = ChunkedTranslateValidator(target_lang='es')
        result = validator.validate(user_input, generated_content)
        self.assertTrue(result)

    #  Validate handles empty user input and generated content
    def test_validate_handles_empty_input_and_content(self):
        user_input = ""
        generated_content = ""
        validator = ChunkedTranslateValidator(target_lang='es')
        result = validator.validate(user_input, generated_content)
        self.assertFalse(result)

    #  Validate ensures translation is in the target language
    def test_translation_not_in_target_language(self):
        user_input = "Original>\nThis is a test.\nTranslation>\nEsto es una prueba."
        generated_content = "<summary>Test Summary</summary><scene>Test Scene</scene>\nOriginal>\nThis is a test.\nTranslation>\n这是一个测试。"
        validator = ChunkedTranslateValidator(target_lang='es')
        result = validator.validate(user_input, generated_content)
        self.assertFalse(result)

    #  Validate handles content with summary and scene tags
    def test_handles_summary_and_scene_tags(self):
        user_input = "Original>\nThis is a test.\nTranslation>\nEsto es una prueba."
        generated_content = "<summary>Test Summary</summary><scene>Test Scene</scene>\nOriginal>\nThis is a test.\nTranslation>\nEsto es una prueba."
        validator = ChunkedTranslateValidator(target_lang='es')
        result = validator.validate(user_input, generated_content)
        self.assertTrue(result)

    #  Validate returns True for matching original and translation lengths
    def test_mismatched_original_and_translation_lengths(self):
        user_input = "Original>\nThis is a test.\nTranslation>\nEsto es una prueba."
        generated_content = "<summary>Test Summary</summary><scene>Test Scene</scene>\nOriginal>\nThis is a test.\nTranslation>\nEsto es una prueba.\nOriginal>\nThis is a test!\nTranslation>\nEsto es una prueba!"
        validator = ChunkedTranslateValidator(target_lang='es')
        result = validator.validate(user_input, generated_content)
        self.assertFalse(result)


class TestAtomicTranslateValidator(unittest.TestCase):

    def test_validate_returns_true_when_generated_content_matches_target_language(self):
        validator = AtomicTranslateValidator(target_lang='en')
        user_input = "Hello"
        generated_content = "Hello"

        result = validator.validate(user_input, generated_content)
        self.assertTrue(result)

    def test_validate_returns_false_when_generated_content_not_matches_target_language(self):
        validator = AtomicTranslateValidator(target_lang='en')
        user_input = "Hello"
        generated_content = "你好"

        result = validator.validate(user_input, generated_content)
        self.assertFalse(result)


class TestContextReviewerValidateValidator(unittest.TestCase):
    def test_validate_returns_true_when_true_in_generated_content(self):
        validator = ContextReviewerValidateValidator(target_lang='en')
        user_input = "Some input"
        generated_content = "This is true"
        result = validator.validate(user_input, generated_content)
        self.assertTrue(result)

    def test_validate_returns_true_when_false_in_generated_content(self):
        validator = ContextReviewerValidateValidator(target_lang='en')
        user_input = "Some input"
        generated_content = "FALse!"
        result = validator.validate(user_input, generated_content)
        self.assertTrue(result)

    def test_validate_returns_false_when_no_true_or_false_in_generated_content(self):
        validator = ContextReviewerValidateValidator(target_lang='en')
        user_input = "Some input"
        generated_content = "I dont know!"
        result = validator.validate(user_input, generated_content)
        self.assertFalse(result)
