from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.live_api_contract import assert_translation_contract


class TestLiveAPITranslationContract(unittest.TestCase):
    @patch("tests.live_api_contract.detect_lang", return_value="es")
    def test_accepts_target_language_and_preserved_facts(self, mock_detect_lang):
        assert_translation_contract(
            "John paid 12 dollars on July 5.",
            "John pagó 12 dólares el 5 de julio.",
            "es",
            preserved_terms=("John", "12", "5"),
        )

        mock_detect_lang.assert_called_once()

    def test_rejects_empty_translation(self):
        with self.assertRaisesRegex(AssertionError, "must not be empty"):
            assert_translation_contract("Hello", "   ", "es")

    def test_rejects_unchanged_translation_after_normalization(self):
        with self.assertRaisesRegex(AssertionError, "must differ"):
            assert_translation_contract("Hello, World!", " hello world ", "es")

    @patch("tests.live_api_contract.detect_lang", return_value="en")
    def test_rejects_wrong_target_language(self, mock_detect_lang):
        with self.assertRaisesRegex(AssertionError, "expected target language"):
            assert_translation_contract("Hello", "Still in English", "es")

    @patch("tests.live_api_contract.detect_lang", return_value="es")
    def test_rejects_missing_preserved_fact(self, mock_detect_lang):
        with self.assertRaisesRegex(AssertionError, "did not preserve"):
            assert_translation_contract(
                "John paid 12 dollars.", "John pagó varios dólares.", "es", preserved_terms=("John", "12")
            )
