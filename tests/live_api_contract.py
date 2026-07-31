"""Deterministic assertions shared by opt-in live provider tests."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from openlrc.utils import detect_lang


def _comparable_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def assert_translation_contract(
    source: str, translation: str, target_lang: str, *, preserved_terms: Iterable[str] = ()
) -> None:
    """Check stable live-translation properties without scoring one reference wording."""
    stripped = translation.strip()
    if not stripped:
        raise AssertionError("translation must not be empty")

    if _comparable_text(stripped) == _comparable_text(source):
        raise AssertionError("translation must differ from the source text")

    try:
        detected_lang = detect_lang(stripped)
    except RuntimeError as exc:
        raise AssertionError(f"translation language could not be detected: {exc}") from exc
    if detected_lang != target_lang:
        raise AssertionError(f"expected target language {target_lang!r}, detected {detected_lang!r}")

    normalized_translation = _comparable_text(stripped)
    missing_terms = [term for term in preserved_terms if _comparable_text(term) not in normalized_translation]
    if missing_terms:
        raise AssertionError(f"translation did not preserve required terms: {missing_terms!r}")
