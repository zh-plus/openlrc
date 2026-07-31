#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

# whisper.cpp CLI transcription options used by Transcriber._build_extra_args().
default_whisper_cpp_options = {
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,
    "initial_prompt": None,
    "suppress_nst": False,  # -sns: suppress non-speech tokens
}

# File name suffixes used throughout the pipeline.
# The processing chain produces files like:
#   audio.wav -> preprocessed/audio_preprocessed.wav
#     -> audio_preprocessed_transcribed.json
#       -> audio_preprocessed_transcribed_optimized.json
#         -> audio_preprocessed_transcribed_optimized_translated.json
PREPROCESSED_SUFFIX = "_preprocessed"
TRANSCRIBED_SUFFIX = "_transcribed"
OPTIMIZED_SUFFIX = "_optimized"
RELAXED_OPTIMIZED_SUFFIX = "_optimized_relaxed"
TRANSLATED_SUFFIX = "_translated"
COMPARE_SUFFIX = "_compare"
EDIT_REPORT_SUFFIX = ".edit-report"
EDIT_SESSION_SUFFIX = ".edit-session"
NONTRANS_SUFFIX = "_nontrans"
BILINGUAL_SUFFIX = "_bilingual"
LOUDNORM_SUFFIX = "_ln"

# Directory name for preprocessed audio files.
PREPROCESSED_DIR = "preprocessed"

# Currently bottleneck-ed by Spacy
supported_languages = {
    "ca",
    "zh",
    "hr",
    "da",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "el",
    "it",
    "ja",
    "ko",
    "lt",
    "mk",
    "nb",
    "pl",
    "pt",
    "ro",
    "ru",
    "sl",
    "es",
    "sv",
    "uk",
}

supported_languages_lingua = {
    "CATALAN",
    "CHINESE",
    "CROATIAN",
    "DANISH",
    "DUTCH",
    "ENGLISH",
    "FINNISH",
    "FRENCH",
    "GERMAN",
    "GREEK",
    "ITALIAN",
    "JAPANESE",
    "KOREAN",
    "LITHUANIAN",
    "MACEDONIAN",
    "BOKMAL",
    "POLISH",
    "PORTUGUESE",
    "ROMANIAN",
    "RUSSIAN",
    "SLOVENE",
    "SPANISH",
    "SWEDISH",
    "UKRAINIAN",
}
