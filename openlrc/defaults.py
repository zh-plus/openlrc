#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
from lingua import Language

# Check https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py for details
default_asr_options = {
    "batch_size": 8,
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "temperature": 0.0,

    # We assume the voice is valid after VAD, log_prob_threshold is not reliable, set these 3 to None to prevent
    # miss-transcription, see https://github.com/openai/whisper/discussions/29#discussioncomment-3726710 for details
    "compression_ratio_threshold": None,
    "log_prob_threshold": None,
    "no_speech_threshold": None,

    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": False,

    "word_timestamps": True,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    # "hallucination_silence_threshold": 2,
    "hotwords": None,
}

# Check https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L123 for details
default_vad_options = {
    "vad_onset": 0.500,
    "vad_offset": 0.363
}

default_preprocess_options = {
    'atten_lim_db': 15
}

# Currently bottleneck-ed by Spacy
supported_languages = {
    'ca', 'zh', 'hr', 'da', 'nl', 'en', 'fi', 'fr', 'de', 'el', 'it', 'ja', 'ko', 'lt', 'mk', 'nb', 'pl', 'pt', 'ro',
    'ru', 'sl', 'es', 'sv', 'uk'
}

supported_languages_lingua = {
    Language.CATALAN, Language.CHINESE, Language.CROATIAN, Language.DANISH, Language.DUTCH, Language.ENGLISH,
    Language.FINNISH, Language.FRENCH, Language.GERMAN, Language.GREEK, Language.ITALIAN, Language.JAPANESE,
    Language.KOREAN, Language.LITHUANIAN, Language.MACEDONIAN, Language.BOKMAL, Language.POLISH, Language.PORTUGUESE,
    Language.ROMANIAN, Language.RUSSIAN, Language.SLOVENE, Language.SPANISH, Language.SWEDISH, Language.UKRAINIAN
}
