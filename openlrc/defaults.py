#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
from lingua import Language

# Check https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L184 for details
default_asr_options = {
    "beam_size": 3,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "temperature": 0,

    # We assume the voice is valid after VAD, log_prob_threshold is not reliable, set these 3 to None to prevent
    # miss-transcription, see https://github.com/openai/whisper/discussions/29#discussioncomment-3726710 for details
    "compression_ratio_threshold": None,
    "log_prob_threshold": None,
    "no_speech_threshold": None,

    "condition_on_previous_text": False,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": False,
    "max_initial_timestamp": 0.0,
    "word_timestamps": True,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "hallucination_silence_threshold": 2,
}

# Check https://github.com/guillaumekln/faster-whisper/blob/3b4a6aa1c22d293ddde9f08bdd31fc842086a6ea/faster_whisper/vad.py#L14 for details
default_vad_options = {
    "threshold": 0.382,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 2000,
    "window_size_samples": 1024,
    "speech_pad_ms": 400,
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
