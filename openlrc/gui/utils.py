#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
from pathlib import Path
from zipfile import ZipFile


def get_asr_options(beam_size, best_of, patience, length_penalty, repetition_penalty, no_repeat_ngram_size, temperature,
                    compression_ratio_threshold, log_prob_threshold, no_speech_threshold, condition_on_previous_text,
                    initial_prompt, prefix, suppress_blank, suppress_tokens, without_timestamps, max_initial_timestamp,
                    word_timestamps, prepend_punctuations, append_punctuations, hallucination_silence_threshold):
    options = {
        "beam_size": beam_size,
        "best_of": best_of,
        "patience": patience,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "temperature": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": log_prob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "prefix": prefix,
        "suppress_blank": suppress_blank,
        "suppress_tokens": [int(x) for x in suppress_tokens.split(",") if x.strip().lstrip('-').isdigit()],
        "without_timestamps": without_timestamps,
        "max_initial_timestamp": max_initial_timestamp,
        "word_timestamps": word_timestamps,
        "prepend_punctuations": prepend_punctuations,
        "append_punctuations": append_punctuations,
        "hallucination_silence_threshold": hallucination_silence_threshold,
    }

    return options


def get_vad_options(threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms,
                    window_size_samples, speech_pad_ms):
    options = {
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": max_speech_duration_s,
        "min_silence_duration_ms": min_silence_duration_ms,
        "window_size_samples": window_size_samples,
        "speech_pad_ms": speech_pad_ms,
    }

    return options


def get_preprocess_options(atten_lim_db):
    options = {
        'atten_lim_db': atten_lim_db
    }

    return options


def zip_files(file_paths, zip_filename='zipped'):
    file_paths = [Path(path) for path in file_paths]
    zip_filename = file_paths[0].parent.with_name(zip_filename).with_suffix('.zip')
    with ZipFile(zip_filename, 'w') as zip_object:
        _ = [zip_object.write(lrc_path, arcname=lrc_path.name) for lrc_path in file_paths]

    return zip_filename
