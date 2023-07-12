#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

# Check https://github.com/guillaumekln/faster-whisper/blob/3b4a6aa1c22d293ddde9f08bdd31fc842086a6ea/faster_whisper/transcribe.py#L153 for details
default_asr_options = {
    "beam_size": 3,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
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
