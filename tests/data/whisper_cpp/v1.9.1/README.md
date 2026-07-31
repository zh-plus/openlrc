# whisper.cpp v1.9.1 fixture

`test_audio_vad_cpu.json` was captured from the vendored `whisper-cli` v1.9.1
using `tests/data/test_audio.wav`, the `base` model, Silero VAD v6.2.0, and the
explicit CPU compatibility flags `-ng -nfa`.

The fixture keeps the authentic v1.9.1 object, segment, and token shapes. It is
trimmed to the first three segments, and the machine-specific model path is
replaced with `<WHISPER_MODEL_PATH>`. Refresh it only after verifying the
submodule identity, `whisper-cli --version`, critical `--help` flags, and the
opt-in real transcription smoke tests.
