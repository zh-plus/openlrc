# Changelog

## OpenLRC Mac 0.2.0

Local translation and first-stage CLI release for the macOS fork.

### Added

- Added `vendor/llama.cpp` as a pinned submodule for local LLM translation.
- Added reusable `openlrc.setup.llama_cpp` and `scripts/setup_llama_cpp.py` to
  build `llama-server` / `llama-cli` and download the default Qwen GGUF model.
- Added `openlrc.llama_resources` to resolve `llama-server`, `llama-cli`, and
  GGUF model paths from explicit config, environment variables, app resources,
  submodule builds, default user model directories, and `PATH`.
- Added `openlrc.local_llm_server` to start, reuse, health-check, idle-shutdown,
  and manually close local `llama-server` processes.
- Added opt-in local Qwen translation through `LRCer.local()` and
  `TranslationConfig.local_qwen35_9b(...)`.
- Added `ModelProvider.LOCAL_LLAMA` routing through the existing
  OpenAI-compatible `GPTBot` path with local translation cost fixed at zero.
- Added the first-stage `openlrc` Typer/Rich CLI with `doctor`, `models status`,
  `setup`, `transcribe`, `translate`, and `run` workflows.
- Added an `openlrc-mac` CLI alias for the fork distribution identity.

### Changed

- Renamed the Python distribution metadata from `openlrc` to `openlrc-mac`
  while keeping `import openlrc` compatible with upstream-style user code.
- Updated `openlrc --version` to report `OpenLRC Mac 0.2.0` and the upstream
  OpenLRC base version.
- Kept `openlrc run` safe by default: translation remains disabled unless the
  user passes `--translation local` or `--translation online`.
- Extracted whisper.cpp setup into reusable package helpers while keeping the
  existing setup script as a compatibility wrapper.
- Updated README, TODO, and architecture documentation for the local
  translation flow, CLI workflow, and fork distribution identity.
- Updated optional dependency install hints to use `openlrc-mac[...]`.

### Verified

- Verified the full pytest suite: `226 passed, 25 skipped`.
- Verified `openlrc --version` and `openlrc-mac --version`.
- Verified CLI smoke tests for `doctor`, `models status`, and local
  `run --translation local`.
- Verified real local translation with `llama.cpp` / Qwen through the OpenLRC
  pipeline.
- Verified Ruff linting for `openlrc/`, `tests/`, and `scripts/`.

## OpenLRC Mac 0.1.2

Dependency update release for the local `whisper.cpp` backend.

### Changed

- Updated the `vendor/whisper.cpp` submodule from `v1.8.5` to `v1.9.1`.
- Rebuilt the local `whisper-cli` with the updated submodule source.

### Verified

- Verified `resolve_whisper_cli("")` resolves the rebuilt local CLI.
- Verified a real local transcription smoke test with
  `tests/data/test_audio.wav`.
- Verified the full pytest suite: `195 passed, 25 skipped`.
- Verified Ruff linting: `uv run ruff check openlrc/ tests/`.

## OpenLRC Mac 0.1.1

Code cleanup release for the macOS `whisper.cpp` fork.

### Removed

- Removed the legacy upstream Streamlit GUI package from `openlrc/gui_streamlit`.
- Removed the broken `openlrc gui` console entry point instead of keeping a
  non-working GUI command.
- Removed the unused Streamlit screenshot resource.
- Removed faster-whisper/CUDA-era transcription compatibility fields from
  `TranscriptionConfig`:
  - `compute_type`
  - `device`
  - `vad_options`
- Removed faster-whisper default ASR/VAD option tables from `openlrc/defaults.py`.
- Removed unused runtime dependencies that were only needed by deleted or
  non-runtime surfaces:
  - `click`
  - `pip`

### Changed

- Standardized transcription defaults around `whisper.cpp` CLI options.
- Updated `Transcriber` to expose only the active `whisper.cpp` constructor
  surface.
- Updated `whisper_types.py` docs so Segment and Word are described as OpenLRC
  pipeline types instead of faster-whisper replacements.
- Updated tests to assert that removed faster-whisper fields are no longer
  accepted.

### Verified

- Verified the full pytest suite: `195 passed, 25 skipped`.
- Verified Ruff linting: `uv run ruff check openlrc/ tests/`.
- Verified formatting for touched files.

## OpenLRC Mac 0.1.0

Initial macOS-focused fork release based on upstream OpenLRC `1.7.0a1`.

### Added

- Integrated `whisper.cpp` as a Git submodule under `vendor/whisper.cpp`.
- Added `scripts/setup_whisper_cpp.py` to initialize the submodule, build
  `whisper-cli` in CMake Release mode, and download default GGML models.
- Added `openlrc/whisper_resources.py` to resolve `whisper-cli`, Whisper models,
  and VAD models through one shared runtime path.
- Added support for semantic Whisper model names such as `base`, `small`,
  `medium`, and `large-v3-turbo`.
- Added environment variable overrides for local development and future macOS
  app packaging:
  - `OPENLRC_WHISPER_CLI`
  - `OPENLRC_WHISPER_MODEL`
  - `OPENLRC_WHISPER_VAD_MODEL`
  - `OPENLRC_WHISPER_MODEL_DIR`

### Changed

- Replaced the old default transcription path with local `whisper.cpp` CLI
  transcription.
- Updated `TranscriptionConfig` defaults to semantic local resources:
  - `cli_path=""`
  - `whisper_model="ggml-base.bin"`
  - `vad_model="ggml-silero-v6.2.0.bin"`
- Rewrote the README around this fork's macOS, `whisper.cpp`, and future GUI
  direction.
- Added `click` as an explicit dependency because the real transcription flow
  imports SpaCy's CLI path during language model setup.

### Verified

- Built `vendor/whisper.cpp/build/bin/whisper-cli` successfully on macOS with
  Metal backend enabled.
- Downloaded and resolved `ggml-base.bin` and `ggml-silero-v6.2.0.bin` from
  `~/Library/Application Support/OpenLRC/models/`.
- Verified real local transcription through `WhisperCLIBackend`,
  `Transcriber`, and `LRCer().transcribe(...)`.
- Verified the full pytest suite: `194 passed, 25 skipped`.

---

## Upstream OpenLRC Changelog

The entries below are inherited from [zh-plus/openlrc](https://github.com/zh-plus/openlrc)
and predate this macOS fork.

## 1.7.0a1

Add a token-efficient lean translation pipeline, broaden model-provider routing, and refresh the configuration surface for mixed-model workflows.

### New Features:

- Add `LeanTranslator` with compact anchored prompts, sliding context, checkpoint resume, retry handling, binary-split recovery, and atomic fallback for missing lines.
- Add lean mode integration to `LRCer` through `TranslationConfig(translate_mode="lean")`, including optional Context Review and separate CR/translation chatbots.
- Add LiteLLM support as an optional provider gateway via `openlrc[litellm]` and `litellm:<provider/model>` routing.
- Add `ModelConfig.extra_body` passthrough for provider-specific request parameters across OpenAI-compatible, Anthropic, Gemini, and LiteLLM backends.

### Other Changes:

- Replace legacy string-based translation config fields with `ModelConfig` objects for primary, retry, and Context Review chatbots.
- Decouple model capability limits from the static model registry; `max_tokens` and `context_window` are now opt-in runtime caps.
- Improve custom-provider and provider-prefix inference for OpenAI-compatible, Anthropic, Gemini, LiteLLM, and third-party model names.
- Restrict lingua language detection to the supported language set and add anchor-based validation for lean translation output.
- Split media helpers from `utils` into `media_utils` to keep heavyweight media dependencies isolated.
- Centralize live API test configuration, add LeanTranslator and LiteLLM regression coverage, and update CI to exercise the new optional `litellm` extra.
- Refresh README examples for lean mode, mixed-model translation, live test configuration, and the current local-LLM/provider support status.

## 1.6.3

Improve translation reliability, reduce heavy import overhead, and tighten release tooling around the `uv` workflow.

### New Features:

- Add token-aware, scene-sensitive chunk splitting for long subtitle translation.
- Add chunked guideline generation for long subtitle files.

### Other Changes:

- Fully defer heavyweight package-root imports and move torch/deepfilternet behind the optional `full` extra.
- Harden ChatBot and GeminiBot lifecycle handling, retry behavior, and error recovery.
- Reduce translation log noise and add `OPENLRC_LOG_LEVEL`.
- Improve line-mismatch retry behavior and raise the default context window to 32K.
- Replace runtime `assert` checks with explicit exceptions.
- Fix empty-translation early return, continuous-script duplicate word handling, and lazy import test stdout parsing.
- Gate long stress tests behind `OPENLRC_TEST_STRESS` and expand chunking/translation coverage.

## 1.6.2

Refine the transcription and translation pipeline, expand test coverage, and improve the release workflow around `uv`.

### New Features:

- Add `TranscriptionConfig` and `TranslationConfig` dataclasses for clearer runtime configuration.
- Decouple transcription and translation in `LRCer`.
- Add OpenRouter fallback model support.

### Other Changes:

- Add `skip_preprocess` support.
- Fix `retry_model` validation.
- Improve duplicate handling and video format inference in the decoupled flow.
- Refresh README examples to the new config API.
- Add mock and live regression coverage for translator and `LRCer` behavior.
- Add Ruff and Pyright checks and update CI environment variables for live API tests.

## 1.6.1

Replace poetry with uv package manager. Fix #69.

### Other Changes:

- Hot fix for missed mp3 files issue.
- Allow un-recorded (not in models.py) model name.

## 1.6.0

Update faster-whisper to the latest version. Add models.py for model info.

### Other Changes:

- Fixed issue #60.
- Change default parameters for new faster-whisper.
- Update installation guide in README.

## 1.5.2

Code refactoring, documentation updates, and minor bug fixes.

### Other Changes:

- Fixed issue #54.
- Changed the default LLM model to `gpt-4o-mini`.
- Minor improvements on prompts.
- Improve post-optimizations for subtitles.

## 1.5.1

Minor bugfixes and improvements.

### Other Changes:

- Fix default check_format returning False issue.
- Fix edges cases for srt file generation.
- Prepare translation evaluator for benchmarking.

## 1.5.0

This update add Gemini Models support for translation.

### New Features:

- Support Gemini as translation engine.

### Other Changes:

- Extract the `check_format` section into a standalone `validators.py` module.
- Add stop_sequences support for Chatbot.
- Use stop sequences when building Context.
- Also remove generated .wav files from videos if `clear_temp=True`.
- Minor improvements on Context Reviewer prompt.

## 1.4.1

Minor bugfixes and improvements.

### Other Changes:

- Improve timestamp accuracy.
- Fix edges cases for transcription.

## 1.4.0

### New Features:

- Add glossary support for domain specific translation.
- Add `retry_model` args for retrying translation with different models.
- Introduce `privider: model_name` for arbitrary model routing.

### Other Changes:

- Extend 0.5s for each suitable sentence.
- Reduce noise suppression chunk-size for faster processing.
- Enhance translation workflow by Context Reviewer Agent.

## 1.3.1

### New Features:

- Add custom endpoint (base_url) support for OpenAI & Anthropic.
- Generating bilingual subtitles.

### Other Changes:

- Fix dep issues from ctranslate2 and streamlit-related packages.

## 1.3.0

Add basic GUI support via streamlit.

#### Other Changes:

- Add clear_temp_folder args.

## 1.2.0

This update add Claude Models support for translation.

## 1.1.0

This update improve the translation quality.

#### Other Changes:

- Update faster-whisper version to 1.0.0.
- Set hallucination_silence_threshold to 2, which alleviates the hallucination issue.
- Add proxy argument.

## 1.0.3

This update fix minor issues.

#### Other Changes:

- Remove water mark in srt and lrc.
- Improve logging.

## 1.0.2

This update add minor features.

#### Other Changes:

- Binlingual subtitle support (Beta).
- Improve translation prompt.

## 1.0.1

This update fix minor issues.

#### Other Changes:

- Fix issue that prevent the usage of whisper-large-v3.
- Remove tags from translation texts.

## 1.0.0

This update introduce new features.

#### New features:

- Resume from previous translation.
- Add atomic translation for src-trans inconsistency.

#### Other Changes:

- Update default whisper model to `whisper-large-v3`.

## 0.2.3

This update improves preprocess efficiency and minor changes.

#### Other Changes:

- Introduce multiprocessing for loudness normalization.
- Fix the .srt generation issue for video input.
- Add preprocess options, which users can tune.

## 0.2.2

This update addresses minor issues.

#### Other Changes:

- Split audio during noise suppression to avoid out-of-memory.
- Improve translation prompt.

## 0.2.1

This update adds a preprocessor to enhance input audio (loudness normalization & noise suppression).

#### New features:

- Loudness Normalization from [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize)
- Noise Suppression from [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

#### Other Changes:

- Now all the intermediate files are saved in `./path/to/audio/preprocess`.

## 0.2.0

This update switch the underlying transcription model from `whisperx` to `faster-whisper`, which enable VAD parameter
tuning.

#### New features:

- Switch whisperx back to faster-whisper for VAD parameter tuning.

#### Other Changes:

- Update translation prompt
  from https://github.com/machinewrapped/gpt-subtrans/commit/82bd2ca0d868f209d0e0c5f7c04255523daabe3c.
- Change the default parameters of `faster-whisper` for consistent transcription.

## 0.1.5

Emergent bugfix release.

## 0.1.4

This update add input video support and introduce context configuration.

#### New features:

- Add `word_align` and `sentence_split` for non-word-boundary languages to split long text into sentences.
- Add text-normalization for help matching sentences.
- Add skip-translation support.

#### Other Changes:

- Use `pathlib` to handle paths.
- Improve timeline accuracy.

## 0.1.3

This update add input video support and introduce context configuration.

#### New features:

- Add input video support.
- Add context configuration for inputs.

#### Other Changes:

- Add test suites to CI.
- Add language detection for translated content.
- Improve prompt by adding background info.
- Update punctuator model.
- Replace `opencc` with more light-weight `zhconv`.

## 0.1.2

This update improves the timeline consistency of translated subtitles.
Thanks [gpt-subtrans](https://github.com/machinewrapped/gpt-subtrans)!

#### New features:

- Fix misaligned timeline issue by improving translation prompt.
- Add output srt format support.
- Add changeable temperature and top_p parameter for GPTBot.

#### Other Changes:

- Report total OpenAI translation fee for multiple audios.
- Improve repeat-checking algorithm.

## 0.1.1

This update enhances the efficiency of processing multiple audio files.

#### New features:

- Implementation of a producer-consumer model to process multiple audio files.

#### Other Changes:

- Update logger with colored format.
- Minor parameter modification that makes the timeline of translation more intuitive.

## 0.1.0

This update significantly improves translation quality, but at the cost of slower translation speed.

#### New Features:

- Use multi-step prompt for translation.
- Update the default model to `gpt-3.5-turbo-16k`.
- Automatically fix json encoder error using GPT.

#### Other Changes:

- Calculate the accurate price for OpenAI API requests.

## 0.0.6

This update greatly improves the quality of transcription (both in time-alignment and text-quality).

#### New Features:

- Use `whisperx` to improve transcription accuracy.
- Add Traditional Chinese to Mandarin optimization when `target_lang=zh-cn`.

## 0.0.5

#### New Features:

#### Other Changes:

- Update build tool to poetry.

## 0.0.4

#### New Features:

- Use async call to communicate with OpenAI api.
- Abstract the GPT communication module as `GPTBot`.
- Add fee limit for GPTBot.
