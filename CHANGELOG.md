# Changelog

## OpenLRC Mac 0.3.0

User-guided Translation Brief, glossary compliance, and recoverable
transactional subtitle editing release, including the previously unreleased
Pro, Chunk Planner v2, checkpoint, and relaxed subtitle optimization work.

### Highlights

- **All-new Pro mode:** combines a document-level Translation Brief, aligned
  per-chunk ContextTimeline, Hy-MT2 translation, and conservative final review
  while keeping only one local large model resident at a time.
- **Glossary service:** loads legacy mappings or versioned catalogs, merges task
  and Brief terminology deterministically, validates every occurrence, and
  produces inspectable compliance reports without silently rewriting subtitles.
- **Recoverable multi-round editing:** applies deterministic checks and bounded
  semantic review through line-scoped, all-or-nothing patches, with rollback,
  checkpoints, standalone editing, and historical Restore.

### Added

- Added a shared translation chunk planner with stable chunk signatures so
  Timeline generation, translation, fallback, review, and resume use identical
  segment boundaries, plus contextual checkpoint schema v4 and validated
  migration from compatible v2 Normal/Normal Plus and v3 Pro state.
- Added public `SubtitleOptimizationMode` and CLI
  `--subtitle-optimization aggressive|relaxed`. Relaxed cleanup preserves
  segment count, order, timestamps, short/repeated lines, and `<unk>` content.
- Added versioned glossary catalogs, required/preferred compliance reports, and
  `openlrc glossary validate|inspect|check` commands.
- Added stable edit issue/patch/round/session models, deterministic structure,
  glossary, entity, number, placeholder, and immutable-field checks, plus a
  one-round default and three-round hard limit for Normal Plus and Pro.
- Added `LRCer.edit(...) -> EditResult` and independent
  `verify|review|retranslate|restore` actions that do not rerun ASR. Verify and
  restore are offline; model-using actions require explicit current config.
  JSON/Markdown reports and optional compact Restore sessions are written to the
  user output directory.
- Added full and partial manual Translation Briefs to Hy-MT2 contextual and
  standalone editing flows. `--brief-summary`, `--brief-characters JSON|@PATH`,
  and `--brief-tone-style` lock user-supplied sections while the context model
  fills only missing sections; a complete manual Brief skips automatic Brief
  and terminology extraction.

### Changed

- Renamed the public Hy-MT2 contextual modes to `normal` and `normal-plus`.
  `context` and `context-plus` remain deprecated aliases for the 0.3.x
  compatibility window and are canonicalized before business logic and
  checkpoint fingerprinting.
- Added consistent `--glossary`, `--force-glossary`, glossary strictness,
  `--edit-rounds 0..3`, and `--enable-restore` options to `run` and `translate`.
- Made subtitle JSON replacement atomic. A successful edit checkpoint is
  durable before the translated JSON is committed; incomplete work always
  retains the full process checkpoint.
- Extended every Hy-MT2 retry/split/atomic path with aligned Pro story/scene
  context. Timeline responses must echo exact chunk and ordered segment IDs;
  invalid alignment triggers retry. Fully local Pro stages Qwen -> Hy-MT2 ->
  Qwen, can resume translation/review stages, and invalidates stale source,
  model, planner, or Timeline state.
- Defined Translation Briefs as source-language semantic context plus bilingual
  character/terminology mappings. Target-agnostic Timeline prompts consume only
  that source-semantic projection. Briefs no longer generate character
  descriptions, audience guesses, or pre-translation ASR interpretations;
  confidence-gated language checks and prompt versions reject incompatible
  derived state without over-validating short terms or names.
- Kept user terminology in the single `--glossary` interface. Task glossary
  mappings override conflicting manual characters and automatically inferred
  Brief entries, with one resolved state shared by prompts, validation, and
  reports.

### Fixed

- Made glossary compliance occurrence-based: repeated source terms now require
  independent target occurrences and are counted separately. Case-insensitive
  conflicts are case-folded consistently, and CJK phrases can span subtitle
  lines without an artificial space.
- Added explicit edit-round lifecycle states. Model/no-patch failures retry the
  same round after resume, a failed chunk rolls back every patch in that round,
  and Restore round 0 now always returns the original Hy-MT2 draft before
  deterministic repair.
- Made standalone edit progress durable before model work and after every
  completed parent chunk. Incomplete work is retained, completed work is the
  only state eligible for cleanup, and a committed checkpoint can finish safely
  if subtitle replacement was interrupted.
- Normalized `%`, `percent`, `per cent`, full-width percent signs, and `百分之`
  in deterministic checks without confusing printf placeholders such as `%s`.
- Resolved task and Brief terminology once before prompt rendering, preventing
  duplicate entries and preserving task precedence. Targeted retranslation now
  builds preceding context with a bounded reverse scan instead of reconstructing
  the full translation history for every chunk.
- Made `GlossaryOptions.report_matches` control report/session match details
  while retaining compliance metrics and issues. Removed the unimplemented
  `EditConfig.readability_checks` field until a real readability validator is
  available.
- Reworked Chunk Planner v2 so equal timestamp gaps prefer the latest valid
  boundary instead of producing long runs of single-line chunks. Every split
  and tail decision now preserves the line and token caps, and planner version
  and signature changes invalidate incompatible translation/review progress.
- Added a dedicated high-risk revision validator. Blank, multiline, tagged,
  fenced, JSON-shaped, or internal-protocol revisions are rejected; the
  Hy-MT2 draft is retained and that review chunk remains incomplete. Checkpoint
  replay also validates chunk coverage, reviewed/failed indexes, result IDs,
  Timeline alignment, protocol version, and raw/final translation counts.

### Verified

- Verified the full pytest suite: `425 passed, 25 skipped`, plus 261 subtests.
- Verified Ruff linting, touched-file formatting, touched-module Pyright, and
  checkpoint atomicity tests.
- Verified real local Normal and Pro flows with Qwen3.5 9B and Hy-MT2 7B Q6_K,
  including relaxed optimization, source-language Brief/Timeline semantics,
  orderly staged model shutdown, exact line/timestamp preservation, and no
  delimiter or internal-protocol leakage. A 31-line Pro run produced aligned
  `30 + 1` planner chunks and complete raw/final/review IDs.
- Verified a complete manual Normal Brief without configuring or loading a
  context model; the short local smoke loaded only the Hy-MT2 server and
  completed in 15.64 seconds.
- Compared all four modes on the same 15-line, three-scene local sample. Fast,
  Normal, Normal Plus, and Pro completed in 21.27s, 42.21s, 70.56s, and 99.80s
  with 1, 2, 3, and 3 model loads respectively; every mode preserved all 15
  timestamps and subtitle lines without protocol leakage.
- Verified the editing loop on character titles, technical terms, amounts,
  dates, and colloquial ambiguity. Normal Plus/Pro achieved 100% glossary and
  exact line/timeline preservation while reporting unresolved findings. The
  independent retranslate/review/verify/Restore paths preserved all unselected
  lines and restored the expected round-0 and round-1 snapshots.

## OpenLRC Mac 0.2.2

Hy-MT2 three-mode contextual translation pipeline release.

### Added

- Added Hy-MT2 `fast`, `context`, and `context-plus` modes. Context modes use an
  explicitly configured general model to produce a structured Translation Brief;
  context-plus scans every translated chunk and only replaces high-risk lines.
- Added `HyMT2Mode` and `ContextLLMConfig` with online providers and managed local
  Qwen3.5 9B support.
- Added structured Translation Briefs for document summary, character names,
  glossary, tone/style, target audience, and ASR ambiguities, with user glossary
  entries taking precedence.
- Added staged local model execution: general model -> unload -> Hy-MT2, followed
  by Hy-MT2 -> unload -> general-model review in context-plus. Only one
  OpenLRC-owned model remains resident at a time.
- Added a conservative Hy-MT2 high-risk review protocol. Failed review chunks keep
  the Hy-MT2 draft, continue processing, and mark the result `review_incomplete`.
- Added versioned Hy-MT2 checkpoints containing the source fingerprint, mode,
  Brief, raw Hy-MT2 translations, final translations, fallback metrics, review
  progress, and risk results.

### Changed

- Removed lean/standard engine selection from the Hy-MT2 public surface and kept
  only its three product modes. The former standard pipeline is now described as
  classic; online translation and general local Qwen default to classic.
- Switched Hy-MT2 delimiters to exact ID alignment and rejected duplicate,
  unexpected, or out-of-order IDs. Small exact-ID gaps may still use atomic fill;
  Hy-MT2 no longer uses `±3` fuzzy alignment.
- Preserved the Translation Brief, glossary, recent translations, and two source
  lines on either side during binary split and atomic fallback.
- Recorded retry counts, split depth, atomic IDs, validator issues, and mode in
  checkpoints; context-plus can resume unfinished review chunks without translating
  the document again.
- Made checkpoint writes atomic through a same-directory temporary file and replace.
- Exposed incomplete review chunks in CLI results and allowed completed subtitle
  tasks to resume only failed review chunks without reloading Hy-MT2.
- Made successful `run` workflows clear per-input temporary files by default;
  `--keep-temp` retains them, and incomplete review checkpoints are always kept.
  Standalone `translate` removes completed checkpoints unless `--keep-checkpoint`
  is provided.
- Added `--hy-mt2-mode`, `--context-provider`, `--context-model`,
  `--context-base-url`, and `--context-fee-limit` CLI options.
- Updated the README and CLI reference for the new Hy-MT2 modes.

### Verified

- Verified the full pytest suite: `278 passed, 25 skipped`.
- Verified Hy-MT2 delimiter translator tests: `65 passed, 5 skipped`.
- Verified Ruff linting, touched-file formatting, and touched-module Pyright.
- Verified real CLI smokes for `fast`, fully local `context`, and fully local
  `context-plus` with Hy-MT2 7B Q6_K / Qwen3.5 9B. Output line counts matched and
  no delimiter, anchor, or review JSON leaked into subtitles.

## OpenLRC Mac 0.2.1

Initial Hy-MT2 local translation profile release.

### Added

- Added Hy-MT2 local LLM profiles for `hy-mt2-7b` and
  `hy-mt2-30b-a3b`.
- Added the default Hy-MT2 7B Q6_K GGUF download profile using
  `tencent/Hy-MT2-7B-GGUF` / `HY-MT2-7B-Q6_K.gguf`.
- Added Hy-MT2 CLI support through `--local-model-profile`, including
  `openlrc setup llama --local-model-profile hy-mt2-7b` and local
  `run` / `translate` workflows.
- Added `TranslationConfig.local_hy_mt2_7b(...)`,
  `TranslationConfig.local_hy_mt2(...)`, and `LRCer.local_hy_mt2(...)`.
- Added a Hy-MT2 delimiter prompt pipeline for lean local translation using
  `<seg id="N">...</seg>` input/output blocks.

### Changed

- Mapped the Hy-MT2 official sampling parameters into the real local
  `GPTBot` call path, including `temperature`, `top_p`, `top_k`,
  `repeat_penalty`, and `max_tokens`.
- Kept Qwen as the default bare `LRCer.local()` / `--translation local` model
  while allowing Hy-MT2 to be selected explicitly by profile.
- Updated lean translation so prompt-specific parsers and retry instructions
  live on the prompter instead of being hard-coded to the `#id` anchor format.
- Improved anchor fallback parsing to tolerate `#<1>` style model output.

### Verified

- Verified Hy-MT2 profile selection, model resource resolution, and the delimiter
  prompt/parser path.
- Verified a real local Hy-MT2 7B Q6_K translation probe.

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
