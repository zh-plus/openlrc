# OpenLRC Mac

OpenLRC Mac is a macOS-focused fork of
[zh-plus/openlrc](https://github.com/zh-plus/openlrc). It turns local audio or
video files into `.lrc` or `.srt` subtitles.

This fork keeps the upstream subtitle pipeline, but focuses on local-first macOS
workflows:

- Local transcription with [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  and Metal acceleration.
- Optional local translation with [llama.cpp](https://github.com/ggml-org/llama.cpp)
  and a Qwen GGUF model.
- An `openlrc` / `openlrc-mac` CLI for setup, diagnostics, model status,
  transcription, translation, glossary compliance, and transactional editing.
- A keyboard-first Textual TUI v2 with mouse-equivalent actions for workflows,
  jobs and recovery, models and setup, diagnostics, and settings, with two
  runtime themes and English/Simplified Chinese presentation.
- A typed in-process Workflow API for transcription, the five canonical
  translation modes, full runs, progress events, cancellation, and structured
  results shared by the CLI, TUI, and future GUI.

The distribution name is `openlrc-mac`. The Python import package remains
`openlrc` for compatibility with upstream-style scripts.

## Status

Working:

- Build and use local `whisper.cpp` for transcription.
- Build and use local `llama.cpp` for opt-in Qwen translation.
- Generate `.lrc` and `.srt` subtitles through the existing OpenLRC pipeline.
- Run common workflows from the CLI without editing Python scripts.
- Run `Transcribe`, `Translate`, and `Run` through one lifecycle-managed
  Workflow layer without duplicating the `LRCer` pipeline.
- Load versioned task glossaries, report deterministic compliance, and run
  auditable line-scoped subtitle edits without repeating ASR.
- Supply a complete or partial Translation Brief, locking known story,
  character-name, and tone/style guidance while inference fills only omissions.
- Run the Textual TUI v2 through `openlrc tui` or `openlrc-tui`; its initial
  release supports New Work, Jobs, Models, and Settings while keeping Edit
  Subtitle visibly disabled for a later typed Edit integration. Appearance
  changes preview immediately and support persisted English or Simplified
  Chinese UI text, two distinct dark themes, and a static blue Logo when motion
  is disabled.

Still in progress:

- macOS app packaging.
- GUI model management.
- More polished defaults and error messages for end users.

## Requirements

- macOS, tested on Apple Silicon.
- Python `>=3.11,<3.15`; the repository default is Python 3.13.14 through
  `.python-version`.
- [uv](https://github.com/astral-sh/uv).
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`.
- CMake and Xcode Command Line Tools for building `whisper.cpp` and `llama.cpp`.

## Quick Start

```shell
git clone --recurse-submodules <repo-url>
cd openlrc_mac
uv sync
```

`uv sync` creates the project environment without changing the `python3`
selected by your shell. Development and CI use the `dev` dependency group;
LiteLLM support is installed explicitly with `uv sync --group dev --extra
litellm`.

Set up local transcription:

```shell
uv run openlrc setup whisper
```

Optional: set up local Qwen translation:

```shell
uv run openlrc setup llama
```

Check the local environment:

```shell
uv run openlrc doctor
uv run openlrc models status
```

Launch the interactive TUI:

```shell
uv run openlrc tui
# equivalent standalone entrypoint
uv run openlrc-tui
```

Generate subtitles without translation:

```shell
uv run openlrc run video.mp4 --src-lang en
```

Generate locally translated subtitles:

```shell
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn --translation local
```

Use the Hy-MT2 7B Q6_K local translation profile:

```shell
uv run openlrc setup llama --local-model-profile hy-mt2-7b
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn --translation local --local-model-profile hy-mt2-7b
```

Hy-MT2 defaults to `fast` mode (delimiter translation, no Context Review).
Context modes require an explicit general-purpose model:

```shell
# Normal: Qwen context brief -> unload -> Hy-MT2 translation
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b \
  --hy-mt2-mode normal --context-provider local --context-model qwen3.5-9b

# Normal Plus: Qwen brief -> Hy-MT2 deterministic repair -> Qwen semantic edit
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b \
  --hy-mt2-mode normal-plus --context-provider local --context-model qwen3.5-9b

# Pro: Qwen brief + per-chunk timeline -> Hy-MT2 -> Qwen high-risk review
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b \
  --hy-mt2-mode pro --context-provider local --context-model qwen3.5-9b
```

Only one local LLM is resident during staged context modes. Hy-MT2 exposes
`fast`, `normal`, `normal-plus`, and `pro`; the deprecated `context` and
`context-plus` values are accepted as aliases for one compatibility release.
Its internal translator engine is not a user-facing option. Pro precomputes a bounded per-chunk ContextTimeline
with the same chunk boundaries used for translation and review. Online
translation and the general local Qwen profile use the classic OpenLRC
context-aware pipeline by default.

Hy-MT2 context preparation keeps a concise source-language summary, optional
tone/style guidance, and Timeline state in the source language. Only character
names and glossary terms carry source-to-target mappings. Character descriptions,
target-audience guesses, and pre-translation ASR interpretations are excluded
from the Brief and Hy-MT2 prompt. Control instructions remain English and
Timeline generation is target-agnostic.

If you already know the work, you can supply all or part of the Translation
Brief instead of relying entirely on model inference:

```shell
uv run openlrc run movie.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b --hy-mt2-mode normal \
  --brief-summary "A dry workplace comedy about a failed product launch." \
  --brief-characters '@characters.json' \
  --brief-tone-style "Keep the dialogue understated and conversational."
```

`--brief-characters` accepts an inline JSON array or `@PATH` to a UTF-8 JSON
file, for example
`[{"source_name":"John","target_name":"强尼"}]`. Supplied fields are
locked; a context model fills only missing fields. Supplying summary,
characters, and tone/style makes the Brief fully manual and skips automatic
Brief and terminology extraction. User terminology remains a separate task
glossary supplied through `--glossary`; it takes precedence over conflicting
manual or automatically inferred character mappings and is injected only once.

Subtitle cleanup defaults to the historical `aggressive` profile. Use the
alignment-safe `relaxed` profile to preserve short/repeated lines, `<unk>`
segments, line count, ordering, and timestamps:

```shell
uv run openlrc run video.mp4 --translation local \
  --subtitle-optimization relaxed
```

By default, `openlrc run` does not translate. Translation must be explicitly
enabled with `--translation local` or `--translation online`.
Temporary preprocessing/checkpoint files are removed after complete success.
Use `--keep-temp` to retain them. An incomplete Normal Plus or Pro edit always
keeps its checkpoint so the next run can resume unfinished chunks.
The standalone `translate` command removes a completed checkpoint by default;
use `--keep-checkpoint` when the compare JSON is needed for debugging.

Task glossaries accept the legacy JSON mapping or the versioned catalog schema.
They are never applied by blind text replacement: OpenLRC checks the translated
output and reports unresolved required terms instead.

```shell
uv run openlrc glossary validate glossary.json --source-language en --target-language zh-cn
uv run openlrc translate preprocessed/input_preprocessed_transcribed.json \
  --translation local --local-model-profile hy-mt2-7b --hy-mt2-mode normal-plus \
  --context-provider local --context-model qwen3.5-9b \
  --glossary glossary.json --force-glossary --edit-rounds 2 --enable-restore
```

Independent editing consumes source and translated JSON directly and does not
rerun transcription. `verify` and `restore` are offline; `review` requires an
explicit context model; `retranslate` requires an explicit Hy-MT2 model.

```shell
uv run openlrc edit --source source.json --target translated.json --action verify
uv run openlrc edit --source source.json --target translated.json \
  --action review --ids 12,18-24 --context-provider local --context-model qwen3.5-9b
uv run openlrc edit --source source.json --target translated.json \
  --action restore --round 0 --session translated.edit-session.json
```

## Python API

The Python API remains available for scripts and app orchestration. Existing
`LRCer` entry points remain compatible:

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.run("video.mp4", src_lang="en", skip_trans=True)
```

New integrations should use the typed Workflow API when they need progress,
cancellation, structured errors, artifact discovery, or consistent cleanup:

```python
from openlrc.workflow import (
    ExecutionContext,
    TranscribeRequest,
    WorkflowExecutor,
    WorkflowKind,
)

context = ExecutionContext(
    WorkflowKind.TRANSCRIBE,
    event_sink=lambda event: print(event),
)
result = WorkflowExecutor().execute(
    TranscribeRequest(("video.mp4",), src_lang="en"),
    context,
)
print(result.status, result.outputs)
```

Each `ExecutionContext` represents one job and cannot be reused after execution
or closure. `WorkflowResult.artifacts` contains only files retained when the job
finishes; `ArtifactCreatedEvent` remains the chronological record of files that
were created during execution, including temporary Run artifacts later cleaned
up after success.

The Workflow translation surface uses only `standard`, `fast`, `normal`,
`normal-plus`, and `pro`. Existing CLI/config aliases `context` and
`context-plus` are normalized before events and results are emitted.

Local translation shortcut:

```python
from openlrc import LRCer

lrcer = LRCer.local()
lrcer.run("video.mp4", src_lang="en", target_lang="zh-cn")
```

Hy-MT2 local translation shortcut:

```python
from openlrc import ContextLLMConfig, HyMT2Mode, LRCer, TranslationBriefInput

lrcer = LRCer.local_hy_mt2()
lrcer.run("video.mp4", src_lang="en", target_lang="zh-cn")

context_llm = ContextLLMConfig.local_qwen35_9b()
lrcer = LRCer.local_hy_mt2(
    mode=HyMT2Mode.NORMAL,
    context_llm=context_llm,
    translation_brief=TranslationBriefInput(
        summary="A dry workplace comedy about a failed product launch.",
        tone_style="Keep the dialogue understated and conversational.",
    ),
)

# Existing translations can be verified without loading a model.
result = LRCer().edit("source.json", "translated.json", action="verify")
```

If OpenLRC starts the local `llama-server`, it closes it when the workflow ends.
Longer-lived Python objects also have an idle timeout fallback.

## More Documentation

- [CLI_REFERENCE.md](CLI_REFERENCE.md): all current CLI commands, options, and
  behavior.
- [TUI_REFERENCE.md](TUI_REFERENCE.md): TUI v2 pages, shortcuts, workflows,
  lifecycle, themes, languages, and initial-release boundaries.
- [TODO.md](TODO.md): living development roadmap.
- [CHANGELOG.md](CHANGELOG.md): fork changelog and inherited upstream history.

## Development

Useful checks:

```shell
uv run --with pytest python -m pytest -q
uv run ruff check openlrc/ tests/ scripts/
uv run ruff format --check openlrc/ tests/ scripts/
uv run pyright openlrc/
```

Manual media experiments belong in `manual tests/`; that directory is ignored by
Git.

## Upstream And Credits

This fork is based on [zh-plus/openlrc](https://github.com/zh-plus/openlrc).
Most of the subtitle pipeline, translation flow, and project structure come from
that upstream project.

Related projects:

- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)

## License

This fork follows the upstream OpenLRC license. See [LICENSE](LICENSE).
