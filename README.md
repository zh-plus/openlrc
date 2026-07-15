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
- A first-stage `openlrc` / `openlrc-mac` CLI for setup, diagnostics, model
  status, transcription, translation, and full subtitle generation.

The distribution name is `openlrc-mac`. The Python import package remains
`openlrc` for compatibility with upstream-style scripts.

## Status

Working:

- Build and use local `whisper.cpp` for transcription.
- Build and use local `llama.cpp` for opt-in Qwen translation.
- Generate `.lrc` and `.srt` subtitles through the existing OpenLRC pipeline.
- Run common workflows from the CLI without editing Python scripts.

Still in progress:

- macOS app packaging.
- GUI model management.
- Menu-style terminal UI.
- More polished defaults and error messages for end users.

## Requirements

- macOS, tested on Apple Silicon.
- Python `>=3.10,<3.13`.
- [uv](https://github.com/astral-sh/uv).
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`.
- CMake and Xcode Command Line Tools for building `whisper.cpp` and `llama.cpp`.

## Quick Start

```shell
git clone --recurse-submodules <repo-url>
cd openlrc_mac
uv sync
```

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
# Fully local: Qwen context brief -> unload -> Hy-MT2 translation
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b \
  --hy-mt2-mode context --context-provider local --context-model qwen3.5-9b

# Context plus: Qwen brief -> Hy-MT2 translation -> Qwen high-risk review
uv run openlrc run video.mp4 --src-lang en --target-lang zh-cn \
  --translation local --local-model-profile hy-mt2-7b \
  --hy-mt2-mode context-plus --context-provider local --context-model qwen3.5-9b
```

Only one local LLM is resident during staged context modes. Hy-MT2 exposes only
`fast`, `context`, and `context-plus`; its internal translator engine is not a
user-facing option. Online translation and the general local Qwen profile use
the classic OpenLRC context-aware pipeline by default.

By default, `openlrc run` does not translate. Translation must be explicitly
enabled with `--translation local` or `--translation online`.
Temporary preprocessing/checkpoint files are removed after complete success.
Use `--keep-temp` to retain them. An incomplete context-plus review always keeps
its checkpoint so the next run can resume failed review chunks.
The standalone `translate` command removes a completed checkpoint by default;
use `--keep-checkpoint` when the compare JSON is needed for debugging.

## Python API

The Python API remains available for scripts and future app orchestration:

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.run("video.mp4", src_lang="en", skip_trans=True)
```

Local translation shortcut:

```python
from openlrc import LRCer

lrcer = LRCer.local()
lrcer.run("video.mp4", src_lang="en", target_lang="zh-cn")
```

Hy-MT2 local translation shortcut:

```python
from openlrc import ContextLLMConfig, HyMT2Mode, LRCer

lrcer = LRCer.local_hy_mt2()
lrcer.run("video.mp4", src_lang="en", target_lang="zh-cn")

context_llm = ContextLLMConfig.local_qwen35_9b()
lrcer = LRCer.local_hy_mt2(mode=HyMT2Mode.CONTEXT, context_llm=context_llm)
```

If OpenLRC starts the local `llama-server`, it closes it when the workflow ends.
Longer-lived Python objects also have an idle timeout fallback.

## More Documentation

- [CLI_REFERENCE.md](CLI_REFERENCE.md): all current CLI commands, options, and
  behavior.
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
