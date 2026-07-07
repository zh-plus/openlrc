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

By default, `openlrc run` does not translate. Translation must be explicitly
enabled with `--translation local` or `--translation online`.

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
