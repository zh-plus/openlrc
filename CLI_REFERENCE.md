# OpenLRC Mac CLI Reference

本文档记录当前第一阶段 CLI 的全部命令和行为。CLI 使用 `Typer + Rich`
实现，入口来自 `pyproject.toml` 的 `[project.scripts]`：

```shell
openlrc
openlrc-mac
```

两个入口等价。下面示例统一使用 `uv run openlrc ...`。

## 总览

```shell
uv run openlrc --help
uv run openlrc --version
uv run openlrc doctor [--strict]
uv run openlrc models status
uv run openlrc setup whisper [OPTIONS]
uv run openlrc setup llama [OPTIONS]
uv run openlrc setup all [OPTIONS]
uv run openlrc transcribe PATH... [OPTIONS]
uv run openlrc translate JSON... --translation local|online [OPTIONS]
uv run openlrc run PATH... [OPTIONS]
```

当前 CLI 是普通命令行工具，不是菜单式 TUI。裸 `openlrc` 会显示 help。

## 基础命令

### `openlrc --version`

显示 OpenLRC Mac 版本、发行包名和上游 OpenLRC 基线版本。

```shell
uv run openlrc --version
```

当前输出类似：

```text
OpenLRC Mac 0.2.0 (distribution: openlrc-mac; upstream base: OpenLRC 1.7.0a1)
```

### `openlrc doctor`

检查本机环境、submodule、二进制和默认模型是否可用。

```shell
uv run openlrc doctor
uv run openlrc doctor --strict
```

检查内容包括：

- `ffmpeg`
- `cmake`
- `vendor/whisper.cpp`
- `whisper-cli`
- 默认 Whisper 模型
- 默认 VAD 模型
- `vendor/llama.cpp`
- `llama-server`
- `llama-cli`
- 默认 Qwen GGUF 模型

默认模式只展示状态。`--strict` 会在有缺失项时返回非零退出码，适合 CI 或脚本检查。

### `openlrc models status`

只查看默认模型安装状态，不下载、不删除模型。

```shell
uv run openlrc models status
```

检查内容包括：

- 默认 Whisper 模型
- 默认 Whisper VAD 模型
- 默认本地 Qwen GGUF 模型

## Setup 命令

### `openlrc setup whisper`

初始化/构建 `whisper.cpp`，并下载 Whisper 和 VAD 模型。

```shell
uv run openlrc setup whisper
```

常用参数：

```shell
uv run openlrc setup whisper --model small
uv run openlrc setup whisper --model base --vad-model silero-v6.2.0
uv run openlrc setup whisper --model-dir "/path/to/models"
uv run openlrc setup whisper --skip-build
uv run openlrc setup whisper --skip-models
```

参数说明：

- `--model`: whisper.cpp 模型名，默认 `base`。
- `--vad-model`: VAD 模型名，默认 `silero-v6.2.0`。
- `--model-dir`: 模型下载目录。
- `--skip-build`: 初始化 submodule，但跳过 CMake build。
- `--skip-models`: 跳过模型下载。

默认模型目录：

```text
~/Library/Application Support/OpenLRC/models/
```

### `openlrc setup llama`

初始化/构建 `llama.cpp`，并下载默认本地翻译模型。

```shell
uv run openlrc setup llama
```

默认模型：

```text
repo: unsloth/Qwen3.5-9B-GGUF
file: Qwen3.5-9B-Q4_K_M.gguf
```

常用参数：

```shell
uv run openlrc setup llama --skip-build
uv run openlrc setup llama --skip-models
uv run openlrc setup llama --force
uv run openlrc setup llama --model-dir "/path/to/llm-models"
uv run openlrc setup llama --model-repo "org/repo" --model-file "model.gguf"
uv run openlrc setup llama --model-url "https://example.com/model.gguf"
```

参数说明：

- `--model-repo`: Hugging Face GGUF repo。
- `--model-file`: GGUF 文件名。
- `--revision`: Hugging Face revision，默认 `main`。
- `--model-url`: 直接下载 URL。
- `--model-dir`: GGUF 模型下载目录。
- `--skip-build`: 初始化 submodule，但跳过 CMake build。
- `--skip-models`: 跳过模型下载。
- `--force`: 即使模型已存在也重新下载。

默认 LLM 模型目录：

```text
~/Library/Application Support/OpenLRC/models/llm/
```

### `openlrc setup all`

同时运行 whisper 和 llama setup。

```shell
uv run openlrc setup all
uv run openlrc setup all --skip-build
uv run openlrc setup all --skip-models
```

## 工作流命令

### `openlrc transcribe`

只做预处理和转写，输出转写 JSON。它不会生成最终 `.lrc` / `.srt` 字幕。

```shell
uv run openlrc transcribe input.mp4 --src-lang en
```

常用参数：

```shell
uv run openlrc transcribe input.mp4 input2.wav --src-lang en
uv run openlrc transcribe input.mp4 --whisper-model small
uv run openlrc transcribe input.mp4 --vad-model ""
uv run openlrc transcribe input.mp4 --noise-suppress
uv run openlrc transcribe input.mp4 --skip-preprocess
```

参数说明：

- `PATH...`: 一个或多个音频/视频文件。
- `--src-lang`: 源语言代码；不传则由底层流程自动处理。
- `--whisper-model`: Whisper 模型名、文件名或路径，默认 `base`。
- `--vad-model`: VAD 模型名、文件名或路径；传空字符串可禁用 VAD。
- `--noise-suppress`: 转写前启用降噪。
- `--skip-preprocess`: 使用已经存在的预处理音频文件。

适合场景：

- 调试 Whisper 转写。
- 分阶段处理，先拿转写 JSON，之后再单独翻译。
- 不想直接生成字幕成品时保留中间结果。

### `openlrc translate`

读取已有转写 JSON，执行翻译和字幕导出。

```shell
uv run openlrc translate preprocessed/input_preprocessed_transcribed.json --translation local --target-lang zh-cn
```

参数说明：

- `JSON...`: 一个或多个 `transcribe` 生成的转写 JSON。
- `--translation local|online`: 翻译后端，必填。
- `--target-lang`: 目标语言，默认 `zh-cn`。
- `--bilingual-sub`: 同时生成双语字幕。
- `--llama-model`: 本地 GGUF 模型 alias、文件名或路径，默认 `qwen3.5-9b`。
- `--llama-port`: 本地 `llama-server` 端口，默认 `8088`。
- `--idle-timeout`: OpenLRC 自己启动的本地 server 空闲关闭时间，默认 `300` 秒。
- `--translate-mode lean|standard`: 翻译策略，默认 `lean`。

`--translation local` 会使用本地 `llama.cpp` / Qwen。`--translation online`
会使用现有在线 LLM 配置，需要用户自己设置对应 provider 的 API key。

### `openlrc run`

完整字幕工作流。默认只生成原语言字幕，不翻译。

```shell
uv run openlrc run input.mp4 --src-lang en
```

等价行为：

```text
preprocess -> transcribe -> post-process -> export .srt/.lrc
```

启用本地翻译：

```shell
uv run openlrc run input.mp4 --src-lang en --target-lang zh-cn --translation local
```

启用在线翻译：

```shell
uv run openlrc run input.mp4 --src-lang en --target-lang zh-cn --translation online
```

参数说明：

- `PATH...`: 一个或多个音频/视频文件。
- `--translation none|local|online`: 翻译后端，默认 `none`。
- `--src-lang`: 源语言代码。
- `--target-lang`: 目标语言，默认 `zh-cn`。
- `--whisper-model`: Whisper 模型名、文件名或路径。
- `--vad-model`: VAD 模型名、文件名或路径；传空字符串可禁用 VAD。
- `--noise-suppress`: 转写前启用降噪。
- `--bilingual-sub`: 翻译时生成双语字幕。
- `--clear-temp`: 成功后清理预处理临时文件。
- `--skip-preprocess`: 使用已经存在的预处理音频文件。
- `--llama-model`: 本地 GGUF 模型 alias、文件名或路径。
- `--llama-port`: 本地 `llama-server` 端口，默认 `8088`。
- `--idle-timeout`: OpenLRC 自己启动的本地 server 空闲关闭时间，默认 `300` 秒。
- `--translate-mode lean|standard`: 翻译策略，默认 `lean`。

`run` 和 `transcribe` 的核心区别：

- `transcribe` 只输出转写 JSON。
- `run --translation none` 输出原语言 `.lrc` / `.srt` 成品字幕。
- `run --translation local` 输出本地翻译后的 `.lrc` / `.srt` 成品字幕。

## 本地 llama-server 生命周期

当使用 `--translation local` 时，OpenLRC 会启动或复用本地 `llama-server`。

当前行为：

- 默认地址：`http://127.0.0.1:8088/v1`。
- 如果 8088 上已有匹配 model alias 的 server，OpenLRC 会复用它。
- 如果 server 是 OpenLRC 自己启动的，CLI workflow 会在命令结束时主动关闭。
- Python API 长生命周期对象还有 idle timeout 兜底，默认 300 秒。
- 如果 server 是用户外部手动启动的，OpenLRC 不会关闭它。

当前 CLI 还没有单独的 `server stop` 或 `llama stop` 命令。

## 常用环境变量

Whisper 资源覆盖：

```shell
export OPENLRC_WHISPER_CLI="/path/to/whisper-cli"
export OPENLRC_WHISPER_MODEL="/path/to/ggml-base.bin"
export OPENLRC_WHISPER_VAD_MODEL="/path/to/ggml-silero-v6.2.0.bin"
export OPENLRC_WHISPER_MODEL_DIR="$HOME/Library/Application Support/OpenLRC/models"
```

llama.cpp 资源覆盖：

```shell
export OPENLRC_LLAMA_SERVER="/path/to/llama-server"
export OPENLRC_LLAMA_CLI="/path/to/llama-cli"
export OPENLRC_LLAMA_MODEL="/path/to/Qwen3.5-9B-Q4_K_M.gguf"
export OPENLRC_LLAMA_MODEL_DIR="$HOME/Library/Application Support/OpenLRC/models/llm"
```

在线翻译常见 API key：

```shell
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```
