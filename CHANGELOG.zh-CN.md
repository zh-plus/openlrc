# OpenLRC Mac 更新日志

这个文件只记录 OpenLRC Mac fork 自己的变更。英文版见
[CHANGELOG.md](CHANGELOG.md) 中的 OpenLRC Mac 条目；两个文件需要同步更新。

## OpenLRC Mac 0.2.0

面向 macOS fork 的本地翻译和第一阶段 CLI 版本。

### 新增

- 新增 `vendor/llama.cpp` submodule，用于本地 LLM 翻译。
- 新增可复用的 `openlrc.setup.llama_cpp` 和 `scripts/setup_llama_cpp.py`，用于构建
  `llama-server` / `llama-cli` 并下载默认 Qwen GGUF 模型。
- 新增 `openlrc.llama_resources`，统一解析 `llama-server`、`llama-cli` 和 GGUF
  模型路径，覆盖显式配置、环境变量、app resources、submodule build、默认用户模型目录和
  `PATH`。
- 新增 `openlrc.local_llm_server`，管理本地 `llama-server` 的启动、复用、健康检查、
  空闲自动关闭和手动关闭。
- 新增 opt-in 本地 Qwen 翻译入口：`LRCer.local()` 和
  `TranslationConfig.local_qwen35_9b(...)`。
- 新增 `ModelProvider.LOCAL_LLAMA`，复用现有 OpenAI-compatible `GPTBot` 路径，并将
  本地翻译费用固定为 0。
- 新增第一阶段 `openlrc` Typer/Rich CLI，覆盖 `doctor`、`models status`、`setup`、
  `transcribe`、`translate` 和 `run` 工作流。
- 新增 `openlrc-mac` CLI 别名，用于体现 fork 的独立发行身份。

### 变更

- 将 Python distribution metadata 从 `openlrc` 改为 `openlrc-mac`，同时保留
  `import openlrc` 兼容现有上游式用户代码。
- 更新 `openlrc --version`，显示 `OpenLRC Mac 0.2.0` 和上游 OpenLRC 基线版本。
- 保持 `openlrc run` 的安全默认行为：除非用户显式传入 `--translation local` 或
  `--translation online`，否则默认不翻译。
- 将 whisper.cpp setup 抽到可复用 package helper，原 setup 脚本保留为兼容 wrapper。
- 更新 README、TODO 和架构文档，补齐本地翻译流程、CLI 工作流和 fork 发行身份说明。
- 将 optional dependency 安装提示更新为 `openlrc-mac[...]`。

### 验证

- 完整 pytest 测试通过：`226 passed, 25 skipped`。
- 验证 `openlrc --version` 和 `openlrc-mac --version`。
- 验证 CLI smoke test：`doctor`、`models status` 和本地
  `run --translation local`。
- 通过 OpenLRC pipeline 验证真实 `llama.cpp` / Qwen 本地翻译。
- Ruff lint 检查通过：`openlrc/`、`tests/` 和 `scripts/`。

## OpenLRC Mac 0.1.2

本地 `whisper.cpp` 后端的依赖更新版本。

### 变更

- 将 `vendor/whisper.cpp` submodule 从 `v1.8.5` 更新到 `v1.9.1`。
- 使用更新后的 submodule 源码重新构建本地 `whisper-cli`。

### 验证

- 验证 `resolve_whisper_cli("")` 可以解析到重新构建后的本地 CLI。
- 使用 `tests/data/test_audio.wav` 完成真实本地转写 smoke test。
- 完整 pytest 测试通过：`195 passed, 25 skipped`。
- Ruff lint 检查通过：`uv run ruff check openlrc/ tests/`。

## OpenLRC Mac 0.1.1

面向 macOS `whisper.cpp` fork 的代码清理版本。

### 移除

- 移除了上游遗留的 Streamlit GUI 包：`openlrc/gui_streamlit`。
- 移除了已经不可用的 `openlrc gui` 控制台入口，不再保留无效 GUI 命令。
- 移除了未使用的 Streamlit 截图资源。
- 从 `TranscriptionConfig` 中移除了 faster-whisper / CUDA 时代的兼容字段：
  - `compute_type`
  - `device`
  - `vad_options`
- 从 `openlrc/defaults.py` 中移除了 faster-whisper 默认 ASR/VAD 参数表。
- 移除了删除旧入口后不再需要的运行时依赖：
  - `click`
  - `pip`

### 变更

- 将转写默认配置收敛到当前实际使用的 `whisper.cpp` CLI 参数。
- 更新了 `Transcriber` 构造参数，只暴露当前有效的 `whisper.cpp` 后端配置。
- 更新了 `whisper_types.py` 文档，将 Segment 和 Word 描述为 OpenLRC pipeline
  内部数据类型，不再绑定到 faster-whisper。
- 更新测试，确认已移除的 faster-whisper 字段不再被接受。

### 验证

- 完整 pytest 测试通过：`195 passed, 25 skipped`。
- Ruff lint 检查通过：`uv run ruff check openlrc/ tests/`。
- 本次触碰文件的格式检查通过。

## OpenLRC Mac 0.1.0

基于上游 OpenLRC `1.7.0a1` 的第一个 macOS 方向 fork 版本。

### 新增

- 将 `whisper.cpp` 作为 Git submodule 集成到 `vendor/whisper.cpp`。
- 新增 `scripts/setup_whisper_cpp.py`，用于初始化 submodule、以 CMake Release
  模式构建 `whisper-cli`，并下载默认 GGML 模型。
- 新增 `openlrc/whisper_resources.py`，统一解析 `whisper-cli`、Whisper 模型和
  VAD 模型路径。
- 支持 `base`、`small`、`medium`、`large-v3-turbo` 等语义化 Whisper 模型名。
- 新增本地开发和未来 macOS app 打包所需的环境变量覆盖：
  - `OPENLRC_WHISPER_CLI`
  - `OPENLRC_WHISPER_MODEL`
  - `OPENLRC_WHISPER_VAD_MODEL`
  - `OPENLRC_WHISPER_MODEL_DIR`

### 变更

- 将默认转写路径替换为本地 `whisper.cpp` CLI 转写。
- 将 `TranscriptionConfig` 默认值更新为本地语义化资源：
  - `cli_path=""`
  - `whisper_model="ggml-base.bin"`
  - `vad_model="ggml-silero-v6.2.0.bin"`
- 重写 README，使其围绕当前 fork 的 macOS、`whisper.cpp` 和未来 GUI 方向展开。
- 将 `click` 显式加入依赖，因为真实转写流程在语言模型设置时会经过 SpaCy CLI 路径。

### 验证

- 在 macOS 上成功构建 `vendor/whisper.cpp/build/bin/whisper-cli`，并启用 Metal 后端。
- 成功下载并解析位于 `~/Library/Application Support/OpenLRC/models/` 的
  `ggml-base.bin` 和 `ggml-silero-v6.2.0.bin`。
- 通过 `WhisperCLIBackend`、`Transcriber` 和 `LRCer().transcribe(...)` 验证真实本地转写。
- 完整 pytest 测试通过：`194 passed, 25 skipped`。
