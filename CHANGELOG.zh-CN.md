# OpenLRC Mac 更新日志

这个文件只记录 OpenLRC Mac fork 自己的变更。英文版见
[CHANGELOG.md](CHANGELOG.md) 中的 OpenLRC Mac 条目；两个文件需要同步更新。

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
