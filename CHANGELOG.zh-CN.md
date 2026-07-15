# OpenLRC Mac 更新日志

这个文件只记录 OpenLRC Mac fork 自己的变更。英文版见
[CHANGELOG.md](CHANGELOG.md) 中的 OpenLRC Mac 条目；两个文件需要同步更新。

## OpenLRC Mac 0.2.2

Hy-MT2 三模式上下文翻译管线版本。

### 新增

- 新增 Hy-MT2 `fast`、`context`、`context-plus` 三种模式：
  - `fast` 直接执行 delimiter lean 翻译，不运行 Context Review；
  - `context` 先由显式配置的通用模型生成结构化 Translation Brief，再由
    Hy-MT2 翻译；
  - `context-plus` 在翻译完成后使用同一通用模型扫描全部 chunk，只修正
    high-risk 行。
- 新增 `HyMT2Mode` 和 `ContextLLMConfig`，支持在线通用模型及本地
  Qwen3.5 9B 上下文模型。
- 新增结构化 Translation Brief，覆盖摘要、人物译名、术语、语气风格、
  目标受众和 ASR 歧义；用户 glossary 与生成结果冲突时以用户配置为准。
- 新增纯本地阶段式模型运行：`context` 按“通用模型 -> 卸载 -> Hy-MT2”
  执行，`context-plus` 再卸载 Hy-MT2 并重新加载通用模型校对，任一时刻
  只驻留一个 OpenLRC-owned 模型。
- 新增 Hy-MT2 专用 high-risk review 协议。单个 review chunk 最终失败时
  保留 Hy-MT2 草稿、继续其他 chunk，并标记 `review_incomplete`。
- 新增版本化 Hy-MT2 checkpoint，保存 source fingerprint、模式、Brief、
  原始 Hy-MT2 译文、最终译文、fallback metrics、review 进度和风险结果。

### 变更

- Hy-MT2 对外取消 lean/standard engine 配置，只保留三种产品模式；原
  `standard` 管线对外改称 classic。在线翻译和通用本地 Qwen 默认使用 classic。
- Hy-MT2 delimiter 改为精确 ID 对齐，拒绝重复、未知和乱序 ID；少量精确
  ID 缺失仍允许 atomic fill，不再使用 `±3` fuzzy matching。
- binary split 和 atomic fallback 现在保留 Translation Brief、glossary、
  最近译文及目标行前后各两条源字幕。
- checkpoint 额外记录 retry、split depth、atomic IDs、validator 问题和模式，
  `context-plus` 可从未完成的 review chunk 恢复而无需重新翻译。
- checkpoint 改为同目录临时文件写完后原子替换，避免中断留下半截 JSON。
- CLI 最终结果显示 `review_incomplete` 和失败 chunk；已生成字幕的任务可在
  下次运行只补跑失败 review，而不重新加载 Hy-MT2。
- `run` 在完整成功后默认清理当前输入的临时文件；`--keep-temp` 可显式保留，
  review 不完整时会强制保留 checkpoint。独立 `translate` 命令默认删除已完成
  checkpoint，可用 `--keep-checkpoint` 保留。
- CLI 新增 `--hy-mt2-mode`、`--context-provider`、`--context-model`、
  `--context-base-url` 和 `--context-fee-limit`。
- 更新 README 和 CLI reference，补充 Hy-MT2 三模式说明。

### 验证

- 完整 pytest 测试通过：`278 passed, 25 skipped`。
- Hy-MT2 delimiter translator 测试通过：`65 passed, 5 skipped`。
- Ruff lint、touched-file format check 和 touched-module Pyright 检查通过。
- `fast`、纯本地 `context`、纯本地 `context-plus` 均通过真实 Hy-MT2 7B
  Q6_K / Qwen3.5 9B CLI smoke；输出行数严格一致，无 delimiter、anchor 或
  review JSON 泄漏。

## OpenLRC Mac 0.2.1

Hy-MT2 初始本地翻译 profile 版本。

### 新增

- 新增 `hy-mt2-7b` 和 `hy-mt2-30b-a3b` 两个本地 LLM profile。
- 新增 Hy-MT2 7B Q6_K 默认 GGUF 下载 profile，使用
  `tencent/Hy-MT2-7B-GGUF` / `HY-MT2-7B-Q6_K.gguf`。
- 新增 Hy-MT2 CLI 支持，通过 `--local-model-profile` 接入
  `openlrc setup llama --local-model-profile hy-mt2-7b` 以及本地
  `run` / `translate` 工作流。
- 新增 `TranslationConfig.local_hy_mt2_7b(...)`、
  `TranslationConfig.local_hy_mt2(...)` 和 `LRCer.local_hy_mt2(...)`。
- 新增 Hy-MT2 lean 本地翻译 delimiter prompt 管线，使用
  `<seg id="N">...</seg>` 输入/输出块。

### 变更

- 将 Hy-MT2 官方采样参数接入真实本地 `GPTBot` 调用链，包括
  `temperature`、`top_p`、`top_k`、`repeat_penalty` 和 `max_tokens`。
- 保持 Qwen 作为裸 `LRCer.local()` / `--translation local` 默认模型，同时允许用户显式选择
  Hy-MT2 profile。
- 调整 lean 翻译流程，让 prompt-specific parser 和 retry instruction 下沉到 prompter，不再硬编码为
  `#id` anchor 格式。
- 改进 anchor fallback parser，兼容 `#<1>` 这类模型输出。

### 验证

- 验证 Hy-MT2 profile 解析、模型资源解析和 delimiter prompt/parser。
- 通过真实 Hy-MT2 7B Q6_K 本地翻译 probe。

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
