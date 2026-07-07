# OpenLRC Mac To Do List

本文件根据 README 中的项目目标和当前 0.2.0 进度整理。它不是发布承诺，而是这个
macOS fork 接下来要推进的功能清单和工程工作。

## 当前基线

- [x] 保留上游 OpenLRC 的字幕切分、优化、翻译和预处理 pipeline。
- [x] 将默认转写路径切换到本地 `whisper.cpp`。
- [x] 使用 Git submodule 管理 `vendor/whisper.cpp`。
- [x] 提供 `scripts/setup_whisper_cpp.py` 构建 `whisper-cli` 并下载默认模型。
- [x] 支持从 submodule build、app bundle resources、显式配置、环境变量和 `PATH`
      解析 `whisper-cli`。
- [x] 支持从默认模型目录、环境变量和显式配置解析 Whisper / VAD 模型。
- [x] 支持本地音频和视频转写，并生成 `.lrc` / `.srt` 字幕。
- [x] 保留现有 LLM 翻译能力。
- [x] 添加 `vendor/llama.cpp` submodule，支持本地 `llama-server` 翻译。
- [x] 提供 `scripts/setup_llama_cpp.py` 构建 `llama-server` / `llama-cli` 并下载默认 Qwen GGUF。
- [x] 提供 `LRCer.local()` 作为本地转写 + 本地翻译的短入口。
- [x] 提供第一阶段 `openlrc` CLI，覆盖 doctor、setup、models status、transcribe、
      translate 和 run。
- [x] 将发行包元数据独立为 `openlrc-mac` / `OpenLRC Mac 0.2.0`，同时保留
      `import openlrc` 和 `openlrc` CLI 兼容入口。
- [x] 删除上游遗留 Streamlit GUI 和 `openlrc gui` 入口。
- [x] 删除 faster-whisper / CUDA 时代的主要配置遗留。
- [x] 将 `whisper.cpp` submodule 更新到 `v1.9.1`。
- [x] 维护英文 `CHANGELOG.md` 和中文 `CHANGELOG.zh-CN.md`。

## P0 - 稳定本地转写后端

- [ ] 修复或验证 `WhisperCLIBackend` 对长时间转写输出的 stdout / stderr 处理，避免
      子进程输出量较大时出现阻塞风险。
- [ ] 为 `whisper.cpp` JSON 输出解析增加更多 fixture，覆盖正常输出、空结果、异常
      JSON、缺少 word timestamp、不同语言等情况。
- [ ] 明确 native VAD 的行为边界：启用、禁用、模型缺失、参数错误时都要有清楚的错误
      信息。
- [ ] 改进真实转写 smoke test 的说明，覆盖音频文件和视频文件两种路径。
- [ ] 清理真实转写测试生成的临时文件规则，确保 `preprocessed/`、中间 JSON、手动媒体
      文件不会误提交。
- [ ] 确认 `whisper.cpp v1.9.1` 的关键 CLI 参数在当前 backend 中仍然稳定可用。

## P0 - 开发体验和测试卫生

- [ ] 处理仍未格式化的历史文件，让
      `uv run ruff format --check openlrc/ tests/` 可以全量通过。
- [ ] 逐步修复 `uv run pyright openlrc/` 中的既有类型问题，优先处理
      `chatbot.py`、`media_utils.py`、`preprocess.py`、`prompter.py`。
- [ ] 保留 `tests/test_lazy_imports.py` 对 `faster_whisper` 的 forbidden import 检查。
- [ ] 补充资源解析测试，覆盖 app bundle resource、环境变量、默认模型目录和 submodule
      build 的优先级。
- [ ] 补充 `scripts/setup_whisper_cpp.py` 的 dry-run 或 mock 测试，避免只依赖真实下载和
      真实构建验证。

## P1 - 模型和资源管理

- [ ] 提供可复用的模型清单，包含模型名、文件名、下载地址、大小、推荐用途和是否已安装。
- [ ] 支持检查模型安装状态，并给出缺失模型的清楚提示。
- [ ] 支持下载、重新下载、删除和定位模型文件。
- [ ] 为 `tiny`、`base`、`small`、`medium`、`large-v3-turbo` 等模型提供更清楚的默认
      选择建议。
- [ ] 为本地 LLM 模型提供清单和安装状态，默认覆盖
      `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf`。
- [ ] 为 `llama-server` 增加更可见的日志和错误提示，特别是端口占用、模型缺失和内存不足。
- [ ] 评估 `Qwen3.5-9B-Q4_K_M.gguf` 在不同 Apple Silicon 机器上的内存、速度和翻译质量。
- [ ] 设计未来 macOS app 的模型目录策略，默认继续使用
      `~/Library/Application Support/OpenLRC/models/`。
- [ ] 避免在功能代码中新增硬编码的本机绝对路径，统一通过 `openlrc.whisper_resources`
      和 `openlrc.llama_resources` 解析。

## P1 - 命令行和开发工具

- [x] 评估并重新提供非 GUI 的 CLI，例如 `setup-whisper`、`transcribe`、
      `translate` 或 `run`。
- [x] 如果新增 CLI，只围绕当前 `whisper.cpp` 实现设计，不恢复旧的 Streamlit GUI 入口。
- [x] 为常用本地流程整理明确命令：初始化 submodule、构建 whisper.cpp / llama.cpp、
      下载模型、真实转写/翻译 smoke test、运行测试。
- [ ] 为第一阶段 CLI 继续补充更完整的真实场景 smoke test 文档和错误示例。
- [ ] 设计第二阶段 interactive CLI / TUI，不让第一阶段 Typer 命令承担菜单界面职责。
- [ ] 保持 `uv` 作为唯一依赖和运行入口，不引入 pip、Poetry 或 Conda 工作流。

## P2 - macOS GUI

- [ ] 选择 GUI 技术路线，并确认它能稳定调用现有 Python pipeline。
- [ ] 提供文件选择：支持音频和视频文件。
- [ ] 提供输出格式选择：`.lrc`、`.srt`，后续可扩展更多格式。
- [ ] 提供 Whisper 模型选择，并显示模型是否已安装。
- [ ] 提供本地翻译模型选择，并显示 Qwen GGUF 是否已安装。
- [ ] 提供模型下载或定位入口，避免用户手动编辑路径。
- [ ] 显示转写、字幕优化和翻译进度。
- [ ] 提供可读的错误信息和日志入口。
- [ ] 支持预览生成的 `.srt` / `.lrc` 文件。
- [ ] 保留高级路径覆盖能力，但不要把它放在主流程中心。
- [ ] 支持取消任务，并确保取消后子进程和临时文件状态可控。

## P2 - Pipeline 边界重构

- [ ] 将媒体预处理、转写、字幕后处理、翻译、输出写入拆成更清楚的服务边界。
- [ ] 为 GUI 和 CLI 提供共享的 job runner，避免界面层复制 pipeline 逻辑。
- [ ] 统一进度事件结构，让 CLI、GUI 和测试都可以复用。
- [ ] 统一错误类型，把资源缺失、模型缺失、ffmpeg 失败、whisper.cpp 失败和 LLM 失败
      区分清楚。
- [ ] 保持与上游 OpenLRC API 的实际兼容性，只有确认不再支持的旧入口才删除。

## P2 - 文档和发布维护

- [ ] README 继续保持诚实：这是 macOS `whisper.cpp` 开发 fork，不是成熟终端应用。
- [ ] 保持 `CHANGELOG.md` 和 `CHANGELOG.zh-CN.md` 同步更新。
- [ ] 为普通用户补充“从零开始”的 macOS 安装、模型下载和转写教程。
- [ ] 为开发者补充 submodule 更新流程和验证清单。
- [ ] 记录 manual test 的本地约定：`manual tests/` 目录被 Git 忽略，只用于本机媒体
      实验。

## 暂缓或不做

- [ ] 暂不恢复上游 Streamlit GUI。
- [ ] 暂不恢复 faster-whisper / CUDA 配置入口。
- [ ] 暂不把模型文件、`whisper.cpp` build output、手动测试媒体放入 Git。
- [ ] 暂不把 `llama.cpp` build output 或 GGUF 模型文件放入 Git。
- [ ] 暂不推进 macOS `.app` 打包基础，包括 app bundle 资源布局、签名、notarization 和
      打包版依赖边界。
- [ ] 暂不承诺通用 PyPI 包体验，优先服务 macOS 本地 `whisper.cpp` 路线。
- [ ] 暂不重写完整上游 pipeline，先围绕转写后端、资源解析、模型管理和 GUI 编排逐步
      收敛。
