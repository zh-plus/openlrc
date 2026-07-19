# OpenLRC Mac To Do List

Last verified: 2026-07-18

本文档记录 OpenLRC Mac 在 0.3.0 基线上的已完成能力、当前风险和后续优先级。
它不是发布承诺。项目仍以 macOS 本地字幕工作流为核心，优先保证转写稳定性、
翻译一致性和可恢复性。产品最终提供 CLI、TUI、GUI 三种访问模式：CLI 与 TUI
优先获得新功能，GUI 在交互和服务稳定后跟进。

## 当前基线：0.3.0

### 核心流水线

- [x] 保留上游 OpenLRC 的预处理、字幕切分、优化、翻译和输出 pipeline。
- [x] 将默认转写后端切换为本地 `whisper.cpp`。
- [x] 支持本地音频/视频转写和 `.lrc` / `.srt` 输出。
- [x] 支持 native Whisper VAD，并允许通过空配置禁用。
- [x] 支持分阶段 `pre_process`、`transcribe`、`translate`、`post_process`。
- [x] 提供 aggressive/relaxed 两种字幕优化 profile；宽松模式保持 segment
      数量、顺序和时间边界，并使用独立内部缓存。
- [x] 成功后默认只清理当前输入的临时文件，支持显式保留调试产物。

### whisper.cpp 与资源

- [x] 以 submodule 管理 `vendor/whisper.cpp`，当前 pin 为 `v1.9.1`。
- [x] 提供 `openlrc.setup.whisper_cpp` 和兼容 wrapper
      `scripts/setup_whisper_cpp.py`。
- [x] 提供 `openlrc setup whisper` CLI。
- [x] 统一解析显式配置、环境变量、app bundle、submodule build 和 `PATH`。
- [x] 统一解析 Whisper/VAD 语义名、文件名、用户目录和显式路径。
- [x] 为资源解析顺序、缺失错误、禁用 VAD 和 JSON adapter 提供单元测试。

### llama.cpp、本地 Qwen 与 Hy-MT2

- [x] 以 submodule 管理 `vendor/llama.cpp`。
- [x] 提供 `openlrc.setup.llama_cpp`、setup wrapper 和 `openlrc setup llama`。
- [x] 统一解析 `llama-server`、`llama-cli`、GGUF profile 和模型路径。
- [x] 提供 `LocalLLMServer` 启动、健康检查、普通模式复用、owned process
      关闭和 idle timeout。
- [x] 提供 `LRCer.local()` / `TranslationConfig.local_qwen35_9b()`，裸 local
      默认使用 Qwen3.5 9B classic pipeline。
- [x] 提供 `hy-mt2-7b` 可下载 profile 和 `hy-mt2-30b-a3b` 显式 GGUF profile。
- [x] 提供 Hy-MT2 delimiter prompt、精确 ID validator、retry、binary split 和
      atomic fallback。
- [x] 对外暴露 Hy-MT2 `fast`、`normal`、`normal-plus`、`pro` 四种模式；
      `context` / `context-plus` 作为 deprecated alias 归一化后进入业务与 fingerprint。
- [x] 支持通用模型生成结构化 Translation Brief。
- [x] 支持完整和 Partial 人工 Translation Brief：人工 summary、characters、
      tone/style 整段锁定，缺失字段由模型补全，完整输入跳过自动 Brief 生成。
- [x] `run`、`translate` 和适用的独立编辑动作统一支持
      `--brief-summary`、`--brief-characters JSON|@PATH`、`--brief-tone-style`；
      人工术语继续使用 `--glossary`。
- [x] Brief 使用源语言语义字段和双语人物/术语映射；Timeline 只消费源语言
      语义投影，并通过 prompt version 隔离旧 checkpoint。
- [x] 精简 Hy-MT2 Brief：人物只保留双语名称映射，不在翻译前生成或注入人物
      描述、目标受众推测和 ASR 解释。
- [x] 支持纯本地 staged 执行，任一时刻只驻留一个 OpenLRC-owned 大模型。
- [x] 支持 Normal Plus 确定性定向修复和有界事务式语义编辑。
- [x] 支持 Pro 全局 Brief、逐 chunk ContextTimeline、Hy-MT2 翻译、确定性修复和
      有界事务式语义编辑。
- [x] contextual Hy-MT2 统一使用 schema v4 checkpoint，支持 v2 Normal/Normal Plus
      与 v3 Pro 迁移、Timeline、translation、edit 恢复及输入变化失效。
- [x] 支持 edit 失败时保留 Hy-MT2 草稿与可用字幕，并把 session 标记 incomplete。
- [x] 使用原子 checkpoint，支持中断后只恢复未完成 edit scope。
- [x] 完整成功后默认清理 checkpoint；显式 keep 或 incomplete edit 时保留。

### CLI、发行与文档

- [x] 发行身份为 `openlrc-mac`，当前版本 0.3.0，同时保留 `import openlrc`。
- [x] 提供等价的 `openlrc` / `openlrc-mac` console scripts。
- [x] CLI 覆盖 `doctor`、`models status`、`setup`、`transcribe`、`translate`、
      `run`、`glossary` 和 `edit`。
- [x] `openlrc run` 默认不翻译，必须显式选择 local 或 online。
- [x] CLI 暴露本地模型 profile、Hy-MT2 模式、context provider/model 和
      checkpoint/temp 保留选项。
- [x] 删除旧 Streamlit GUI、`openlrc gui` 和主要 faster-whisper/CUDA 配置。
- [x] 维护 README、CLI reference、英文/中文 changelog。
- [x] 将 `AGENTS.md`、`PROJECT_ARCHITECTURE.md` 和 Hy-MT2 适配文档设为
      local-only；架构或整体信息变化时同步维护架构文档。

## P0 - 稳定本地转写后端

- [ ] 修复或实测证明 `WhisperCLIBackend` 在长音频/大 JSON 输出下不会因
      stdout/stderr pipe 填满而阻塞。
- [ ] 增加真实 whisper.cpp JSON fixture：正常、多 segment、空结果、异常 JSON、
      缺少 word timestamp、缺少 offset、不同语言。
- [ ] 明确 native VAD 的完整错误边界：启用、禁用、模型缺失、参数错误、
      whisper.cpp 不支持相关参数。
- [ ] 验证当前 v1.9.1 pin 的关键 CLI 参数和 JSON schema。
- [ ] 增加音频与视频两条真实转写 smoke 流程，并记录输出检查项。
- [ ] 为 `openlrc.setup.whisper_cpp` 增加 mock/dry-run 测试，避免验证依赖真实
      build 和下载。
- [ ] 增加 Metal 是否实际启用的诊断信息，而不是只依赖构建默认值。

## P0 - 开发体验和测试基线

- [ ] 处理当前 4 个未通过全量 Ruff format check 的历史文件。
- [ ] 修复或配置 Pyright 对 LiteLLM、Torch、DeepFilterNet 可选依赖的 4 个
      missing-import 问题。
- [ ] 将 pytest 纳入稳定的开发依赖/离线环境，避免 `uv run --with pytest`
      在无网络和无缓存环境中无法执行。
- [ ] 保持 `tests/test_lazy_imports.py` 对 heavy import 和 `faster_whisper`
      forbidden import 的保护。
- [ ] 补充 app bundle resource 优先级测试；现有测试已覆盖显式配置、环境变量、
      user dir、vendor build 和 `PATH`。
- [x] 为 llama.cpp setup 下载与 build 结果提供 mock 单元测试。
- [x] 为 checkpoint 原子替换和失败保留旧文件提供测试。
- [x] 为 Hy-MT2 三阶段顺序、草稿保留和 incomplete review 恢复提供测试。
- [ ] 清理或重写过时的 `.github/workflows/dispatch_CI.yml`：该工作流仍使用
      Python 3.9、Poetry 和 faster-whisper，应改为当前 Python 3.10-3.12、`uv`
      和 whisper.cpp 技术栈，或在确认无用途后删除。
- [ ] 增加 macOS / Apple Silicon 的基础 CI；真实 Metal/模型测试可先保持手动。

## P1 - 术语表（Glossary）与定向多轮编辑

术语表用于规定名称、品牌、技术词和固定表达的目标译法，保证不同字幕块和多轮
编辑中的翻译一致。代码字段和 CLI 参数保留 `glossary`，文档使用中文名称。

### 术语表（Glossary）

- [x] `TranslationConfig` 支持 JSON 术语表路径。
- [x] classic/内部 delimiter 翻译可注入术语表。
- [x] Translation Brief 合并用户术语表，冲突时用户配置优先。
- [x] retry/split/atomic fallback 保留术语表，或明确执行无术语表重试。
- [x] 在 `run` / `translate` CLI 增加 `--glossary` 和强制使用选项。
- [x] 定义并校验术语表 schema，给出重复项、空值、类型错误和文件不存在的
      用户可读错误。
- [x] 支持查看最终合并术语表，并区分用户提供与模型生成条目。
- [x] 记录术语命中、未命中和因格式失败而移除术语表的指标。
- [x] 为术语大小写、alias、Unicode/CJK 和字幕跨行场景增加测试。
- [x] CLI 通过统一 `GlossaryService` 复用 package 逻辑；未来 TUI/GUI 沿用该服务。

### 定向多轮编辑

- [x] 定义稳定的字幕 line/segment ID、编辑状态和原因记录格式。
- [x] 支持从 compare/checkpoint 中选择指定行或 chunk 重新翻译，不重跑转写和
      整份字幕。
- [x] 支持把术语表、Translation Brief、相邻源文、最近译文和 Pro Timeline
      带入定向重译。
- [x] 首发 `verify` / `review` / `retranslate` / `restore` 四个动作。
- [x] 保证定向编辑不改变未选择行、时间轴、行数和 ID 对齐。
- [x] 记录每轮编辑前后文本、模型信息、原因、验证结果和停止原因，支持回滚比较。
- [x] 提供稳定 `LRCer.edit()` API 与 CLI 入口，供后续 TUI/GUI 编排。
- [ ] 后续增加 `apply-instruction` 和 `accept-draft`，不在 0.3.0 首发范围内。

### 后续上下文演进

- [x] 已实现 Hy-MT2 `pro`：通用模型先生成逐 chunk
      `ContextTimeline`，Hy-MT2 保持专职翻译，最后进入事务式语义编辑。
- [x] Pro checkpoint 保存 chunk signature、Timeline、翻译和 edit 进度，
      避免恢复时错配。
- [ ] 强化生成 Brief/Timeline 的语义质量门槛：评估重复术语漏提取、目标译法为空
      或不稳定，以及全局 Brief 将未来事件带入当前 scene 的情况。

## P1 - 模型和资源管理

- [x] `doctor` 和 `models status` 可检查默认 Whisper、VAD 和 Qwen 资源。
- [x] 注册 Qwen3.5 9B、Hy-MT2 7B 和 Hy-MT2 30B-A3B profile。
- [ ] 建立统一模型清单：名称、类型、repo、文件、大小、量化、推荐内存、用途、
      是否已安装。
- [ ] 让 `models status` 覆盖所有已注册 profile，而不只检查默认模型。
- [ ] 提供模型下载、断点续传、校验、重新下载、删除和定位命令。
- [ ] 为 tiny/base/small/medium/large-v3-turbo 提供明确选择建议。
- [ ] 为 Qwen/Hy-MT2 量化提供 Apple Silicon 内存和速度建议。
- [ ] 增强 `llama-server` 日志和错误分类：端口占用、alias 冲突、模型缺失、
      上下文不足、内存不足、启动超时。
- [ ] 评估不同 Apple Silicon 机器上的 Qwen3.5 9B 和 Hy-MT2 7B 速度、内存、
      翻译质量。
- [ ] 保持模型目录位于 `~/Library/Application Support/OpenLRC/models/`，
      不把模型放进 Git 或 package。

## P1 - CLI、TUI 与共享服务边界

产品访问层规划：CLI 是当前主入口和最快更新入口；TUI 是下一阶段的交互式终端
入口，功能更新尽量紧跟 CLI；GUI 面向稳定、易用的图形体验，允许晚于 CLI/TUI
接入新功能。三者必须复用同一套 package service，不能形成三套 pipeline。

- [x] 保持 CLI 为当前完整、可脚本化的主入口。
- [ ] 为真实常见失败补充 CLI 文档和 smoke：缺 ffmpeg、缺 submodule、缺模型、
      端口冲突、context 配置不完整、review incomplete。
- [ ] 让 doctor 显示可选 Hy-MT2 profile、Metal/build 信息和更具体的修复命令。
- [ ] 抽出 CLI/TUI/GUI 共用的 job runner 和服务接口。
- [ ] 统一 progress event，覆盖预处理、转写、优化、Brief、翻译、review、导出。
- [ ] 统一错误类型：资源、模型、ffmpeg、whisper.cpp、provider、llama-server、
      checkpoint、格式校验。
- [ ] 增加可控取消机制，并验证取消后的 subprocess、模型和临时文件状态。
- [ ] 新增独立 TUI 前端；不要把当前 Typer CLI 改造成菜单框架。
- [ ] 建立 CLI/TUI/GUI 功能矩阵：新能力默认先落到 API/CLI，再同步 TUI，最后
      在交互稳定后进入 GUI。

## P2 - macOS GUI 与打包

- [ ] 选择 GUI 技术路线，并确认可稳定调用 Python service/job runner。
- [ ] 文件选择、输出格式、语言、Whisper 模型和翻译 profile 选择。
- [ ] 模型安装状态、下载/定位入口和磁盘占用提示。
- [ ] 展示各阶段 progress、日志、错误和取消状态。
- [ ] 预览 `.srt` / `.lrc`，并集成术语表与定向编辑工作流。
- [ ] 保留高级路径覆盖，但不放在普通用户主流程中心。
- [ ] GUI 优先接入已经在 CLI/TUI 验证稳定的功能，不要求与实验性命令同步发布。
- [ ] 在服务边界稳定后再处理 app bundle 资源、签名、notarization 和 packaging。

## 文档与发布维护

- [x] README 始终区分“已经可用”和“规划中”，不提前宣称 GUI/打包完成。
- [x] CLI 行为变化时同步 `CLI_REFERENCE.md`。
- [x] 发布相关变更同步 `CHANGELOG.md` 与 `CHANGELOG.zh-CN.md`。
- [x] 已实现并验证的架构、模块、状态或测试发生变化后，同步本地
      `PROJECT_ARCHITECTURE.md`；纯规划和路线讨论只写 TODO 或独立计划文档。
- [x] 已实现并验证的 Hy-MT2 profile、prompt、模式或 checkpoint 变化后，
      同步本地 Hy-MT2 适配文档。
- [ ] 为普通用户补充从零开始的 macOS 安装、模型下载和转写/翻译教程。
- [ ] 为开发者补充 submodule pin 更新和验证清单。

## 暂缓或不做

- [ ] 暂不恢复上游 Streamlit GUI。
- [ ] 暂不恢复 faster-whisper / CUDA 配置入口。
- [ ] 暂不把模型、vendor build、虚拟环境、cache、manual media 或本地文档
      上传 Git。
- [ ] 暂不自动下载或转换 Hy-MT2 30B-A3B；用户显式提供本地 GGUF。
- [ ] 暂不把 RAG 或本地视觉上下文放在术语表/定向编辑之前。
- [ ] 暂不承诺通用 PyPI 体验或完整 `.app` packaging。
- [ ] 暂不整体重写上游 pipeline；继续围绕明确模块边界逐步演进。
