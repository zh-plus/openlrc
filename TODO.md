# OpenLRC Mac To Do List

Last verified: 2026-07-31

本文档记录 OpenLRC Mac 在 0.4.2 基线上的已完成能力、当前风险和后续优先级。
它不是发布承诺。项目仍以 macOS 本地字幕工作流为核心，优先保证转写稳定性、
翻译一致性和可恢复性。产品最终提供 CLI、TUI、GUI 三种访问模式：CLI 与 TUI
优先获得新功能，GUI 在交互和服务稳定后跟进。

## 当前基线：0.4.2

### 核心流水线

- [x] 保留上游 OpenLRC 的响度标准化、字幕切分、优化、翻译和输出 pipeline；
      预处理只保留 ffmpeg 响度标准化和视频音轨提取。
- [x] 将默认转写后端切换为本地 `whisper.cpp`。
- [x] 支持本地音频/视频转写和 `.lrc` / `.srt` 输出。
- [x] 支持 native Whisper VAD，并允许通过空配置禁用。
- [x] 支持分阶段 `pre_process`、`transcribe`、`translate`、`post_process`。
- [x] 提供 aggressive/relaxed 两种字幕优化 profile；宽松模式保持 segment
      数量、顺序和时间边界，并使用独立内部缓存。
- [x] 成功后默认只清理当前输入的临时文件，支持显式保留调试产物。
- [x] 提供 `openlrc.workflow` 同步 Workflow service，统一 Transcribe、Translate、
      Run 的请求、事件、结果、取消、错误和 owned subprocess 生命周期。
- [x] Run 根据主翻译资源自动选择执行策略：在线 Standard 使用 Pipeline；本地
      Qwen/Hy-MT2 使用 Memory Saver，全部转写完成后才开始本地翻译。

### whisper.cpp 与资源

- [x] 以 submodule 管理 `vendor/whisper.cpp`，当前 pin 为 commit `f049fff9`；
      由该源码构建的本机 `whisper-cli` 报告接口版本 `1.9.1`，不把它误写成
      submodule Git tag。
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

- [x] 发行身份为 `openlrc-mac`，当前版本 0.4.2，同时保留 `import openlrc`。
- [x] 提供等价的 `openlrc` / `openlrc-mac` console scripts。
- [x] CLI 覆盖 `doctor`、`models status`、`setup`、`transcribe`、`translate`、
      `run`、`glossary` 和 `edit`。
- [x] `openlrc run` 默认不翻译，必须显式选择 local 或 online。
- [x] CLI 暴露本地模型 profile、Hy-MT2 模式、context provider/model 和
      checkpoint/temp 保留选项。
- [x] 删除旧 Streamlit GUI、`openlrc gui` 和主要 faster-whisper/CUDA 配置。
- [x] 删除 DeepFilterNet、DeepFilterLib、Torch、Torchaudio、ONNX Runtime、
      `full` extra 和所有 Noise Suppression 产品入口；不保留旧参数兼容层。
- [x] 项目 Python 支持范围为 `>=3.11,<3.15`，CI 完整覆盖 3.11、3.12、3.13、
      3.14；`.python-version` 和本地 uv 环境固定 Python 3.13.14。
- [x] 维护 README、CLI reference、英文/中文 changelog。
- [x] 将 `AGENTS.md`、`PROJECT_ARCHITECTURE.md` 和 Hy-MT2 适配文档设为
      local-only；架构或整体信息变化时同步维护架构文档。

## P0 - CI 与发布门禁

> 本节及下列 P0 的 `[x]` 表示代码或文档已经在当前工作区实现，并完成相应本地
> 验证；不表示这些改动已经 Git commit、push，或已经在 GitHub Actions 远端运行。
> 必须依赖远端或非受限硬件环境的验收继续单独保留为 `[ ]`。

- [x] 重建 `.github/workflows/ci.yml`：只在 `master` push、以 `master` 为 base 的
      pull request 和 GitHub 手动触发时运行；使用 `contents: read` 最小权限，并按
      workflow/ref 取消已经过时的同分支运行。
- [x] 主测试 job 使用 `python -m pytest -q` 收集完整 pytest 测试，替换
      `unittest discover`；后者无法正确收集 pytest 函数、fixture 和 marker。
- [x] 主测试矩阵在 `macos-latest` 上覆盖 Python 3.11、3.12、3.13、3.14；每个 job 先执行
      `uname -m` 并要求结果为 `arm64`，防止 runner 标签变化后静默改用非 Apple
      Silicon 架构。
- [x] 四个 Python 版本都安装 core、dev 和 `litellm` extra，并分别运行完整 pytest；
      默认 CI 不再解析或安装 DeepFilterNet/PyTorch/ONNX Runtime/spaCy。
- [x] 删除仅供旧 Live API 相似度断言使用的 spaCy 及运行时模型下载辅助函数；
      provider smoke 改为使用现有 Lingua 验证非空输出、目标语言和关键事实保留，
      并通过 wheel metadata 回归禁止重新引入 spaCy。
- [x] 建立独立的 Python 3.13 `Quality and build` job，依次运行 Ruff lint、
      全库 Ruff format check、生产代码 Pyright、`openlrc --version`、`uv build`
      、wheel metadata 检查和当前 push/PR 变更范围的空白错误检查。
- [x] 主 CI 显式设置 `OPENLRC_TEST_LIVE_API=0`、
      `OPENLRC_TEST_STRESS=0`、`OPENLRC_TEST_REAL_WHISPER=0` 和
      `OPENLRC_TEST_REAL_WHISPER_METAL=0`；普通 checkout 不初始化 submodule，
      测试中的 setup/download 只使用 mock，因此不会下载模型、构建 vendor 或运行
      真实 Whisper/LLM 推理。
- [x] 新增 `.github/workflows/live_api.yml`：只允许 GitHub 手动触发，在
      Ubuntu/Python 3.13 上把仓库 `OPENROUTER_API_KEY` secret 映射为测试专用变量，
      运行 agents/chatbot/translate/lean-translator provider 测试；缺少 secret
      时立即失败但不输出密钥。该 workflow 不加入主 CI；是否配置为 branch
      protection required check 也不在本任务范围内。
- [x] 删除上游遗留的 `.github/workflows/dispatch_CI.yml`，同时移除该 workflow
      中的 Python 3.9、Poetry、faster-whisper/CUDA 和写入 API key 的旧流程；不恢复
      Windows/Linux 主 CI 兼容承诺。
- [x] Python 3.11 最低版本基线直接使用标准库 `StrEnum` / `datetime.UTC`，并对
      本轮实际修改文件完成 Ruff 格式化。
- [x] 在 Python 3.13.14 本地完成两个 workflow 的 YAML 解析、完整 pytest、聚焦回归、
      Ruff、Pyright、CLI version/Doctor、`uv build`、wheel 隔离安装和
      `git diff --check`；完整测试为 `641 passed, 25 skipped`，CPU 音频/VAD及
      视频/ffmpeg 真实 smoke 为 `2 passed`。
- [x] 获得明确推送许可后，在 GitHub Actions 验证 `Tests / Python 3.11`、
      `Tests / Python 3.12`、`Tests / Python 3.13`、`Tests / Python 3.14` 和
      `Quality and build` 首次全绿；2026-07-31 的
      [Actions run 30620600378](https://github.com/infinitebook/openlrc_mac/actions/runs/30620600378)
      五个 job 均成功。本任务没有修改 GitHub branch protection，因此哪些 job
      被设为 required 仍由仓库设置决定。
- [x] pytest 8、Pyright、lazy-import guard、llama setup、checkpoint 和 Hy-MT2
      staged/resume 测试基线已经建立。

现有 `.github/workflows/bandit.yml` 和 `.github/workflows/codeql.yml` 继续作为
独立安全扫描，不属于上述主测试/质量 workflow。Bandit 当前设置 `exit_zero: true`，
因此只上传发现而不阻断合并；CodeQL 负责 Python 静态安全分析。两者是否列为 branch
protection required check 仍取决于 GitHub 仓库设置，本轮没有重写这两个 workflow。

## P0 - whisper.cpp 输入契约与错误边界

- [x] 并发排空 `whisper-cli` stdout/stderr，避免长音频和大 JSON 输出死锁。
- [x] 锁定并记录 `vendor/whisper.cpp` submodule commit
      `f049fff95a089aa9969deb009cdd4892b3e74916`。该 commit 的 `git describe`
      可能显示 `v1.8.5-95-gf049fff9`，但由它构建的本机
      `whisper-cli --version` 明确输出 `whisper.cpp version: 1.9.1`。这里的
      `1.9.1` 指 CLI 输出/JSON 接口版本，不是 OpenLRC Mac 版本、Whisper 模型版本，
      也不声称该 commit 带有 `v1.9.1` Git tag。
- [x] 使用当前 `whisper-cli --help` 核验本项目实际依赖的参数：`-ojf`/`-of`
      生成 owned JSON，`--vad`/`-vm` 启用并指定 VAD，`-ng` 禁用 GPU，
      `-nfa` 禁用 flash attention；未来升级 binary 时必须重新核验这些参数。
- [x] 新增版本化 fixture
      `tests/data/whisper_cpp/v1.9.1/test_audio_vad_cpu.json`。它由上述本机
      `whisper-cli 1.9.1` 使用 `tests/data/test_audio.wav`、Whisper base 模型、
      Silero VAD v6.2.0 和 `-ng -nfa`（CPU、关闭 flash attention）真实生成；只截取
      覆盖契约所需的真实 segment，并把模型绝对路径替换为
      `<WHISPER_MODEL_PATH>`。生成条件和刷新规则记录在同目录 `README.md`，
      这里的“新增”不代表已经完成 Git commit。
- [x] 基于真实 fixture 及其参数化变体覆盖：空 `transcription`、token 时间缺失、
      segment 数值 offsets 缺失、检测语言变化、错误字段类型、损坏 JSON 和非 object
      JSON 根结构。
- [x] 将 `whisper-cli` 非零退出、没有 JSON 输出、损坏 JSON、非 object 根结构、
      owned JSON 读取失败和不可恢复 schema 错误统一转换为
      `TranscribeException`；错误包含输出来源、退出码或 JSON 行列，stderr 只保留
      最后约 4096 个字符，避免把整份 JSON 或无限日志写进错误信息。
- [x] segment 优先使用毫秒 offsets，缺失或无效时回退字符串 timestamps；两者都
      无效时报告具体 segment 序号并失败。token 优先使用自身时间，两种 token 时间
      都无效时继承已经验证的所属 segment 时间范围。
- [x] 覆盖 VAD 配置非空时追加 `--vad -vm`、空配置时完全禁用 VAD、VAD 模型缺失
      以及 `whisper-cli` 拒绝 VAD 参数并非零退出等边界。
- [x] 为 `openlrc.setup.whisper_cpp` 增加直接 mock 测试，验证 submodule 初始化、
      CMake configure/build、模型下载命令和产物 postcondition；不为生产代码增加
      只服务于测试的 dry-run 接口。

## P0 - macOS 真实转写与 Metal 诊断

- [x] 为 `transcribe` 和 `run` 增加
      `--whisper-gpu/--no-whisper-gpu` 与
      `--whisper-flash-attn/--no-whisper-flash-attn`。两项默认开启并映射到
      `TranscriptionConfig.asr_options`；GPU 失败不会自动重试 CPU，避免掩盖 Metal
      初始化故障。
- [x] Doctor 通过有界 `whisper-cli --version` 探测版本，不运行推理。本仓库 vendor
      binary 读取 `vendor/whisper.cpp/build/CMakeCache.txt` 判断 Metal build
      capability；外部或未来 App Bundle binary 显示“构建能力未知、运行时未探测”。
      Metal 未构建或未知不否定 CPU 可用，因此不会仅凭该状态让 Doctor 失败。
- [x] 在 `tests/test_whisper_smoke.py` 增加
      `OPENLRC_TEST_REAL_WHISPER=1` 才运行的真实音频 smoke：复制测试 WAV 到 pytest
      临时目录，启用 Silero VAD，显式关闭 GPU/flash attention，解析真实
      `whisper-cli 1.9.1` JSON，并检查时间非负、单调和 owned process 回收。
- [x] 在同一测试文件增加相同 opt-in 开关控制的真实视频 smoke：复制测试 MP4 到
      pytest 临时目录，禁用 VAD，真实执行 ffmpeg 音轨预处理和 CPU Whisper
      Workflow，并确认所有输出/artifact 都位于临时目录。
- [x] 增加 `OPENLRC_TEST_REAL_WHISPER_METAL=1` 才运行的 macOS-only Metal smoke；
      它要求真实转写成功、JSON 契约有效、日志表明 GPU/Metal 已启用且没有 Metal
      buffer 初始化错误。默认 CI 明确禁用该开关。
- [ ] 在不限制 Metal buffer 的 Apple Silicon macOS 终端运行上述 Metal smoke 并
      实际通过；当前受限 Codex 环境的 CPU 通过结果不能替代这项验收。
- [x] 在本地 `DevelopDocument/whisper.cpp v1.9.1 升级与 Smoke 核验清单.md`
      记录 submodule/binary 身份、参数核验、fixture 刷新步骤、CPU smoke 结果和
      Metal 待验收项；该文档继续 local-only，不进入 Git。

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
- [x] CLI 通过统一 `GlossaryService` 和 Workflow 配置复用 package 逻辑；TUI v2
      与未来 GUI 沿用该服务。

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
- [ ] 后续增加 `apply-instruction` 和 `accept-draft`，不在 0.4.0 首发范围内。

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

产品访问层规划：CLI 是当前完整、可脚本化的入口；TUI v2 是新的交互式终端入口并
尽量紧跟 CLI；GUI 面向稳定、易用的图形体验，允许晚于 CLI/TUI
接入新功能。三者必须复用同一套 package service，不能形成三套 pipeline。

- [x] 保持 CLI 为当前完整、可脚本化的主入口。
- [ ] 为真实常见失败补充 CLI 文档和 smoke：缺 ffmpeg、缺 submodule、缺模型、
      端口冲突、context 配置不完整、review incomplete。
- [ ] 让 doctor 显示可选 Hy-MT2 profile 和更具体的修复命令。
- [x] 抽出 CLI/TUI/GUI 共用的同步 `WorkflowExecutor` 和 typed request/result 接口。
- [x] 统一 progress event，覆盖预处理、转写、优化、Brief、Timeline、翻译、
      deterministic repair、semantic review、导出和清理。
- [x] 统一错误类型：资源、模型、ffmpeg、whisper.cpp、provider、llama-server、
      checkpoint、格式校验。
- [x] 增加协作取消与 owned-process registry；取消时清理 Whisper、ffprobe/ffmpeg
      worker 和 OpenLRC-owned llama-server，保留 checkpoint 和已完成产物。
- [x] 收口 Workflow 运行时语义：Context 一次性使用，取消优先于并发进程错误，
      Run 中间 artifact 不冒充主输出，最终结果不返回已删除路径，逐文件事件可归属。
- [x] 将 `transcribe`、`translate`、`run` CLI 迁移到 Workflow；保留参数、输出表、
      review incomplete 退出语义以及五模式配置推断。
- [x] 移除旧 Textual TUI 界面层、专属测试和旧入口，保留 application/workflow
      共享服务作为 TUI v2 基础。
- [x] 开发 TUI v2：品牌化卡片 Home、列表式内页、全键盘和鼠标等价操作。
- [x] 初版提供 New Work、Jobs、Models、Settings；Edit 卡片保持 disabled，
      后续在 typed Edit contract 完成后启用。
- [x] 重建 macOS native picker、Terminal Browser、统一输入 normalizer 与启动前
      Preflight；Resume 必须回到 Confirm 审查。
- [x] 重建 Jobs、计划输出、增量日志、Settings working copy，以及
      Starting/取消/历史持久化竞态保护的 TUI v2 消费层。
- [x] 建立 CLI/TUI/GUI 功能边界：新能力默认先落到 API/CLI，再同步 TUI，最后
      在交互稳定后进入 GUI。
- [x] TUI v2 使用真实 Textual SVG snapshot 与 Pilot 覆盖 160×50、100×30、80×24、
      Home 卡片、动态状态、disabled Edit、键盘/鼠标路径和单活跃任务限制。
- [x] 完成 2026-07-24 可用性修订：列表首尾循环、页面分组、New Work 三步进度、
      readonly Preflight、clean/dirty Draft、80×24 键盘输入 Modal 和 Logo 整列流光。
- [x] 完成 Appearance 运行时主题、stable action 焦点恢复、静态蓝色 Logo、
      English/简体中文设置、持久化与两主题/中文 snapshot。
- [x] 完成 Home Logo/介绍中心轴、卡片内部标题位置、显式分组 spacer 和 disabled
      Save 连续结构边框，并通过 80×24、100×30、160×50 Pilot/SVG/PTY 验收。
- [x] 完成 TUI v2 中低优先级可靠性收口：动态重组统一恢复 stable action/focus，
      Jobs `d` 局部快捷键、有效 Preflight 摘要、持久化 warning、Glossary 竞态、
      active-field recipe、Terminal picker Space 和 i18n 私有依赖 fallback 均有回归。
- [x] 删除未进入渲染层的 ASCII Status 设置；旧 Settings JSON 继续兼容，并在下次
      保存时自然移除废弃字段。

## P2 - macOS GUI 与打包

- [ ] 选择 GUI 技术路线，并确认可稳定调用 Python service/job runner。
- [ ] 文件选择、输出格式、语言、Whisper 模型和翻译 profile 选择。
- [ ] 模型安装状态、下载/定位入口和磁盘占用提示。
- [ ] 展示各阶段 progress、日志、错误和取消状态。
- [ ] 预览 `.srt` / `.lrc`，并集成术语表与定向编辑工作流。
- [ ] 保留高级路径覆盖，但不放在普通用户主流程中心。
- [ ] GUI 优先接入已经在 CLI、当前 TUI v2 验证稳定的功能，不要求与实验性命令同步发布。
- [ ] 在服务边界稳定后再处理 app bundle 资源、签名、notarization 和 packaging。
- [ ] 打包实现后补充 app bundle resource 优先级测试；当前测试已覆盖显式配置、
      环境变量、user dir、vendor build 和 `PATH`。
- [ ] 调研仍在维护、适合 Apple Silicon 且不强制引入大型 Torch 依赖链的降噪方案；
      在替代方案的质量、许可、体积和取消语义验证完成前，不恢复 Noise Suppression。

## 文档与发布维护

- [x] README 始终区分“已经可用”和“规划中”，不提前宣称 GUI/打包完成。
- [x] CLI 行为变化时同步 `CLI_REFERENCE.md`。
- [x] 发布相关变更同步 `CHANGELOG.md` 与 `CHANGELOG.zh-CN.md`。
- [x] 已实现并验证的架构、模块、状态或测试发生变化后，同步本地
      `PROJECT_ARCHITECTURE.md`；纯规划和路线讨论只写 TODO 或独立计划文档。
- [x] TUI 架构、页面、设置 schema、主题、语言、焦点、Logo 或验收变化时，同步
      本地持续维护的 TUI v2 开发文档；不从公共文档链接 `DevelopDocument/`。
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
