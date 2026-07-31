# OpenLRC Mac 更新日志

这个文件只记录 OpenLRC Mac fork 自己的变更。英文版见
[CHANGELOG.md](CHANGELOG.md) 中的 OpenLRC Mac 条目；两个文件需要同步更新。

## OpenLRC Mac 0.4.2

本版本以旧模块和旧依赖清理为主线，同时把默认开发环境升级为 Python 3.13，
并建立 Python 3.11-3.14 的完整 CI 支持。

### 新增

- 围绕 macOS/Apple Silicon 产品范围重建 required test/quality workflow，覆盖
  Python 3.11-3.14、uv、pytest、Ruff、Pyright 和包构建；真实 provider 检查拆到
  独立手动 workflow，不进入 required CI。
- `transcribe` 与 `run` 新增显式 Whisper GPU 和 flash-attention 正反开关；
  Doctor 新增不运行推理的 whisper.cpp 版本与 Metal build capability 状态。
- 新增去除本机路径的 v1.9.1 whisper.cpp 真实 JSON fixture、setup/输入契约回归，
  以及 opt-in CPU/VAD、视频/ffmpeg 和 Metal smoke。

### 变更

- 包支持范围改为 Python `>=3.11,<3.15`，仓库开发环境固定 Python 3.13.14；
  Python 3.13 同时作为 quality、包构建和手动 Live API 默认版本。
- 更新 tiktoken、Lingua 与 LiteLLM 约束以支持 Python 3.14；直接使用 Python
  3.11 标准库 `StrEnum` 与 `datetime.UTC`，删除 Python 3.10 兼容写法。
- 手动 Live API 测试不再使用 spaCy 对固定参考译文评分，改用现有 Lingua 检查
  非空输出、目标语言和关键事实保留等确定性契约。
- 预处理简化为视频音轨提取和 ffmpeg 响度标准化。`LRCer` 与 Workflow 请求的可选
  参数改为 keyword-only，删除旧位置参数后不会发生静默错位。

### 删除

- 删除 DeepFilterNet、DeepFilterLib、Torch、Torchaudio、CUDA PyTorch package
  source 和 `full` extra。
- 删除 faster-whisper 遗留且产品代码未使用的 ONNX Runtime。
- 删除 spaCy、运行时语言模型下载辅助函数，以及此前仅为 spaCy/Weasel 兼容而
  直接声明的 Click 依赖。
- 从 `LRCer`、`TranscriptionConfig`、Workflow request、Settings、recipe、CLI 和
  TUI 删除 Noise Suppression。旧 `--noise-suppress` 会报告 unknown option，旧
  Python 参数会直接产生 `TypeError`。

### 修复

- whisper-cli 非零退出、无输出、损坏 JSON 和不可恢复 schema 统一转换为有界且带
  上下文的转写错误；segment 缺少 offsets 时回退 timestamps，token 缺少时间时
  使用所属 segment 范围。
- Settings、Workflow 参数、Input Files、运行结果、Home、Doctor、Models 和 Setup
  动态重组时会按 stable action 恢复选中行与键盘焦点；Job Detail 的 `d` 现在优先打开
  Delete，不再被全局 Doctor 快捷键截获。
- Preflight 按构造后的真实 Workflow request 生成翻译摘要；Source Subtitle 会显示为
  transcribe-only，不再错误展示翻译模式和目标语言。
- 启动和 Workflow 终态的历史持久化失败会作为非致命 warning 告知用户，同时保留成功
  的内存结果；Interrupted 恢复状态回写失败时也不会阻断 TUI 启动。
- 无效 Default Glossary 的 Inspect 会被禁用；compose 后文件变化等竞态只显示错误通知，
  不再让异常逃出 Textual event handler。
- Workflow recipe 只保留当前 work type、backend、Hy-MT2 mode 和 Context 规则实际启用
  的字段。旧 recipe 仍可加载，但隐藏的陈旧设置不会再次进入 dirty/history/Resume。
- Terminal picker 中获得焦点的按钮支持 Space，路径输入中的空格不受影响；删除未被渲染
  使用的 ASCII Status 设置，并为 Textual 私有语言 context 增加惰性英文 fallback。

### 验证

- Python 3.13.14 完整 pytest 为 `640 passed, 25 skipped, 19 warnings`；
  预处理、Workflow、application、CLI、TUI、lazy-import 与包 metadata
  聚焦回归为 `193 passed`。
- 使用已安装 v1.9.1 binary 的 opt-in CPU 音频/VAD与视频/ffmpeg 真实 smoke 通过；
  结果为 `2 passed`，Metal 用例未选中。20 个 TUI SVG baseline 及隔离的
  80×24 Home -> Doctor -> Home -> Quit PTY 导航 smoke 也通过。
- Ruff lint/format、Pyright、YAML 解析、CLI version、Doctor/strict、包构建和
  wheel metadata/入口隔离安装、`git diff --check` 均在本地通过。首次远端 CI
  全绿和非受限 Apple Silicon Metal smoke 仍待完成。

## OpenLRC Mac 0.4.1

本版本集中修复 TUI 可靠性：关闭 Starting 取消窗口，保护 Settings/Draft 导航事务，
加入可关闭的 Context assistance，并改善 Brief 输入与运行日志展示。

### 新增

- 在 Python 配置、Hy-MT2 factory、Workflow recipe、CLI
  `--context-assistance` 和 TUI 中新增公开 `ContextAssistance` 策略，支持
  `auto` / `off`。`off` 可让 Normal 或零轮语义审查的 Normal Plus 使用完整人工
  Translation Brief 运行，且不会构造 Context model。

### 修复

- TUI 挂载期间把 OpenLRC 运行日志捕获到页面下方固定的 `RUNTIME OUTPUT` 框，
  不再让 terminal stream 覆盖 Textual 画面；退出 TUI 时会恢复原 terminal handler。
- Workflow 进入终态后，Esc、Home、Jobs 和 New Work 会统一移除已完成任务的页面栈并
  丢弃已消费 Draft，不再暴露刚才的配置页面。
- 新建 Hy-MT2 Draft 在需要 Context assistance 时默认使用 Settings 中配置的本地
  Qwen；配置为空时回退到内置 `qwen3.5-9b` profile。缺少旧 Context 字段的历史 recipe
  Resume 时也会补入当前默认值。
- Translation Brief 改用多行编辑器，并显示键盘提示、固定 Apply/Cancel 操作区；在
  80×24 下输入框、操作和底边框仍完整可见。非法 Character 映射会留在编辑器中显示
  行级错误且不修改 Draft，不再在页面重组时导致程序崩溃。
- TUI 在调度 Workflow/Setup worker 前创建共享 cancellation token，关闭 Starting
  阶段的取消窗口。即使 controller 尚未注册，确认取消也会送达 operation、跳过尚未
  开始的工作、清理 owned resource，并只产生一个 Cancelled 终态。
- dirty Settings 的根页面 Esc、返回 Home 和 Quit 统一使用事务式离开保护。保存失败
  会保留 working copy 和当前页面，只有保存成功或明确 Discard 才执行后续导航。
- 从历史 Resume 前保护现有 dirty Workflow Draft。只有确认丢弃后才创建新任务；
  Keep 或取消会保留原 Draft。
- 将 Hy-MT2 Context 依赖规则收口到配置、Workflow、application、CLI 和 `LRCer`
  共用的 resolver。Auto 继续让模型补全缺失/Partial Brief；Off 要求 summary，并将
  未提供的人物和语气规范化为显式空值；Pro 和带语义审查的 Normal Plus 拒绝 Off。
- 从 Git 和源码发行包中排除仓库内 UV/虚拟环境缓存，避免 release archive 意外携带
  本机构建缓存内容。

### 验证

- 0.4.1 发布回归为 `578 passed, 25 skipped, 21 warnings`；Ruff lint、修改文件的
  format check、生产 `openlrc/` Pyright（`0 errors, 0 warnings`）和
  `git diff --check` 通过。
- `openlrc --version` 显示 0.4.1；`uv build --offline` 成功生成 0.4.1
  sdist/wheel。包内容核对确认源码包无本地缓存，版本 metadata、TUI stylesheet、
  application/workflow 模块及 `openlrc`、`openlrc-mac`、`openlrc-tui` 入口完整。

## OpenLRC Mac 0.4.0

共享 Workflow/application service 与 Textual TUI v2 版本，同时强化本地转写、
取消、恢复和中英双语终端体验。

### 版本亮点

- **共享执行基础**：CLI 与 TUI 统一使用类型化 Workflow/application contract，
  不复制 `LRCer` 或字幕 pipeline。
- **Textual TUI v2**：提供键盘/鼠标等价的 New Work、Jobs、Models、Doctor 和
  working-copy Settings，并覆盖 English/简体中文与确定性视觉 baseline。
- **更安全的本地处理**：统一强化 Whisper、ffmpeg、Qwen、Hy-MT2 的 subprocess
  ownership、取消、原子输出、checkpoint 保留、多文件调度和冲突保护。

### 新增

- 新增 `openlrc.workflow` 同步类型化执行层，统一 Transcribe、Translate 与 Run
  的请求、逐文件阶段/chunk 事件、产物和 review 结果、结构化错误、协作取消与
  owned-process 清理；`LRCer` 继续作为业务编排器。
- Workflow 完整覆盖 Standard、Fast、Normal、Normal Plus、Pro，包括在线 provider、
  本地 Qwen/Hy-MT2、Brief/Timeline、确定性修复、语义审校、checkpoint、Restore
  和分阶段单模型驻留；旧配置/CLI alias 在事件和结果中使用 canonical 名称。
- 新增共享本地 Preflight；交互任务启动前会检查输入 schema、所需资源、输出冲突与
  目标目录写权限。
- 新增可供未来 GUI 复用的 application services，覆盖版本化原子设置、无密钥历史、
  Keychain/环境变量解析、provider 测试、资源状态、Workflow recipe 和单任务控制。
- 新增 Textual TUI v2：品牌化单屏 Home、New Work、Jobs/Recovery、Models/Setup、
  Doctor、working-copy Settings、四类 Workflow 与键盘/鼠标等价 ActionList；
  Edit Subtitle 初版保持 disabled。
- 新增类型化 glossary、可取消 Setup application adapter 和共享 operation guard，
  防止 Workflow 与资源修改并行执行。
- 新增 `openlrc tui` / `openlrc-tui` 入口、真实 Textual SVG baseline、Pilot
  交互覆盖和 `TUI_REFERENCE.md`。
- TUI 新增可持久化的 English/简体中文语言选项；Home、Workflow、Jobs、Models、
  Doctor、Settings、Modal、帮助、通知和动态计数使用 presentation-only 本地化。

### 变更

- 在 TUI v2 重建前移除实验性的 TUI v1 界面层、专属测试以及
  `openlrc tui` / `openlrc-tui` 入口；可复用的 application 与 Workflow 服务保留。
- 转写 sentence segmentation 不再加载 spaCy 模型/CLI；该阶段仅需要的标点判断改用
  Unicode 分类，因此 Transcribe 不会意外触发语言模型下载。
- `transcribe`、`translate`、`run` CLI 已迁移到 `WorkflowExecutor`，保留原参数、
  Generated Files 表和 review incomplete 成功退出语义；交互取消会等待 owned
  资源清理后以 130 退出。
- 多文件 Run 改为受控 futures 与 cancellation-aware queue：重复输入只执行一次，
  结果保持首次输入顺序，worker 异常向上传播，失败、取消或 incomplete review
  不清理恢复材料。
- Run 使用固定资源策略：在线 Standard 保留可重叠的 producer/consumer Pipeline；
  本地 Qwen 和所有 Hy-MT2 模式使用 Memory Saver，全部文件转写完成后才开始翻译。
- Workflow Run 新增供交互界面使用的 checkpoint 保留选项，同时保持 CLI 原有清理
  默认行为。
- Model lifecycle event 增加兼容的 role/endpoint，并提供共享资源状态服务。
- TUI v2 使用更简洁的 New Work 文案、四个精确任务标题与 amber 功能词；配置页统一
  使用单列分组线框、三步进度和突出的 Continue to Preflight 主动作。
- Appearance 接入 Textual 运行时主题：OpenLRC Dark 注册为青蓝/暖黄色主题，
  Textual Dark 使用独立的内置调色板；共享 CSS token 会立即预览并在 Save 后持久化。
- Home 右侧五张四行高卡片现在作为一个整体垂直居中；所有共享 ActionList 压缩为
  一行标题或两行标题/detail；标题与 action 之间使用显式单行 spacer，分组之间只
  保留一行间距，动态 detail 会即时重排。

### 修复

- 将 Click 声明为 spaCy/Weasel 兼容所需的直接运行依赖，避免 Whisper 转写后出现
  `No module named 'click'`。
- 并发 drain whisper.cpp stdout/stderr，避免大 JSON 因 pipe 背压死锁；转写 JSON
  改为原子提交。
- `WhisperCLIBackend` 优先从 owned 临时输出文件读取 JSON，并为旧版 whisper.cpp
  保留 stdout fallback，兼容 `--no-prints` 不输出 JSON 或 stdout 混入可读转写文本。
- Whisper、ffmpeg 抽取/响度归一化和 OpenLRC-owned llama-server 纳入统一
  terminate/kill/wait 清理；外部复用的 llama-server 不会被关闭。
- provider retry wait、预处理 chunk、Brief/Timeline、翻译与 review chunk 支持
  协作取消；不会从另一线程强制关闭正在执行的远端 SDK 请求。
- 已请求取消时，worker 或 subprocess 同时产生的异常不再覆盖取消结果；失败事件
  补齐 credential 脱敏，`ExecutionContext` 明确为一次性对象且 cleanup 幂等。
- Run 转写 JSON 作为中间 artifact，不再进入主输出；最终结果过滤已删除文件，
  Brief、Timeline 和模型生命周期事件继承逐文件标识。
- Workflow `ffprobe` 纳入 owned-process 取消清理；DeepFilterNet 在取消和预处理异常
  后也保证释放模型内存。
- 修复 application job 在 worker 注册前重复启动和取消丢失的竞态；初次历史写入失败
  时不再执行，终态写入失败会释放 active slot。
- 通过运行时 owned-path 跟踪保护已有 checkpoint 和同名媒体 sidecar；处理前发现
  有歧义的输出冲突会明确拒绝，不再覆盖或清理用户文件。
- TUI 所有 action list 现在会在 enabled 项之间首尾循环，并跳过分组标题和 disabled
  行；空白 provisional Draft 静默丢弃，只有输入或参数真实变化才提示恢复。
- Preflight 请求和检查改为不可聚焦的只读区域，焦点只包含 Start/Back；Page Up/Down
  可翻阅长内容且不移动 action focus，返回输入编辑器后参数页会立即刷新文件计数。
- Settings 输入弹窗改用统一 Apply/Cancel action list，80×24 下不再遮挡按钮，支持
  完整键盘和鼠标操作，credential 输入保持遮罩。
- wide/compact ASCII Logo 流光改为按终端 `x` 坐标整列移动，同列多行字符同步变色。
- Appearance 普通设置改为原地更新，Language 重组会在新 ActionList 挂载后按
  stable action ID 恢复焦点；Enter 应用后无需鼠标即可继续使用方向键。
- 关闭 Logo animation 或启用 Reduced motion 时重新渲染静态亮蓝 Logo，不再冻结
  最后一个黄色流光 frame；动画未经过的字符也保持蓝色。
- Home 的全宽 Logo 字符块现在与两句介绍共用中心轴；无 detail 的四行卡片标题移至
  上方内部格，wide/compact Logo 每行统一补齐为固定 terminal-cell 矩阵。
- 分组标题后和相邻 action 之间新增不可交互且延续边框的 spacer 行；clean Settings
  Save 只降低文字和交互状态，结构边框继续与 Heading、spacer、Discard 使用同一 token。

### 验证

- 使用 80×24、100×30、160×50 确定性 Home SVG，以及 Workflow、Jobs、Models、
  Settings、Doctor、确认弹窗代表性 snapshot 验证 TUI v2；Pilot 覆盖 disabled card
  焦点/点击、active mouse route、输入快捷键隔离和持久化 Workflow result。
- 实际 100×30 PTY 完成 Home、New Workflow、Jobs、Models、Settings、Doctor
  （10/10 本地检查）、Esc 返回和确认退出。另一实际 TUI 使用已安装 whisper.cpp/base
  模型以 CPU/no-flash-attention 完成 Transcribe to JSON，保存成功 Job，并从
  30.3 秒音频生成 13 个 segment。全量测试为 `527 passed, 25 skipped`；Ruff、
  Pyright（`0 errors`）和 `git diff --check` 通过。
- TUI 交互修订定向测试 `28 passed`；真实 Textual SVG 覆盖 80×24、100×30、
  160×50、三个连续 wide Logo fixed frame，80×24 Preflight 已转图检查。全量测试
  为 `539 passed, 25 skipped, 21 warnings`；Ruff、Pyright（`0 errors`）与
  `git diff --check` 通过。实际 PTY 验证输入计数刷新、Preflight focus/退出路径，
  实际 TUI `Transcribe to JSON` 端到端进入 Succeeded 并产生 output。
- Appearance 与中文化定向 TUI/Pilot/SVG 为 `34 passed`，连同 application settings
  为 `57 passed`；两套 80×24 主题、简体中文 Home 和静态 Logo 已建立 baseline
  并转 PNG 检查。全量测试为 `546 passed, 25 skipped, 21 warnings`，Ruff 与
  Pyright（`0 errors`）通过。隔离 Application Support 的实际 80×24 PTY 还验证了
  主题/语言切换后键盘焦点连续可用、关闭动效立即显示静态亮蓝 Logo，以及退出重启后
  主题、语言和动效设置均正确恢复。
- 视觉布局修复的 TUI/Application 定向回归为 `65 passed`，包含 19 个归一化 SVG
  baseline 和最终彩色 PNG 人工检查。Pilot 覆盖三种 Home 尺寸的垂直居中、动态
  一/两行 Action、单行组间距、clean/dirty Save 与固定 Logo 列；隔离设置目录的
  实际 80×24、100×30 `openlrc-tui` 均完成全键盘 Home/Settings 路径，后者还完成
  Appearance 修改与 Save。修改范围 Ruff 和生产 TUI Pyright 通过。
- Logo/卡片/分组跟进修复通过 `42` 项 TUI/Pilot/SVG 回归和全部 19 个归一化
  baseline，并完成 Ruff、生产 TUI Pyright（`0 errors, 0 warnings`）与 PNG
  人工检查。隔离设置目录的真实 PTY 验证了 80×24 Work Type 完整底框，以及
  160×50 Home/Settings 的中心轴、spacer 和 disabled Save 连续边框。
- 移除 TUI v1 后，CLI/application/Workflow 边界定向回归为 `102 passed`；修改范围
  Ruff lint、CLI help 和 `git diff --check` 也通过。
- 使用已安装 whisper.cpp/base 模型完成真实 Workflow Transcribe，并完成真实本地
  Normal 分阶段链路：Qwen3.5 9B 关闭后再加载并关闭 Hy-MT2 7B；验证未下载模型。
- 0.4.0 最终发布回归为 `554 passed, 25 skipped, 21 warnings`；Ruff、生产包
  Pyright（`0 errors`）、CLI 版本输出和 `git diff --check` 通过。`uv build`
  成功生成 0.4.0 sdist/wheel；wheel 包含 TUI stylesheet、application/workflow
  模块和三个 console entrypoint。

## OpenLRC Mac 0.3.0

人工 Translation Brief、术语合规与可恢复事务式字幕编辑版本，并合入
此前未定版的 Pro、Chunk Planner v2、checkpoint 与 relaxed 字幕优化变更。

### 版本亮点

- **全新的 Pro 模式**：结合全文 Translation Brief、与原始分块严格对齐的
  ContextTimeline、Hy-MT2 翻译和保守终审；纯本地运行时任一时刻只驻留一个大模型。
- **术语词典服务**：可加载旧 mapping 或版本化词典，确定性合并任务与 Brief 术语，
  逐 occurrence 校验译法并输出可检查的合规报告，不会静默机械替换字幕。
- **可恢复的多轮编辑**：固定代码检查与有界语义审校统一通过指定行、整轮事务式
  Patch 修改，支持回滚、checkpoint、独立编辑入口和历史轮次 Restore。

### 新增

- 新增共享翻译 Chunk Planner 和稳定 chunk signature，保证 Timeline、翻译、
  fallback、review 和恢复使用完全相同的 segment 边界；contextual checkpoint
  升级为 schema v4，并支持兼容的 v2 Normal/Normal Plus 与 v3 Pro 状态验证迁移。
- 新增公开 `SubtitleOptimizationMode` 和 CLI
  `--subtitle-optimization aggressive|relaxed`。宽松模式保留 segment 数量、
  顺序、时间轴、短句、重复台词和 `<unk>` 内容。
- 新增版本化术语表 catalog、required/preferred 合规报告，以及
  `openlrc glossary validate|inspect|check` 命令。
- 新增稳定的 edit issue/patch/round/session 模型，覆盖结构、术语、人物/实体、数字、
  占位符和 immutable 字段检查；Normal Plus/Pro 默认一轮，硬上限三轮。
- 新增 `LRCer.edit(...) -> EditResult` 以及独立
  `verify|review|retranslate|restore` CLI，不重跑 ASR；离线动作不加载模型，模型
  动作必须显式提供当前配置；JSON/Markdown 报告与可选精简 Restore session 保存到
  用户输出目录。
- Hy-MT2 contextual 与独立编辑流程新增完整/Partial 人工 Translation Brief。
  `--brief-summary`、`--brief-characters JSON|@PATH` 和 `--brief-tone-style`
  会锁定人工字段，只让通用模型补齐缺失部分；三项完整时跳过自动 Brief 和自动
  术语提取。

### 变更

- Hy-MT2 contextual 对外模式更名为 `normal` / `normal-plus`；`context` /
  `context-plus` 在 0.3.x 兼容窗口作为 deprecated alias，并在业务逻辑与
  checkpoint fingerprint 前归一化。
- `run` 与 `translate` 统一增加 `--glossary`、`--force-glossary`、strictness、
  `--edit-rounds 0..3` 和 `--enable-restore`。
- 字幕 JSON 改为原子替换；编辑 checkpoint 成功后才提交译文。失败或 incomplete
  始终保留完整过程 checkpoint。
- Hy-MT2 的 retry、binary split 和 atomic fallback 均携带与 Pro chunk 对齐的
  story/scene；Timeline 必须回显精确 chunk ID 与有序 segment IDs，错配会触发重试。
  纯本地 Pro 按 `Qwen -> Hy-MT2 -> Qwen` 分阶段执行，可恢复翻译/review 阶段，并在
  source、模型、Planner 或 Timeline 过期时使 checkpoint 失效。
- 将 Translation Brief 明确定义为“源语言语义信息 + 双语人物/术语映射”。
  Timeline 只接收该源语言语义投影。Brief 不再生成角色描述、目标受众推测或翻译前
  ASR 解释；带置信度门槛的语言检查和 prompt version 会拒绝不兼容的派生状态，同时
  避免对短姓名、术语和不支持的语言作不可靠强判。
- 人工术语继续统一使用 `--glossary`，不新增重复入口。任务词典覆盖冲突的人工
  Characters 和自动 Brief 条目，Prompt、校验与报告共用同一份去重后的解析状态。

### 修复

- 术语合规改为按 occurrence 统计：同一源术语重复出现时，每一处都必须有独立的
  目标译法，不能由某一处译对后覆盖全部命中。大小写不敏感条目统一 casefold 后
  检查冲突，CJK 术语也可跨字幕行直接拼接命中，不再被人为插入的空格阻断。
- 编辑轮新增明确的 started/failed/committed/completed 生命周期。模型失败或无
  Patch 时恢复后重试当前轮；任一 chunk 失败会回滚该轮全部 Patch；Restore 的
  round 0 始终返回确定性修复前的原始 Hy-MT2 草稿。
- 独立编辑在模型工作前及每个已完成父 chunk 后保存完整进度；incomplete 必须保留，
  只有 complete 才清理。若 checkpoint 已提交但字幕原子替换被中断，下次可直接
  完成提交而不重复调用模型。
- 确定性检查统一识别 `%`、`percent`、`per cent`、全角百分号和“百分之”，同时
  不把 `%s` 等 printf placeholder 错当成百分比。
- Prompt 只使用一次解析后的任务/Brief 术语状态，避免重复注入并保持任务术语优先；
  定向重译的前文窗口改为受 token budget 限制的反向扫描，不再为每个 chunk 重建
  全部历史。
- `GlossaryOptions.report_matches` 现在确实控制报告/session 是否保留逐次匹配明细，
  同时不丢失指标和 issue。暂时移除尚无真实 validator 的
  `EditConfig.readability_checks`，避免公开无效 API。
- 重写 Chunk Planner v2：相同时间 gap 时优先选择最靠后的合法边界，不再连续产生
  大量单行 chunk；每次 split 和尾块处理都必须同时满足行数与 token 上限。Planner
  version/signature 变化会使不兼容的翻译和 review 进度失效。
- 新增 high-risk 修订译文专用校验器。空白、多行、标签、代码围栏、JSON 结构或内部
  协议文本会被拒绝；保留 Hy-MT2 草稿，并将该 review chunk 保持为 incomplete。
  checkpoint 重放还会校验 chunk 覆盖、reviewed/failed 索引、结果 ID、Timeline
  对齐、协议版本和原始/最终译文数量。

### 验证

- 完整 pytest 测试通过：`425 passed, 25 skipped`，另执行 261 个子测试。
- Ruff lint、touched-file format、touched-module Pyright 和 checkpoint 原子性
  测试通过。
- 使用 Qwen3.5 9B 和 Hy-MT2 7B Q6_K 验证真实本地 Normal 与 Pro，包括 relaxed
  优化、源语言 Brief/Timeline、分阶段模型关闭、行数/时间轴精确保留，以及无
  delimiter 或内部协议泄漏。31 行 Pro 样本生成对齐的 `30 + 1` Planner 分块，
  raw/final/review ID 全部完整。
- 真实验证完整人工 Normal Brief 可在不配置、不加载 context model 的情况下运行；
  短样本只加载一次 Hy-MT2 server，并在 15.64 秒内完成。
- 使用同一份 15 行、三个场景的本地样本比较四种模式：Fast、Normal、
  Normal Plus、Pro 分别耗时 21.27s、42.21s、70.56s、99.80s，模型加载次数
  为 1、2、3、3；四种模式均完整保留 15 行及时间轴，且无协议泄漏。
- 使用人物称谓、技术术语、金额日期和口语歧义样本验证编辑闭环：Normal Plus/Pro
  均达到 100% 术语合规和精确行数/时间轴保持，并明确报告 unresolved。独立
  retranslate/review/verify/Restore 保持全部未选行，round 0/1 均恢复预期快照。

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
