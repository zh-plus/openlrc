# OpenLRC Mac TUI v2 Reference

本文档记录当前 Textual TUI v2 的已实现行为。TUI 是本地交互入口，CLI 仍是自动化和
脚本调用入口；两者复用 `openlrc.application`、`openlrc.workflow` 和 `LRCer`，不维护
两套字幕 pipeline。

## 启动

完成 `uv sync` 后可使用任一入口：

```shell
uv run openlrc tui
uv run openlrc-tui
```

TUI 启动时不会自动下载模型或启动本地模型服务。Home 顶部的 `LOCAL READY`、
`LOCAL SETUP` 或 `LOCAL WORKING` 是本机资源和运行状态入口，按 `d` 或点击状态可打开
Doctor。

## Home

Home 固定为单屏左右布局：左侧是 OpenLRC 字符 Logo 和两句简介，右侧是五张纵向卡片。
五张卡片统一为四行高，并作为一个整体在右半区垂直居中；终端高度不足时才恢复为普通
纵向滚动。没有动态信息的卡片只显示标题，并放在两个内部格的上方格，避免终端无法使用
半格位置时形成靠下观感；有状态时使用标题加状态两行。Logo 的字符块与两句简介共用
左栏的同一水平中心轴。

- `New Work`：创建转写、源语言字幕、已有 JSON 翻译或完整字幕任务；
- `Edit Subtitle`：初版保持灰色 disabled，显示 `Coming later`；
- `Jobs & Recovery`：查看历史、日志、产物、术语检查和可恢复任务；
- `Models & Setup`：查看资源并运行 Whisper、Llama 或 All Setup；
- `Settings`：编辑 provider、模型、默认参数、术语表和外观 working copy。

卡片只在有可操作信息时显示第二行，例如 `1 needs review`、`2 failed`、
`Setup required` 或 `Credential error`。disabled Edit 不响应鼠标，也会被键盘焦点跳过。

## 键盘与鼠标

列表行和 Home 卡片共用同一 action ID，因此 Enter、Space 和鼠标点击执行相同动作。
普通列表只有标题时占一行，包含当前值或说明时占两行。分组标题后以及同组相邻 action
之间各有一行空白；这些 spacer 不可聚焦、不响应鼠标，并沿用分组的左右结构边框。
分组末项额外保留底边框，非末尾分组之间只留一行空白，最后一个分组没有多余尾部
margin。运行时新增或清空 detail 会立即重排该行，不需要重组整个页面。

| 输入 | 行为 |
| --- | --- |
| `Up` / `Down` 或 `k` / `j` | 循环移动选择并跳过 disabled/只读行 |
| `Home` / `End` | 跳到第一个/最后一个可用项 |
| `Enter` 或 `Space` | 打开、选择或切换当前行 |
| `Esc` | 返回上一层；dirty Settings 会先确认 |
| `n` | New Work |
| `d` | Doctor |
| `g h` | 返回 Home |
| `g j` | 打开 Jobs & Recovery |
| `?` | 键盘帮助 |
| `q` | 确认退出 |
| `Ctrl-C` | 确认取消活动任务；空闲时确认退出 |

文本输入和多行编辑获得焦点时，`q`、`d`、`n`、`j`、`k` 等单字符快捷键会自动停用，
避免输入内容触发页面跳转。

## New Work

初版提供四种任务：

1. `Transcribe to JSON`：只保留原始 transcription JSON；
2. `Transcribe to subtitle`：生成源语言 `.lrc` / `.srt`，不翻译；
3. `Translate existing JSON`：翻译 OpenLRC transcription JSON；
4. `Full run`：转写并翻译媒体文件。

主标题只保留任务名，完整用途放在下一行；`Transcribe`、`Translate` 和 `Full run`
使用 amber 强调。向导顶部固定显示 `Choose work -> Configure -> Preflight` 三步状态。
参数、输入、引擎、Models、Jobs 和 Settings 使用不可聚焦的分组标题与单层线框；参数页
至少区分 `Basic settings`、`Advanced settings` 和 `Actions`，其中
`Continue to Preflight` 是独立主动作。

翻译任务可选 Online Provider classic、Local Qwen classic，或 Hy-MT2 的 Fast、Normal、
Normal Plus、Pro 四种公开模式。配置页按当前 engine/mode 显示 provider、模型、重试、
Context、Translation Brief、术语表、字幕格式和清理选项。

Hy-MT2 Normal 和零轮语义审查的 Normal Plus 提供 `Context assistance`：

- `Auto` 在 Brief 缺失或不完整时显示并要求 Context provider/model；
- `Off` 隐藏 Context provider/model，要求人工 summary；空 Characters 显示
  `None · explicit`，空 Tone 显示 `No additional guidance`；
- Pro 固定显示 `Required by Pro`；带语义审查的 Normal Plus 固定显示
  `Required by semantic review`，两者都不能选择 Off；
- 从 Off 切换到 Pro 会恢复 Auto；Off 下把 Normal Plus 语义轮数改为大于零会被拒绝，
  不会静默改变配置。

Off 的有效 request 不携带隐藏的 Context 配置，但 Draft 会暂存原 provider/model，便于
切回 Auto。共享 Preflight 在此路径显示 `Context model: Off · Manual Brief`。
新 Draft 在 Auto 需要 Context 时默认使用 `local` provider 和 Settings 当前的 Qwen
model；若该设置为空，则使用内置 `qwen3.5-9b` profile。Resume 旧历史时，缺少的
Context provider/model 也按这一当前默认值补齐。

转写配置提供 `Whisper GPU` 与 `Whisper flash attention` 开关。二者默认开启；当本机
whisper.cpp、Metal 或特定模型组合不兼容时，可以在单个 Workflow 中关闭，也可以在
Settings 的 Transcription 默认值中统一关闭。

文件输入支持 macOS native picker、内置 Terminal Browser 和粘贴多行路径。启动前必须
经过共享 Preflight；Blocked 不能启动，Warning 需要确认，Ready 才直接进入 Running。
请求摘要和检查结果是不可聚焦的只读文本，只有 `Start` 和 `Back` 可选择；内容超过可见
区域时使用 `Page Up` / `Page Down` 翻阅，action 焦点不会移动。TUI 不自行推测输出路径
或复制 CLI 参数拼装。

只打开再退出 New Work 不会保留空 Draft，也不会在下次进入时弹出恢复提示。只有输入、
参数或顺序相对初始 baseline 真正发生变化时，才提供 Continue/Discard。

单行和多行输入弹窗使用与页面一致的列表式 `Apply` / `Cancel`。Brief summary、
Character mapping 和 tone/style 使用支持软换行的多行编辑器，弹窗内持续显示
`Ctrl+Enter Apply · Esc Cancel · Tab Actions · Shift+Tab Editor`。`Tab` /
`Shift+Tab` 在编辑区与 Actions 间移动，Actions 内方向键循环，`Enter` / `Space`
激活；`Esc` 从编辑区直接取消，`Ctrl+Enter` 从编辑区直接应用。80×24 下编辑区、
错误提示、两个动作和底边框保持同屏可见，API key 输入保持遮罩。

Character mapping 每行使用 `Source Name = Target Name`。提交时会先使用与最终 Draft
构造相同的解析器校验；无效行会显示行号、将编辑区边框标红并保持焦点，当前 Draft
不会改变。开始修改后错误提示会清除，只有全部映射有效时才关闭弹窗并刷新参数页。

## Running 与 Jobs

Workflow 和 Setup 共用单活动 operation 边界。Running 页面显示阶段、进度、逐文件状态、
日志和产物；取消会调用共享 token/owned-process cleanup，只有底层返回终态后才显示
Cancelled。TUI 在调度 worker 前同步创建 token，因此 Starting 阶段、controller 尚未
注册时的取消也会跳过 Workflow/Setup 的实际工作。启动前异常也会留在可见的 Failed
页面，不会停在 Starting；完成或启动失败都会清除活动 token。

Workflow 运行期间，OpenLRC logger 输出会进入页面下方固定的 `RUNTIME OUTPUT` 框，
不会直接写入 terminal 并覆盖 Textual 画面；原 terminal handler 在退出 TUI 后恢复。
任务进入终态后，Esc / `g h` 直接返回 Home，Jobs / New Work 则进入对应页面；这些
路径都会先移除本次 Workflow 的整个配置栈并丢弃已消费 Draft。

Jobs 历史不保存 API key。详情页提供：

- 输入、状态、阶段、模型生命周期、日志和 outputs；
- Failed、Cancelled、Interrupted、Needs Review 任务的 Resume；
- effective glossary terms、violations 和 report artifact；
- 删除历史前确认；正在运行的记录不能删除。

Resume 会恢复为新 Draft 和新 Job ID，并返回 Confirm 重新检查，不会绕过 Preflight。
若当前已有 dirty Draft，详情页先显示 `Discard Draft and Resume` /
`Keep Current Draft`；确认前不修改 Draft、baseline、路径或当前页面，Keep/取消继续
停留在 Job Detail，只有 Discard 才加载历史 recipe 并记录 `resumed_from`。

## Models、Doctor 与 Settings

Models 使用 typed resource status，不解析 CLI stdout。Setup Whisper、Llama 和 All 通过
typed setup event/result/cancel contract 运行，下载、编译或覆盖前显示确认，Setup 与字幕
Workflow 不能并行占用可变资源。

Settings 是 working copy：子页修改只保存在内存，必须选择 `Save changes` 才原子写入
`settings.json`。离开 dirty Settings 时可 Save、Discard 或 Stay。API key 写入 macOS
Keychain，不进入 settings 或 job recipe；environment credential 只读取、不删除。
Settings 根页面的 Esc、`g h` 和 Quit 共用同一个离开保护；子页 Esc 只返回 Settings
根页面，不询问。Save 只有写盘成功才执行原导航，失败会保留 dirty working copy 和当前
页面；Discard 恢复已保存设置后执行，Stay/关闭弹窗不执行。
没有任何修改时，Save 和 Discard 都保持灰色 disabled，Save 不继承主动作的 amber；
working copy 变脏后 Save 才恢复 amber 强调和键盘焦点。disabled 只改变文字和交互
状态，Save、spacer 与同组其他行始终使用相同主题结构边框，外框不会中途断色。

Default Glossary 在 Settings 中 validate/inspect；生成结果是否违反术语约束在对应 Job
详情中检查。Home 不设置独立 Glossary Tools 卡片。

## 外观与兼容性

Appearance 提供 `OpenLRC Dark`、Textual 内置 `Textual Dark`、English、简体中文、
Logo animation、Reduced motion 和 ASCII-only 状态符号。Theme 与 Language 选择后
立即预览；只有 Settings 根页面的 `Save changes` 会持久化，`Discard changes` 会同时
恢复已保存的画面和 working copy。旧设置文件没有 language 时继续使用 English。

Theme、Logo animation、Reduced motion 和 ASCII 切换不会销毁当前列表；Language
必须刷新全部文案，但会通过 stable action ID 在新列表挂载后恢复 Language 行焦点。
因此用 Enter 应用主题或语言后，可以立刻继续使用方向键，不需要鼠标重新聚焦。

简体中文覆盖 Home、New Work、Preflight、Running、Jobs、Models、Doctor、Settings、
Modal、帮助、通知和 TUI 动态计数。路径、模型 ID、Provider 名称、后端日志与原始异常
保持原文，便于诊断。

Logo 根据终端宽度使用 wide、compact 或 minimal 版本。wide/compact 流光按终端
`x` 坐标移动，同一列的所有非空字符同时变色，不按扁平字符串逐字游走。可关闭动画；
关闭动画或启用 Reduced motion 会重新渲染无光带的静态亮蓝 Logo，不会把黄色流光
冻结在最后一帧；`NO_COLOR` 或无颜色终端使用灰阶退化。wide/compact 每一行会先补齐
到相同 terminal-cell 宽度，再由占满左栏内容宽度的 Logo widget 居中整块字符矩阵。
这既保证所有行共享水平起点，也使 Logo 与下方两句简介共用中心轴。动画状态不参与
业务逻辑。
Home 已为 80×24、100×30 和 160×50 建立真实 Textual SVG 基线，另有两套 Appearance
主题、简体中文 Home、静态 Logo 和连续 wide fixed-frame baseline。

`Edit Subtitle` 不属于 v2 初版。在 standalone Edit 获得 typed request/result/event、
取消和 owned-resource cleanup 前，它会一直保持 disabled，不能通过 TUI 转发 CLI 命令
来假装支持。
