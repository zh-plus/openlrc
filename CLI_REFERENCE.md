# OpenLRC Mac CLI Reference

本文档记录当前第一阶段 CLI 的全部命令和行为。CLI 使用 `Typer + Rich`
实现，入口来自 `pyproject.toml` 的 `[project.scripts]`：

```shell
openlrc
openlrc-mac
```

两个入口等价。下面 CLI 示例统一使用 `uv run openlrc ...`。

## 总览

```shell
uv run openlrc --help
uv run openlrc --version
uv run openlrc tui
uv run openlrc doctor [--strict]
uv run openlrc models status
uv run openlrc glossary validate|inspect|check [OPTIONS]
uv run openlrc setup whisper [OPTIONS]
uv run openlrc setup llama [OPTIONS]
uv run openlrc setup all [OPTIONS]
uv run openlrc transcribe PATH... [OPTIONS]
uv run openlrc translate JSON... --translation local|online [OPTIONS]
uv run openlrc run PATH... [OPTIONS]
uv run openlrc edit --source SOURCE.json --target TARGET.json --action verify|review|retranslate|restore [OPTIONS]
```

裸 `openlrc` 会显示 CLI help；`openlrc tui` 会启动独立的 Textual TUI v2。TUI 的完整
页面、快捷键和功能边界见 [TUI_REFERENCE.md](TUI_REFERENCE.md)。

## 基础命令

### `openlrc tui`

启动键盘优先、鼠标等价操作的 Textual TUI v2。以下两个入口等价：

```shell
uv run openlrc tui
uv run openlrc-tui
```

该命令进入交互界面，不改变其他 CLI 命令的参数或脚本语义。

### `openlrc --version`

显示 OpenLRC Mac 版本、发行包名和上游 OpenLRC 基线版本。

```shell
uv run openlrc --version
```

当前输出类似：

```text
OpenLRC Mac 0.4.1 (distribution: openlrc-mac; upstream base: OpenLRC 1.7.0a1)
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

- `whisper-cli`
- 默认 Whisper 模型
- 默认 Whisper VAD 模型
- 默认本地 Qwen GGUF 模型
- 默认 Hy-MT2 7B GGUF 模型
- `llama-server`

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
uv run openlrc setup llama --local-model-profile hy-mt2-7b
```

参数说明：

- `--local-model-profile qwen3.5-9b|hy-mt2-7b|hy-mt2-30b-a3b`: 使用已注册 profile 的下载默认值；`hy-mt2-30b-a3b` 不提供默认下载。
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

`transcribe`、`translate` 和 `run` 现在共用 `openlrc.workflow` 执行层。Typer
参数和最终 Generated Files 表保持不变；交互式终端可显示阶段/模型进度，重定向
或测试捕获等非交互输出不会写动态控制字符。`Ctrl-C` 会先请求协作取消并清理
OpenLRC-owned 子进程，然后以退出码 130 结束。失败返回 1；Normal Plus/Pro 的
review incomplete 仍生成可用结果并返回 0，同时保留恢复材料。

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
- `--subtitle-optimization aggressive|relaxed`: 字幕优化模式，默认
  `aggressive`。`relaxed` 不合并、删除或截断字幕，也不改变时间边界。
- `--keep-checkpoint`: 调试时保留已经完成的翻译 checkpoint；默认在完整
  成功后删除，review 不完整时始终保留。
- `--glossary`: legacy mapping 或 versioned catalog 格式的任务术语表 JSON。
- `--force-glossary`: 仅把已启用的任务术语提升为 required；不会机械替换译文。
- `--glossary-strict/--no-glossary-strict`: 同级冲突报错或确定性首项优先。
- `--brief-summary TEXT`: 人工提供全文摘要；不能为空。
- `--brief-characters JSON|@PATH`: 人工人物译名数组，可直接传 JSON，或用
  `@PATH` 读取 UTF-8 JSON 文件。
- `--brief-tone-style TEXT`: 人工提供语气/风格；空字符串表示明确不设置风格。
- `--edit-rounds 0..3`: Normal Plus/Pro 的语义编辑轮数；`0` 只运行确定性阶段。
- `--enable-restore`: 成功后在输出目录保存精简 `.edit-session.json` 并清理过程 checkpoint。
- `--llama-model`: 本地 GGUF 模型 alias、文件名或路径，默认 `qwen3.5-9b`。
- `--local-model-profile`: 本地 LLM profile，决定采样参数和 prompt；支持 `qwen3.5-9b`、`hy-mt2-7b`、`hy-mt2-30b-a3b`。
- `--llama-port`: 本地 `llama-server` 端口，默认 `8088`。
- `--idle-timeout`: OpenLRC 自己启动的本地 server 空闲关闭时间，默认 `300` 秒。
- `--hy-mt2-mode fast|normal|normal-plus|pro`: Hy-MT2 管线模式，默认 `fast`；
  `context` / `context-plus` 是 0.3.x deprecated alias。
- `--context-assistance auto|off`: Context model 使用策略，默认 `auto`。Fast
  不使用 Context/Brief；`off` 只允许 Normal 和零轮语义审查的 Normal Plus 使用
  完整人工 Brief。
- `--context-provider openai|anthropic|google|litellm|third-party|local`: normal/normal-plus/pro 使用的通用模型 provider。
- `--context-model`: 通用模型名；local 时可传 Qwen alias、GGUF 文件名或路径。
- `--context-base-url`: 自定义 OpenAI-compatible context endpoint；`third-party` 时必填。
- `--context-fee-limit`: 通用模型单次调用费用上限，默认 `0.8` 美元。

`--translation local` 默认使用本地 `llama.cpp` / Qwen。传
`--local-model-profile hy-mt2-7b`，或在未传 profile 时传
`--llama-model hy-mt2-7b`，会使用 Hy-MT2 7B Q6_K 的官方参数和 prompt。
`hy-mt2-30b-a3b` 必须同时通过 `--llama-model` 指向本地已转换 GGUF 文件。
`--translation online` 会使用 classic OpenLRC 上下文管线，需要用户自己设置对应 provider 的 API key。
默认本地 Qwen profile 同样使用 classic 管线。底层 lean translator 不再作为
CLI 配置暴露，只由 Hy-MT2 四种产品模式内部使用。

Hy-MT2 四种模式：

- `fast`: delimiter 翻译，不运行 CR，保持 0.2.1 默认行为。
- `normal`: 通用模型生成结构化 Translation Brief，然后由 Hy-MT2 翻译。
- `normal-plus`: 在 Normal 基础上执行确定性定向修复，再由同一个通用模型做事务式语义编辑。
- `pro`: 通用模型先生成全局 Brief 和与翻译边界严格对齐的逐 chunk
  ContextTimeline，再由 Hy-MT2 翻译，最后执行同样的全量 high-risk review。

Context 模式的 Translation Brief 采用“源语言语义信息 + 双语人物/术语映射”：
summary、glossary note 和可选风格提示保持源语言，只有人物 `target_name` 和术语
目标译法使用目标语言。人物只保留原名到目标译名的映射；人物描述、目标受众推测和
翻译前 ASR 解释不进入 Brief、Timeline 或 Hy-MT2 prompt。Timeline 不接收目标语言
或目标映射；Brief、Timeline 和 Hy-MT2 的控制指令固定为英文。

人工 Brief 的三个字段为 summary、characters 和 tone/style。提供其中一部分时，
这些字段会整段锁定，通用模型只补全缺失字段；三项全部提供时直接使用完整人工
Brief，跳过自动 Brief 和自动术语提取。`characters=[]` 表示明确不要人物，
`tone_style=""` 表示明确不设置风格；summary 不允许为空。人物数组格式严格为：

```json
[
  {"source_name": "John", "target_name": "强尼"}
]
```

人工术语不放进 Brief 参数，仍统一使用 `--glossary`。任务词典优先级高于人工
Characters 和自动 Brief；相同源名称冲突时以任务词典为准，并在最终 Prompt 中去重。

`--context-assistance auto` 保持原有行为：Normal 的无 Brief/Partial Brief 需要
context model；Normal Plus 在 Brief 不完整或 `--edit-rounds` 大于零时需要；
Pro 始终需要它生成 Timeline。完整人工 Brief 可让 Normal 跳过 context model，
Normal Plus 仅在 `--edit-rounds 0` 时可跳过。

`--context-assistance off` 是显式的“只使用人工 Brief”模式。除 Fast 外，它要求
`--brief-summary`；未提供/空的 characters 会规范化为 `[]`，未提供/空的 tone/style
会规范化为 `""`，因此用户可以明确表示没有人物映射或额外风格要求。Off 不允许同时
传入 Context provider/model/base URL；Pro 和带语义审查的 Normal Plus 会给出明确
错误。非 Hy-MT2 后端也拒绝 Off。

纯本地模式严格按“通用模型 -> 卸载 -> Hy-MT2”顺序运行；
Normal Plus 和 Pro 翻译后再卸载 Hy-MT2 并重新加载通用模型。Pro 在第一次
通用模型驻留期间顺序生成完整 Timeline，并按 chunk 原子保存进度。Hy-MT2
不提供 classic/lean engine 选择。

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
- `--subtitle-optimization aggressive|relaxed`: 字幕优化模式。`aggressive`
  保留既有合并、重复压缩、截断、`<unk>`/空行清理和最终时间扩展；`relaxed`
  只做不改变 segment 对齐的空白及目标语言标点整理。
- `--clear-temp/--keep-temp`: 完整成功后默认清理当前输入的临时文件；
  `--keep-temp` 可保留。若 Normal Plus 或 Pro 编辑不完整，则无论此选项如何都会
  保留 checkpoint 供下次运行恢复。
- `--skip-preprocess`: 使用已经存在的预处理音频文件。
- `--llama-model`: 本地 GGUF 模型 alias、文件名或路径。
- `--local-model-profile`: 本地 LLM profile，决定采样参数和 prompt。
- `--llama-port`: 本地 `llama-server` 端口，默认 `8088`。
- `--idle-timeout`: OpenLRC 自己启动的本地 server 空闲关闭时间，默认 `300` 秒。
- Hy-MT2 的 `--hy-mt2-mode`、`--context-assistance` 和其他 `--context-*`
  参数与 `translate` 命令一致。
- `--glossary`、`--force-glossary`、`--glossary-strict`、`--edit-rounds`
  和 `--enable-restore` 与 `translate` 命令一致。
- `--brief-summary`、`--brief-characters` 和 `--brief-tone-style` 与
  `translate` 命令一致；Fast、非 Hy-MT2 以及未启用翻译时拒绝这些参数。

### `openlrc glossary`

术语表命令全程离线，不加载翻译模型：

```shell
uv run openlrc glossary validate glossary.json --source-language en --target-language zh-cn
uv run openlrc glossary inspect glossary.json [--force] [--no-strict]
uv run openlrc glossary check glossary.json --source source.json --target translated.json
```

- `validate` 检查文件、JSON、schema/version、空值、语言声明和冲突。
- `inspect` 显示规范化后的条目、required 状态、alias、fingerprint 和冲突数。
- `check` 生成 `<target>.glossary-report.json`；required 术语不合规时返回非零退出码。

Versioned catalog 的最小格式：

```json
{
  "schema_version": 1,
  "source_language": "en",
  "target_language": "zh-cn",
  "entries": [
    {
      "source": "case",
      "target": "案件",
      "aliases": ["cases"],
      "accepted_targets": ["案子"],
      "required": true
    }
  ]
}
```

### `openlrc edit`

独立编辑直接读取源文和现有译文 JSON，不重新运行 ASR：

```shell
uv run openlrc edit --source source.json --target translated.json --action verify
uv run openlrc edit --source source.json --target translated.json \
  --action review --ids 12,18-24 --context-provider local --context-model qwen3.5-9b
uv run openlrc edit --source source.json --target translated.json \
  --action retranslate --ids 12,18-24 --llama-model hy-mt2-7b
uv run openlrc edit --source source.json --target translated.json \
  --action restore --round 0 --session translated.edit-session.json
```

- `verify`: 只运行结构、术语、人物/实体、数字、金额、日期和占位符检查；不加载模型。
- `review`: 只审查 `--ids`，必须显式配置 `--context-provider` 与 `--context-model`。
- `retranslate`: 只重译 `--ids`，必须显式配置 Hy-MT2；Normal/Normal Plus 在
  Auto + Brief 不完整时需要 context model；也可按上述规则使用
  `--context-assistance off`。Pro 始终需要 context model 生成 Timeline。
- `restore`: 从稳定 edit session 恢复指定轮次；会校验源文、当前译文、行数和时间轴 fingerprint。

`review` 和 `retranslate` 接受与翻译入口相同的三个人工 Brief 参数；完整 Brief
只省略前置 Brief 生成，不替代 review 或 Pro Timeline 所需的模型。`verify` 和
`restore` 不消费 Brief，因此传入这些参数会报错。`--context-assistance off`
只适用于 `edit retranslate`；review、verify、restore 和其他 edit action 会拒绝它。

每次命令默认写 `<base>.edit-report.json`；`--markdown-report` 改写 Markdown
报告。`verify` 出现 error severity 时返回非零退出码。事务失败时不提交任何 Patch。

`run` 和 `transcribe` 的核心区别：

- `transcribe` 只输出转写 JSON。
- `run --translation none` 输出原语言 `.lrc` / `.srt` 成品字幕。
- `run --translation local` 输出本地翻译后的 `.lrc` / `.srt` 成品字幕。

## 本地 llama-server 生命周期

当使用 `--translation local` 时，OpenLRC 会启动或复用本地 `llama-server`。

当前行为：

- 默认地址：`http://127.0.0.1:8088/v1`。
- 如果 8088 上已有匹配 model alias 的 server，OpenLRC 会复用它。
- 如果 server 是 OpenLRC 自己启动的，CLI workflow 会在成功、失败或取消后的
  cleanup 中主动关闭。
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
