# Open-Lyrics

[![PyPI](https://img.shields.io/pypi/v/openlrc)](https://pypi.org/project/openlrc/)
[![PyPI - License](https://img.shields.io/pypi/l/openlrc)](https://pypi.org/project/openlrc/)
[![Downloads](https://static.pepy.tech/badge/openlrc)](https://pepy.tech/project/openlrc)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/zh-plus/Open-Lyrics/ci.yml)

Open-Lyrics is a Python library that transcribes audio with
[faster-whisper](https://github.com/guillaumekln/faster-whisper), then translates/polishes the text
into `.lrc` subtitles with LLMs such as
[OpenAI](https://github.com/openai/openai-python) and [Anthropic](https://github.com/anthropics/anthropic-sdk-python).

#### Key Features

- Audio preprocessing to reduce hallucinations (loudness normalization and optional noise suppression).
- Context-aware translation to improve translation quality.
  Check [prompt](https://github.com/zh-plus/openlrc/blob/master/openlrc/prompter.py) for details.
- **Lean translation mode** for token-efficient translation with mixed-model support (e.g. cheap MT model + larger CR model).
- Check [here](#how-it-works) for an overview of the architecture.

## Installation ⚙️

1. Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) according
   to https://opennmt.net/CTranslate2/installation.html to enable `faster-whisper`.

   `faster-whisper` also needs [cuBLAS](https://developer.nvidia.com/cublas) installed.
   <details>
   <summary>For Windows Users (click to expand)</summary> 

   (Windows only) You can download the libraries from Purfview's repository:

   Purfview's [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) provides the required NVIDIA
   libraries for Windows in a [single archive](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs).
   Decompress the archive and place the libraries in a directory included in the `PATH`.

   </details>


2. Add LLM API keys (recommended for most users: `OPENROUTER_API_KEY`):
   - Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.
   - Add your [Anthropic API key](https://console.anthropic.com/settings/keys) to environment variable
     `ANTHROPIC_API_KEY`.
   - Add your [Google API Key](https://aistudio.google.com/app/apikey) to environment variable `GOOGLE_API_KEY`.
   - Add your [OpenRouter API key](https://openrouter.ai/keys) to environment variable `OPENROUTER_API_KEY`.

3. Install [ffmpeg](https://ffmpeg.org/download.html) and add `bin` directory
   to your `PATH`.

4. Install from PyPI:

    ```shell
    pip install openlrc
    ```

   or install directly from GitHub:

    ```shell
    pip install git+https://github.com/zh-plus/openlrc
    ```

5. **(Optional)** If you need noise suppression (`noise_suppress=True`), install the full extras
   which includes DPDFNet:

    ```shell
    pip install 'openlrc[full]'
    ```

   If you want to route translation through LiteLLM, install the LiteLLM extra:

    ```shell
    pip install 'openlrc[litellm]'
    ```

## Lightweight Imports

OpenLRC keeps several package-root APIs lightweight to import.

The following imports are guaranteed not to eagerly load heavyweight runtime dependencies such as
`dpdfnet`, `spacy`, `faster-whisper`, `tiktoken`, or `lingua`:

```python
import openlrc
from openlrc import LRCer
from openlrc import TranscriptionConfig, TranslationConfig
from openlrc import ModelConfig, ModelProvider, list_chatbot_models
```

This is useful when you only need configuration objects, model metadata, or the `LRCer` type itself
without immediately starting transcription or language-processing work.

Heavy dependencies are loaded only when the corresponding features are first used. For example:

- `faster-whisper` is loaded when transcription is first needed.
- `dpdfnet` is loaded when noise suppression is used.
- `spacy` is loaded when sentence segmentation or related NLP helpers are used.
- `tiktoken` is loaded when token counting is used.
- `lingua` is loaded when language detection helpers are used.

> [!NOTE]
> The base `pip install openlrc` does **not** include dpdfnet.
> It is only installed with `pip install 'openlrc[full]'` and is only needed
> for noise suppression (`noise_suppress=True`). On first use, DPDFNet downloads
> its model (default `dpdfnet2_48khz_hr`, ~10MB) to `~/.cache/dpdfnet/models`.

## Usage 🐍

[//]: # (### GUI)

[//]: # ()

[//]: # (> [!NOTE])

[//]: # (> We are migrating the GUI from streamlit to Gradio. The GUI is still under development.)

[//]: # ()

[//]: # (```shell)

[//]: # (openlrc gui)

[//]: # (```)

[//]: # ()

[//]: # (![]&#40;https://github.com/zh-plus/openlrc/blob/master/resources/streamlit_app.jpg?raw=true&#41;)

### Python code

```python
from openlrc import LRCer, ModelConfig, ModelProvider, TranscriptionConfig, TranslationConfig

if __name__ == '__main__':
    lrcer = LRCer()

    # Single file
    lrcer.run('./data/test.mp3',
              target_lang='zh-cn')  # Generate translated ./data/test.lrc with default translate prompt.

    # Multiple files
    lrcer.run(['./data/test1.mp3', './data/test2.mp3'], target_lang='zh-cn')
    # Note we run the transcription sequentially, but run the translation concurrently for each file.

    # Path can contain video
    lrcer.run(['./data/test_audio.mp3', './data/test_video.mp4'], target_lang='zh-cn')
    # Generate translated ./data/test_audio.lrc and ./data/test_video.srt

    # Use glossary to improve translation
    lrcer = LRCer(translation=TranslationConfig(glossary='./data/aoe4-glossary.yaml'))

    # To skip translation process
    lrcer.run('./data/test.mp3', target_lang='en', skip_trans=True)

    # Change asr_options or vad_options (see openlrc.defaults for details)
    vad_options = {"threshold": 0.1}
    lrcer = LRCer(transcription=TranscriptionConfig(vad_options=vad_options))
    lrcer.run('./data/test.mp3', target_lang='zh-cn')

    # Enhance the audio using noise suppression (requires openlrc[full], consumes more time).
    lrcer.run('./data/test.mp3', target_lang='zh-cn', noise_suppress=True)

    # Change the translation model
    lrcer = LRCer(translation=TranslationConfig(
        chatbot=ModelConfig(provider=ModelProvider.ANTHROPIC, name='claude-3-sonnet-20240229')
    ))
    lrcer.run('./data/test.mp3', target_lang='zh-cn')

    # Clear temp folder after processing done
    lrcer.run('./data/test.mp3', target_lang='zh-cn', clear_temp=True)

    # Use a custom OpenAI-compatible endpoint
    lrcer = LRCer(
        translation=TranslationConfig(
            chatbot=ModelConfig(
                provider=ModelProvider.OPENAI,
                name='gpt-4.1-nano',
                base_url='https://example.com/v1',
                api_key='token',
            )
        )
    )

    # Route through LiteLLM (requires openlrc[litellm])
    lrcer = LRCer(translation=TranslationConfig(
        chatbot=ModelConfig(provider=ModelProvider.LITELLM, name='openai/gpt-4o')
    ))

    # Bilingual subtitle
    lrcer.run('./data/test.mp3', target_lang='zh-cn', bilingual_sub=True)

    # Lean translation mode (token-efficient, simplified prompts)
    lrcer = LRCer(translation=TranslationConfig(translate_mode='lean'))
    lrcer.run('./data/test.mp3', target_lang='zh-cn')

    # Lean mode with mixed-model architecture (separate CR and translation models)
    from openlrc.agents import create_chatbot
    from openlrc.translate import LeanTranslator

    mt_bot = create_chatbot(ModelConfig(
        provider=ModelProvider.OPENAI, name='your-mt-model',
        base_url='http://localhost:8000/v1', api_key='token',
    ))
    cr_bot = create_chatbot(ModelConfig(
        provider=ModelProvider.OPENAI, name='your-cr-model',
        base_url='http://localhost:8001/v1', api_key='token',
    ))
    translator = LeanTranslator(chatbot=mt_bot, cr_chatbot=cr_bot, enable_cr=True)
    translations = translator.translate(['Hello', 'World'], 'en', 'zh')
```

`LRCer` supports the context manager protocol, which automatically closes
the underlying LLM connections when the block exits:

```python
with LRCer() as lrcer:
    lrcer.run(['./data/file1.mp3', './data/file2.mp3'], target_lang='zh-cn')
# Connections are closed automatically here.
```

This is recommended when processing multiple files, as the LLM connection
pool is shared across all files within the same `LRCer` instance.

Check more details in [Documentation](https://zh-plus.github.io/openlrc/#/).

### Glossary

Add glossary to improve domain specific translation. For example `aoe4-glossary.json`:

```json
{
  "aoe4": "帝国时代4",
  "feudal": "封建时代",
  "2TC": "双TC",
  "English": "英格兰文明",
  "scout": "侦察兵"
}
```

```python
lrcer = LRCer(translation=TranslationConfig(glossary='./data/aoe4-glossary.json'))
lrcer.run('./data/test.mp3', target_lang='zh-cn')
```

To keep `TranslationConfig` serialization-friendly, save in-memory glossary data to
a JSON file and pass the file path via `TranslationConfig(glossary=...)`.

## Pricing 💰

*pricing data from [OpenAI](https://openai.com/pricing)
and [Anthropic](https://docs.anthropic.com/claude/docs/models-overview#model-comparison)*

| Model Name                   | Pricing for 1M Tokens <br/>(Input/Output) (USD) | Cost for 1 Hour Audio <br/>(USD) |
|------------------------------|-------------------------------------------------|----------------------------------|
| `gpt-3.5-turbo`              | 0.5, 1.5                                        | 0.01                             |
| `gpt-4o-mini`                | 0.5, 1.5                                        | 0.01                             |
| `gpt-4-0125-preview`         | 10, 30                                          | 0.5                              |
| `gpt-4-turbo-preview`        | 10, 30                                          | 0.5                              |
| `gpt-4o`                     | 5, 15                                           | 0.25                             |
| `claude-3-haiku-20240307`    | 0.25, 1.25                                      | 0.015                            |
| `claude-3-sonnet-20240229`   | 3, 15                                           | 0.2                              |
| `claude-3-opus-20240229`     | 15, 75                                          | 1                                |
| `claude-3-5-sonnet-20240620` | 3, 15                                           | 0.2                              |
| `gemini-1.5-flash`           | 0.175, 2.1                                      | 0.01                             |
| `gemini-1.0-pro`             | 0.5, 1.5                                        | 0.01                             |
| `gemini-1.5-pro`             | 1.75, 21                                        | 0.1                              |
| `deepseek-chat`              | 0.18, 2.2                                       | 0.01                             |

**Note the cost is estimated based on the token count of the input and output text.
The actual cost may vary due to the language and audio speed.**

### Recommended translation model

For English audio, we recommend `deepseek-chat`, `gpt-4o-mini`, or `gemini-1.5-flash`.

For non-English audio, we recommend `claude-3-5-sonnet-20240620`.

## How it works

![](https://github.com/zh-plus/openlrc/blob/master/resources/how-it-works.png?raw=true)

To maintain context between translation segments, the process is sequential for each audio file.


[//]: # (## Comparison to https://microsoft.github.io/autogen/docs/notebooks/agentchat_video_transcript_translate_with_whisper/)

## Development Guide

This project uses [uv](https://github.com/astral-sh/uv) for package management.
Install `uv` with the standalone installer:

#### On macOS and Linux

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```shell
uv venv
uv sync
```

### Code quality checks

Before committing, please make sure the following checks pass locally:

```shell
# Lint
uv run ruff check openlrc/ tests/

# Format
uv run ruff format --check openlrc/ tests/
# To auto-fix formatting:
# uv run ruff format openlrc/ tests/

# Type check
uv run pyright openlrc/

# Tests
uv run --extra full --extra litellm pytest
```

For live translation testing as a developer (and for CI usage), set:

```shell
export DEEPSEEK_API_KEY="your-api-key"
export OPENLRC_TEST_LIVE_API=1
```

Live tests use the OpenAI-compatible `deepseek-v4-flash` model by default. See
`tests/conftest.py` for configurable environment variables (e.g.
`OPENLRC_TEST_LLM_BASE_URL` to point at another compatible test endpoint).

### Build and publish a release

Use `uv` end-to-end for release builds and publishing:

```shell
# Build source and wheel distributions
uv build

# Validate the generated metadata before uploading
uvx twine check dist/*

# Publish to PyPI
# Preferred for local publishing:
uv publish
#
# Or publish with an explicit token:
# uv publish --token <pypi-token>
```

If you prefer GitHub Actions publishing, configure PyPI trusted publishing for this repository and push a version tag such as `v1.7.0a2`.

## Todo

- [x] [Efficiency] Batched translate/polish for GPT request (enable contextual ability).
- [x] [Efficiency] Concurrent support for GPT request.
- [x] [Translation Quality] Make translate prompt more robust according to https://github.com/openai/openai-cookbook.
- [x] [Feature] Automatically fix json encoder error using GPT.
- [x] [Efficiency] Asynchronously perform transcription and translation for multiple audio inputs.
- [x] [Quality] Improve batched translation/polish prompt according
  to [gpt-subtrans](https://github.com/machinewrapped/gpt-subtrans).
- [x] [Feature] Input video support.
- [X] [Feature] Multiple output format support.
- [x] [Quality] Speech enhancement for input audio.
- [ ] [Feature] Preprocessor: Voice-music separation.
- [ ] [Feature] Align ground-truth transcription with audio.
- [ ] [Quality]
  Use [multilingual language model](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models) to assess
  translation quality.
- [x] [Efficiency] Add Azure OpenAI Service support.
- [x] [Quality] Use [claude](https://www.anthropic.com/index/introducing-claude) for translation.
- [x] [Feature] Add local LLM support.
- [X] [Feature] Multiple translate engine (Anthropic, Microsoft, DeepL, Google, etc.) support.
- [ ] [**Feature**] Build
  a [electron + fastapi](https://ivanyu2021.hashnode.dev/electron-django-desktop-app-integrate-javascript-and-python)
  GUI for cross-platform application.
- [x] [Feature] Web-based [streamlit](https://streamlit.io/) GUI.
- [ ] Add [fine-tuned whisper-large-v2](https://huggingface.co/models?search=whisper-large-v2) models for common
  languages.
- [x] [Feature] Add custom OpenAI & Anthropic endpoint support.
- [x] [Feature] Add local translation model support (e.g. [SakuraLLM](https://github.com/SakuraLLM/Sakura-13B-Galgame)).
- [ ] [Quality] Construct translation quality benchmark test for each patch.
- [ ] [Quality] Split subtitles using
  LLM ([ref](https://github.com/Huanshere/VideoLingo/blob/ff520309e958dd3048586837d09ce37d3e9ebabd/core/prompts_storage.py#L6)).
- [ ] [Quality] Trim extra long subtitle using
  LLM ([ref](https://github.com/Huanshere/VideoLingo/blob/ff520309e958dd3048586837d09ce37d3e9ebabd/core/prompts_storage.py#L311)).
- [ ] [Others] Add transcribed examples.
    - [ ] Song
    - [ ] Podcast
    - [ ] Audiobook

## Credits

- https://github.com/guillaumekln/faster-whisper
- https://github.com/m-bain/whisperX
- https://github.com/openai/openai-python
- https://github.com/openai/whisper
- https://github.com/machinewrapped/gpt-subtrans
- https://github.com/MicrosoftTranslator/Text-Translation-API-V3-Python
- https://github.com/streamlit/streamlit

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=zh-plus/Open-Lyrics&type=Date)](https://star-history.dera.page/#zh-plus/Open-Lyrics&Date)

## Citation

```
@book{openlrc2024zh,
	title = {zh-plus/openlrc},
	url = {https://github.com/zh-plus/openlrc},
	author = {Hao, Zheng},
	date = {2024-09-10},
	year = {2024},
	month = {9},
	day = {10},
}
```
