# Open-Lyrics

[![PyPI](https://img.shields.io/pypi/v/openlrc)](https://pypi.org/project/openlrc/)
[![PyPI - License](https://img.shields.io/pypi/l/openlrc)](https://pypi.org/project/openlrc/)
[![Downloads](https://static.pepy.tech/badge/openlrc)](https://pepy.tech/project/openlrc)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/zh-plus/Open-Lyrics/ci.yml)

Open-Lyrics is a Python library that transcribes voice files using
[faster-whisper](https://github.com/guillaumekln/faster-whisper), and translates/polishes the resulting text
into `.lrc` files in the desired language using LLM,
e.g. [OpenAI-GPT](https://github.com/openai/openai-python), [Anthropic-Claude](https://github.com/anthropics/anthropic-sdk-python).

#### Key Features:

- Well preprocessed audio to reduce hallucination (Loudness Norm & optional Noise Suppression).
- Context-aware translation to improve translation quality.
  Check [prompt](https://github.com/zh-plus/openlrc/blob/master/openlrc/prompter.py) for details.
- Check [here](#how-it-works) for an overview of the architecture.

## New 🚨

- 2024.5.7:
    - Add custom endpoint (base_url) support for OpenAI & Anthropic:
        ```python
        lrcer = LRCer(base_url_config={'openai': 'https://api.chatanywhere.tech',
                                       'anthropic': 'https://example/api'})
        ```
    - Generating bilingual subtitles
        ```python
        lrcer.run('./data/test.mp3', target_lang='zh-cn', bilingual_sub=True)
        ``` 
- 2024.5.11: Add glossary into prompt, which is confirmed to improve domain specific translation.
  Check [here](#glossary) for details.
- 2024.5.17: You can route model to arbitrary Chatbot SDK (either OpenAI or Anthropic) by setting `chatbot_model` to
  `provider: model_name` together with base_url_config:
    ```python
    lrcer = LRCer(chatbot_model='openai: claude-3-haiku-20240307',
                  base_url_config={'openai': 'https://api.g4f.icu/v1/'})
    ```

## Installation ⚙️

1. Please install CUDA 11.x and [cuDNN 8 for CUDA 11](https://developer.nvidia.com/cudnn) first according
   to https://opennmt.net/CTranslate2/installation.html to enable `faster-whisper`.

   `faster-whisper` also needs [cuBLAS for CUDA 11](https://developer.nvidia.com/cublas) installed.
   <details>
   <summary>For Windows Users (click to expand)</summary> 

   (For Windows Users only) Windows user can Download the libraries from Purfview's repository:

   Purfview's [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) provides the required NVIDIA
   libraries for Windows in a [single archive](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs).
   Decompress the archive and place the libraries in a directory included in the `PATH`.

   </details>


2. Add LLM API keys, you can either:
    - Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.
    - Add your [Anthropic API key](https://console.anthropic.com/settings/keys) to environment
      variable `ANTHROPIC_API_KEY`.

3. Install [PyTorch](https://pytorch.org/get-started/locally/):
   ```shell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install latest [fast-whisper](https://github.com/guillaumekln/faster-whisper)
   ```shell
   pip install git+https://github.com/guillaumekln/faster-whisper
   ```

5. Install [ffmpeg](https://ffmpeg.org/download.html) and add `bin` directory
   to your `PATH`.

6. This project can be installed from PyPI:

    ```shell
    pip install openlrc
    ```

   or install directly from GitHub:

    ```shell
    pip install git+https://github.com/zh-plus/openlrc
    ```

## Usage 🐍

### GUI

> [!NOTE]
> We are migrating the GUI from streamlit to Gradio. The GUI is still under development.

```shell
openlrc gui
```

![](https://github.com/zh-plus/openlrc/blob/master/resources/streamlit_app.jpg?raw=true)

### Python code

```python
from openlrc import LRCer

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
    lrcer = LRCer(glossary='./data/aoe4-glossary.yaml')

    # To skip translation process
    lrcer.run('./data/test.mp3', target_lang='en', skip_trans=True)

    # Change asr_options or vad_options, check openlrc.defaults for details
    vad_options = {"threshold": 0.1}
    lrcer = LRCer(vad_options=vad_options)
    lrcer.run('./data/test.mp3', target_lang='zh-cn')

    # Enhance the audio using noise suppression (consume more time).
    lrcer.run('./data/test.mp3', target_lang='zh-cn', noise_suppress=True)

    # Change the LLM model for translation
    lrcer = LRCer(chatbot_model='claude-3-sonnet-20240229')
    lrcer.run('./data/test.mp3', target_lang='zh-cn')

    # Clear temp folder after processing done
    lrcer.run('./data/test.mp3', target_lang='zh-cn', clear_temp_folder=True)

    # Change base_url
    lrcer = LRCer(base_url_config={'openai': 'https://api.g4f.icu/v1',
                                   'anthropic': 'https://example/api'})

    # Route model to arbitrary Chatbot SDK
    lrcer = LRCer(chatbot_model='openai: claude-3-sonnet-20240229',
                  base_url_config={'openai': 'https://api.g4f.icu/v1/'})

    # Bilingual subtitle
    lrcer.run('./data/test.mp3', target_lang='zh-cn', bilingual_sub=True)
```

Check more details in [Documentation](https://zh-plus.github.io/openlrc/#/).

### Glossary

Add glossary to improve domain specific translation. For example `aoe4-glossary.yaml`:

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
lrcer = LRCer(glossary='./data/aoe4-glossary.yaml')
lrcer.run('./data/test.mp3', target_lang='zh-cn')
```

or directly use dictionary to add glossary:

```python
lrcer = LRCer(glossary={"aoe4": "帝国时代4", "feudal": "封建时代"})
lrcer.run('./data/test.mp3', target_lang='zh-cn')
```

## Pricing 💰

*pricing data from [OpenAI](https://openai.com/pricing)
and [Anthropic](https://docs.anthropic.com/claude/docs/models-overview#model-comparison)*

| Model Name                 | Pricing for 1M Tokens <br/>(Input/Output) (USD) | Cost for 1 Hour Audio <br/>(USD) |
|----------------------------|-------------------------------------------------|----------------------------------|
| `gpt-3.5-turbo-0125`       | 0.5, 1.5                                        | 0.01                             |
| `gpt-3.5-turbo`            | 0.5, 1.5                                        | 0.01                             |
| `gpt-4-0125-preview`       | 10, 30                                          | 0.5                              |
| `gpt-4-turbo-preview`      | 10, 30                                          | 0.5                              |
| `gpt-4o`                   | 5, 15                                           | 0.25                             |
| `claude-3-haiku-20240307`  | 0.25, 1.25                                      | 0.015                            |
| `claude-3-sonnet-20240229` | 3, 15                                           | 0.2                              |
| `claude-3-opus-20240229`   | 15, 75                                          | 1                                |

**Note the cost is estimated based on the token count of the input and output text.
The actual cost may vary due to the language and audio speed.**

### Recommended translation model

For english audio, we recommend using `gpt-3.5-turbo`.

For non-english audio, we recommend using `claude-3-sonnet-20240229`.

## How it works

![](https://github.com/zh-plus/openlrc/blob/master/resources/how-it-works.png?raw=true)

To maintain context between translation segments, the process is sequential for each audio file.

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
- [ ] [Efficiency] Add Azure OpenAI Service support.
- [ ] [Quality] Use [claude](https://www.anthropic.com/index/introducing-claude) for translation.
- [ ] [Feature] Add local LLM support.
- [X] [Feature] Multiple translate engine (Anthropic, Microsoft, DeepL, Google, etc.) support.
- [ ] [**Feature**] Build
  a [electron + fastapi](https://ivanyu2021.hashnode.dev/electron-django-desktop-app-integrate-javascript-and-python)
  GUI for cross-platform application.
- [x] [Feature] Web-based [streamlit](https://streamlit.io/) GUI.
- [ ] Add [fine-tuned whisper-large-v2](https://huggingface.co/models?search=whisper-large-v2) models for common
  languages.
- [x] [Feature] Add custom OpenAI & Anthropic endpoint support.
- [ ] [Feature] Add local translation model support (e.g. [SakuraLLM](https://github.com/SakuraLLM/Sakura-13B-Galgame)).
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

[![Star History Chart](https://api.star-history.com/svg?repos=zh-plus/Open-Lyrics&type=Date)](https://star-history.com/#zh-plus/Open-Lyrics&Date)
