# Open-Lyrics

Open-Lyrics is a Python library that transcribes voice files using
[faster-whisper](https://github.com/guillaumekln/faster-whisper), and translates/polishes the resulting text
into `.lrc` files in the desired language using [OpenAI-GPT](https://github.com/openai/openai-python).

## Installation

1. Please install CUDA and cuDNN first according to https://opennmt.net/CTranslate2/installation.html to
   enable `faster-whisper`.

2. Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.

3. Install [whisperx](https://github.com/m-bain/whisperX)

    ```shell
    pip install git+https://github.com/m-bain/whisperx.git
    ```

4. This project can be installed from PyPI:

    ```shell
    pip install openlrc
    ```

   or install directly from GitHub:

    ```shell
    pip install git+https://github.com/zh-plus/Open-Lyrics
    ```

## Usage

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.run('./data/test.mp3', target_lang='zh-cn')  # Generate translated ./data/test.lrc with default translate prompt.
```

## Known issue

- Sometimes there may be misaligned timeline, trying to fix it.

## Todo

- [x] [Efficiency] Batched translate/polish for GPT request (enable contextual ability).
- [x] [Efficiency] Concurrent support for GPT request.
- [x] [Efficiency & Transcription Quality] Use [whisperx](https://github.com/m-bain/whisperX) for transcription.
- [x] [Translation Quality] Make translate prompt more robust according to https://github.com/openai/openai-cookbook.
- [x] [Usability] Automatically fix json encoder error using GPT.
- [x] [Efficiency] Asynchronously perform transcription and translation for multiple audio inputs.
- [ ] [Quality] Improve batched translation/polish prompt according
  to [gpt-subtrans](https://github.com/machinewrapped/gpt-subtrans).
- [ ] [Usability] Multiple output format support.
- [ ] [Efficiency] Add Azure OpenAI Service support.
- [ ] [Usability] Add local LLM support.
- [ ] [Others] Add transcribed examples.
    - [ ] Song
    - [ ] Podcast
    - [ ] Audiobook

## Credits

- https://github.com/guillaumekln/faster-whisper
- https://github.com/openai/openai-python
- https://github.com/openai/whisper