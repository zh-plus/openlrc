# Open-Lyrics

Open-Lyrics is a open-source project to transcribe (
using [faster-whisper](https://github.com/guillaumekln/faster-whisper)) voice file and
translate/polish ([OpenAI-GPT](https://github.com/openai/openai-python)) the text.

**This new project is rapidly underway, and we welcome any issues or pull requests.**

## Installation

Please install CUDA and cuDNN first according to https://opennmt.net/CTranslate2/installation.html to
enable `faster-whisper`.

Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.

shell
This project can be installed from PyPI:

```shell
pip install openlrc
```

## Usage

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer('./data/test.mp3')  # Generate ./data/test.lrc
```

## Todo

- [ ] Add transcribed examples.
    - [ ] Song
    - [ ] Podcast
    - [ ] Audiobook
- [ ] Make translate prompt more robust.
- [ ] Add local LLM support.
- [ ] Multi-thead support for both whisper model and GPT request.
- [ ] Automatically fix json encoder error using GPT.

## Credits

- https://github.com/guillaumekln/faster-whisper
- https://github.com/openai/openai-python
- https://github.com/openai/whisper