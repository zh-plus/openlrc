# Open-Lyrics

Open-Lyrics is a open-source project to transcribe (
using [faster-whisper](https://github.com/guillaumekln/faster-whisper)) voice file and
translate/polish ([OpenAI-GPT](https://github.com/openai/openai-python)) the text.

## Installation

```shell
pip install openlyc
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

## Credits

- https://github.com/guillaumekln/faster-whisper
- https://github.com/openai/openai-python
- https://github.com/openai/whisper