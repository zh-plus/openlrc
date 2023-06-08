# Open-Lyrics

Open-Lyrics is a open-source project to transcribe voice file and use large language model to translate/polish the text.

## Installation

```shell
pip install openlyc
```

## Usage

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer('./data/test.mp3')
```