# Installation

1. Please install CUDA and cuDNN first according to https://opennmt.net/CTranslate2/installation.html to
   enable `faster-whisper`.

2. Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.

3. Install [PyTorch](https://pytorch.org/get-started/locally/):
   ```shell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install latest [fast-whisper](https://github.com/guillaumekln/faster-whisper)
   ```shell
   pip install git+https://github.com/guillaumekln/faster-whisper
   ```

5. (Optional) If you want to process videos, install [ffmpeg](https://ffmpeg.org/download.html) and add `bin` directory
   to your `PATH`.

6. This project can be installed from PyPI:

    ```shell
    pip install openlrc
    ```

   or install directly from GitHub:

    ```shell
    pip install git+https://github.com/zh-plus/Open-Lyrics
    ```

# Usage

```python
from openlrc import LRCer

lrcer = LRCer()
```

## Single file

```python
lrcer.run('./data/test.mp3', target_lang='zh-cn')
# Generate translated ./data/test.lrc with default translate prompt.
```

## Multiple files

```python
lrcer.run(['./data/test1.mp3', './data/test2.mp3'], target_lang='zh-cn')
# Note we run the transcription sequentially, but run the translation concurrently for each file.
```

## Video file

```python
lrcer.run(['./data/test_audio.mp3', './data/test_video.mp4'], target_lang='zh-cn')
# Generate translated ./data/test_audio.lrc and ./data/test_video.srt
```

## Context

You can provide some extra context to enhance GPT translation quality. Save them as `context.yaml` in the same directory
as your audio file.

```python
# Use context.yaml to improve translation
lrcer.run('./data/test.mp3', target_lang='zh-cn', context_path='./data/context.yaml')
```

#### context.yaml

```yaml
background: "This is a multi-line background.
This is a basic example."
audio_type: Movie
description_map: {
  movie_name1 (without extension): "This
  is a multi-line description for movie1.",
  movie_name2 (without extension): "This
  is a multi-line description for movie2.",
  movie_name3 (without extension): "This is a single-line description for movie 3.",
}
```

## Change default parameters

```python
asr_options = {"beam_size": 5}
vad_options = {"threshold": 0.2}
preprocess_options = {"atten_lim_db": 20}

lrcer = LRCer(asr_options=asr_options, vad_options=vad_options, preprocess_options=preprocess_options)
lrcer.run('./data/test.mp3', target_lang='zh-cn')
```

## Flow control

#### skip translation

```python
lrcer.run('./data/test.mp3', target_lang='zh-cn', skip_trans=True)
```

#### enable noise suppression (false in default)

*Note this would noticeable consume more time (1/10 audio times)*

```python
lrcer.run('./data/test.mp3', target_lang='zh-cn', noise_suppress=True)
```

