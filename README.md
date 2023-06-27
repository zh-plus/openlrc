# Open-Lyrics

Open-Lyrics is a Python library that transcribes voice files using
[faster-whisper](https://github.com/guillaumekln/faster-whisper), and translates/polishes the resulting text
into `.lrc` files in the desired language using [OpenAI-GPT](https://github.com/openai/openai-python).

## Installation

1. Please install CUDA and cuDNN first according to https://opennmt.net/CTranslate2/installation.html to
   enable `faster-whisper`.

2. Add your [OpenAI API key](https://platform.openai.com/account/api-keys) to environment variable `OPENAI_API_KEY`.

3. Install [PyTorch](https://pytorch.org/get-started/locally/):
    ```shell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install [whisperx](https://github.com/m-bain/whisperX)

    ```shell
    pip install git+https://github.com/m-bain/whisperx.git
    ```

5. Install [ffmpeg](https://ffmpeg.org/download.html) and add `bin` directory to your `PATH`.

6. This project can be installed from PyPI:

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

# Single file
lrcer.run('./data/test.mp3', target_lang='zh-cn')  # Generate translated ./data/test.lrc with default translate prompt.

# Multiple files
lrcer.run(['./data/test1.mp3', './data/test2.mp3'], target_lang='zh-cn')
# Note we run the transcription sequentially, but run the translation concurrently for each file.

# Path can contain video
lrcer.run(['./data/test_audio.mp3', './data/test_video.mp4'], target_lang='zh-cn')

# Use context.yaml to improve translation
lrcer.run('./data/test.mp3', target_lang='zh-cn', context_path='./data/context.yaml')
```

### Context

Utilize the available context to enhance the quality of your translation.
Save them as `context.yaml` in the same directory as your audio file.

```yaml
background: "This is a multi-line background.
This is a basic example."
audio_type: Movie
synopsis_map: {
  movie_name1 (without extension): "This
  is a multi-line synopsis for movie1.",
  movie_name2 (without extension): "This
  is a multi-line synopsis for movie2.",
  movie_name3 (without extension): "This is a single-line synopsis for movie 3.",
}
```

## Todo

- [x] [Efficiency] Batched translate/polish for GPT request (enable contextual ability).
- [x] [Efficiency] Concurrent support for GPT request.
- [x] [Efficiency & Transcription Quality] Use [whisperx](https://github.com/m-bain/whisperX) for transcription.
- [x] [Translation Quality] Make translate prompt more robust according to https://github.com/openai/openai-cookbook.
- [x] [Usability] Automatically fix json encoder error using GPT.
- [x] [Efficiency] Asynchronously perform transcription and translation for multiple audio inputs.
- [x] [Quality] Improve batched translation/polish prompt according
  to [gpt-subtrans](https://github.com/machinewrapped/gpt-subtrans).
- [x] [Usability] Input video support.
- [ ] [Usability] Multiple output format support.
- [ ] [Quality]
  Use [multilingual language model](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models) to assess
  translation quality.
- [ ] [Quality] Speech enhancement for input audio.
- [ ] [Efficiency] Add Azure OpenAI Service support.
- [ ] [Usability] Add local LLM support.
- [ ] [Usability] Multiple translate engine (Microsoft, DeepL, Google, etc.) support.
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