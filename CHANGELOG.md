## 1.3.1

### New Features:

- Add custom endpoint (base_url) support for OpenAI & Anthropic.
- Generating bilingual subtitles.

### Other Changes:

- Fix dep issues from ctranslate2 and streamlit-related packages.

## 1.3.0

Add basic GUI support via streamlit.

#### Other Changes:

- Add clear_temp_folder args.

## 1.2.0

This update add Claude Models support for translation.

## 1.1.0

This update improve the translation quality.

#### Other Changes:

- Update faster-whisper version to 1.0.0.
- Set hallucination_silence_threshold to 2, which alleviates the hallucination issue.
- Add proxy argument.

## 1.0.3

This update fix minor issues.

#### Other Changes:

- Remove water mark in srt and lrc.
- Improve logging.

## 1.0.2

This update add minor features.

#### Other Changes:

- Binlingual subtitle support (Beta).
- Improve translation prompt.

## 1.0.1

This update fix minor issues.

#### Other Changes:

- Fix issue that prevent the usage of whisper-large-v3.
- Remove tags from translation texts.

## 1.0.0

This update introduce new features.

#### New features:

- Resume from previous translation.
- Add atomic translation for src-trans inconsistency.

#### Other Changes:

- Update default whisper model to `whisper-large-v3`.

## 0.2.3

This update improves preprocess efficiency and minor changes.

#### Other Changes:

- Introduce multiprocessing for loudness normalization.
- Fix the .srt generation issue for video input.
- Add preprocess options, which users can tune.

## 0.2.2

This update addresses minor issues.

#### Other Changes:

- Split audio during noise suppression to avoid out-of-memory.
- Improve translation prompt.

## 0.2.1

This update adds a preprocessor to enhance input audio (loudness normalization & noise suppression).

#### New features:

- Loudness Normalization from [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize)
- Noise Suppression from [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

#### Other Changes:

- Now all the intermediate files are saved in `./path/to/audio/preprocess`.

## 0.2.0

This update switch the underlying transcription model from `whisperx` to `faster-whisper`, which enable VAD parameter
tuning.

#### New features:

- Switch whisperx back to faster-whisper for VAD parameter tuning.

#### Other Changes:

- Update translation prompt
  from https://github.com/machinewrapped/gpt-subtrans/commit/82bd2ca0d868f209d0e0c5f7c04255523daabe3c.
- Change the default parameters of `faster-whisper` for consistent transcription.

## 0.1.5

Emergent bugfix release.

## 0.1.4

This update add input video support and introduce context configuration.

#### New features:

- Add `word_align` and `sentence_split` for non-word-boundary languages to split long text into sentences.
- Add text-normalization for help matching sentences.
- Add skip-translation support.

#### Other Changes:

- Use `pathlib` to handle paths.
- Improve timeline accuracy.

## 0.1.3

This update add input video support and introduce context configuration.

#### New features:

- Add input video support.
- Add context configuration for inputs.

#### Other Changes:

- Add test suites to CI.
- Add language detection for translated content.
- Improve prompt by adding background info.
- Update punctuator model.
- Replace `opencc` with more light-weight `zhconv`.

## 0.1.2

This update improves the timeline consistency of translated subtitles.
Thanks [gpt-subtrans](https://github.com/machinewrapped/gpt-subtrans)!

#### New features:

- Fix misaligned timeline issue by improving translation prompt.
- Add output srt format support.
- Add changeable temperature and top_p parameter for GPTBot.

#### Other Changes:

- Report total OpenAI translation fee for multiple audios.
- Improve repeat-checking algorithm.

## 0.1.1

This update enhances the efficiency of processing multiple audio files.

#### New features:

- Implementation of a producer-consumer model to process multiple audio files.

#### Other Changes:

- Update logger with colored format.
- Minor parameter modification that makes the timeline of translation more intuitive.

## 0.1.0

This update significantly improves translation quality, but at the cost of slower translation speed.

#### New Features:

- Use multi-step prompt for translation.
- Update the default model to `gpt-3.5-turbo-16k`.
- Automatically fix json encoder error using GPT.

#### Other Changes:

- Calculate the accurate price for OpenAI API requests.

## 0.0.6

This update greatly improves the quality of transcription (both in time-alignment and text-quality).

#### New Features:

- Use `whisperx` to improve transcription accuracy.
- Add Traditional Chinese to Mandarin optimization when `target_lang=zh-cn`.

## 0.0.5

#### New Features:

#### Other Changes:

- Update build tool to poetry.

## 0.0.4

#### New Features:

- Use async call to communicate with OpenAI api.
- Abstract the GPT communication module as `GPTBot`.
- Add fee limit for GPTBot.