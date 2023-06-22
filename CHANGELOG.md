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