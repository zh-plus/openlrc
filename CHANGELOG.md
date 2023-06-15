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