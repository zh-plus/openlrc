#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.
import json

from langcodes import Language
from lingua import LanguageDetectorBuilder

from openlrc.logger import logger

# instruction prompt modified from https://github.com/machinewrapped/gpt-subtrans
base_instruction = r'''You are a translator tasked with revising and translating subtitles into a target language. Your goal is to ensure accurate, concise, and natural-sounding translations for each line of dialogue. The input consists of transcribed audio, which may contain transcription errors. Your task is to first correct any errors you find in the sentences based on their context, and then translate them to the target language according to the revised sentences.
The user will provide a JSON object containing original lines, you should respond with an accurate, concise, and natural-sounding translation for the dialogue. 
The user may provide additional context, such as background, description or title of the source material, a summary of the current scene, or a list of character names. Use this information to improve the quality of your translation.
Your response will be processed by an automated system, so it is imperative that you adhere to the required output format.

Example input (Japanese to Chinese):

{
  "text": [
    {
      "num": 200,
      "original": "変わりゆく時代において、"
      "translation": ""
    },
    {
      "num": 501,
      "original": "生き残る秘訣は、進化し続けることです。"
      "translation": ""
    }
  ]
}

You should respond with JSON:

{
  "text": [
    {
      "num": 200,
      "original": "変わりゆく時代において、",
      "translation": "在变化的时代中，"
    },
    {
      "num": 501,
      "original": "生き残る秘訣は、進化し続けることです。",
      "translation": "生存的秘诀是不断进化。"
    }
  ]
}

Example input (English to German):

{
  "ChunkID": "Scene 1 Chunk 3",
  "context-in": {
    "title": "John and Sarah",
    "background": "The story after John and Sarah go back to downtown.",
    "description": "John and Sarah meet Tom in the office. They discuss their plan to locate a suspect.",
    "previous_summaries": "Chunk 1: John and Sarah is on the way to office, discussing the lunch plan. Chunk 2: John and Sarah meet Tom in the office.",
    "scene": "John and Sarah wake up and get ready for work."
  },
  "text": [
    {
      "num": 700,
      "original": "those who resist change may find themselves left behind."
      "translation": ""
    },
    {
      "num": 701,
      "original": "those resist change find themselves left."
      "translation": ""
    }
  ]
}

You should respond with JSON:

{
  "ChunkID": "Scene 1 Chunk 3",
  "text": [
    {
      "num": 700,
      "original": "those who resist change may find themselves left behind.",
      "translation": "Im Zeitalter der digitalen Transformation,"
    },
    {
      "num": 701,
      "original": "those resist change find themselves left.",
      "translation": "diejenigen, die sich dem Wandel widersetzen, könnten sich zurückgelassen finden."
    }
  ],
  "context-out": {
    "summary": "John and Sarah discuss their plan to locate a suspect, deducing that he is likely in the uptown area.",
    "scene": "John and Sarah are in their office analyzing data and planning their next steps. They deduce that the suspect is probably in the uptown area and decide to start their search there."
  }
}

Please ensure that each line of dialogue remains distinct in the translation. Merging lines together can lead to timing problems during playback.

At the end of each set of translations, include a one or two line synopsis of the input text in ```context-out.summary``` using target language.

Use the available information to add a short description of the current scene in ```context-out.scene``` using target language.

Use the target language when writing content for the summary and scene context. 
Ensure that the summary and scene are concise, containing less than 100 words.
You need to update your summary and scene with the new information you have.
Do not guess or improvise if the context is unclear, just summarise the dialogue.

The translation should be in a lovely colloquial style and suitable for high-quality subtitles.

#######################
There was an issue with the previous translation. 

Remember to include ``summary`` and ``scene`` tags in your response.
Do not translate ``original`` and ``translation`` key value.
Please translate the subtitles again, paying careful attention to ensure that each line is translated separately, and that every line has a matching translation.
Do not merge lines together in the translation, it leads to incorrect timings and confusion for the reader.
Do not directly translate the original text, make full use of the context to make the translation sound more natural.'''


class TranslatePrompter:
    @classmethod
    def format_texts(cls, texts):
        raise NotImplementedError()

    @staticmethod
    def post_process(texts):
        raise NotImplementedError()

    def check_format(self, messages, output_str):
        raise NotImplementedError()


class BaseTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang, audio_type=None, title='', background='', description=''):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()

        self.audio_type = audio_type
        self.title = title
        self.background = background
        self.description = description
        self.user_prompt = f'''Please translate these subtitles from {self.src_lang_display} to {self.target_lang_display}:\n'''

    @staticmethod
    def system():
        return base_instruction

    def user(self, chunk_num, chunk, summaries='', scene=''):
        summaries_str = '\n'.join(f'Chunk {i}: {summary}' for i, summary in enumerate(summaries, 1))

        input_json = {
            "ChunkID": f"Scene 1 Chunk {chunk_num}",
            "context-in": {
                "title": self.title,
                "background": self.background,
                "description": self.description,
                "previous_summaries": summaries_str,
                "scene": scene
            },
            "text": [
                {"num": num, "original": text, "translation": ""} for num, text in chunk
            ]
        }

        # Remove empty context fields
        input_json['context-in'] = {k: v for k, v in input_json['context-in'].items() if v}

        return self.user_prompt + json.dumps(input_json)

    def check_format(self, input_messages, output_content):
        try:
            output_data = json.loads(output_content)
        except json.JSONDecodeError:
            logger.warning(f'Fail to decode json.')
            return False

        # Check json fields
        if 'text' not in output_data:
            logger.warning(f'Fail to find text in json.')
            return False
        if any(['translation' not in j for j in output_data['text']]):
            logger.warning(f'Fail to find translation in text.')
            return False
        if 'context-out' not in output_data:
            logger.warning(f'Fail to find context-out in json.')
            return False
        if 'summary' not in output_data['context-out']:
            logger.warning(f'Fail to find summary in context-out.')
        if 'scene' not in output_data['context-out']:
            logger.warning(f'Fail to find scene in context-out')

        # Check input and output text number
        input_data = json.loads(input_messages[1]['content'][len(self.user_prompt):])
        originals = input_data['text']
        translations = output_data['text']
        if len(originals) != len(translations):
            logger.warning(
                f'Fail to ensure length consistent: original is {len(originals)}, translation is {len(translations)}')

        # Ensure the translated langauge is in the target language
        translations = [t['translation'] for t in translations]
        if len(translations) >= 3:
            # 3-voting for detection stability
            chunk_size = len(translations) // 3
            translation_chunks = [translations[i:i + chunk_size] for i in range(0, len(translations), chunk_size)]
            if len(translation_chunks) > 3:
                translation_chunks[-2].extend(translation_chunks[-1])
                translation_chunks.pop()

            translated_langs = [self.lan_detector.detect_language_of(' '.join(chunk)).name.lower()
                                for chunk in translation_chunks]

            # get the most common language
            translated_lang = max(set(translated_langs), key=translated_langs.count)
        else:
            translated_lang = self.lan_detector.detect_language_of(' '.join(translations)).name.lower()

        target_lang = Language.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated language is {translated_lang}, not {target_lang}.')
            return False

        return True


prompter_map = {
    'base_trans': BaseTranslatePrompter,
}
