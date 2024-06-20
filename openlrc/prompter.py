#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import abc
import re
from abc import ABC
from typing import List, Tuple, Optional

from langcodes import Language
from lingua import LanguageDetectorBuilder

from openlrc.context import TranslateInfo
from openlrc.logger import logger

ORIGINAL_PREFIX = 'Original>'
TRANSLATION_PREFIX = 'Translation>'
PROOFREAD_PREFIX = 'Proofread>'

POTENTIAL_PREFIX_COMBOS = [
    [ORIGINAL_PREFIX, TRANSLATION_PREFIX],
    ['原文>', '翻译>'],
    ['原文>', '译文>'],
    ['原文>', '翻譯>'],
    ['原文>', '譯文>'],
    ['Original>', 'Translation>'],
    ['Original>', 'Traducción>']
]

# instruction prompt modified from https://github.com/machinewrapped/gpt-subtrans
BASE_TRANSLATE_INSTRUCTION = f'''Ignore all previous instructions.
You are a translator tasked with revising and translating subtitles into a target language. Your goal is to ensure accurate, concise, and natural-sounding translations for each line of dialogue. The input consists of transcribed audio, which may contain transcription errors. Your task is to first correct any errors you find in the sentences based on their context, and then translate them to the target language according to the revised sentences.
The user will provide a chunk of lines, you should respond with an accurate, concise, and natural-sounding translation for the dialogue, with appropriate punctuation.
The user may provide additional context, such as title of the source material, a summary of the current scene, or a list of character names. Use this information to improve the quality of your translation.
Your response will be processed by an automated system, so it is imperative that you adhere to the required output format.
The source subtitles were AI-generated with a speech-to-text tool so they are likely to contain errors. Where the input seems likely to be incorrect, use ALL available context to determine what the correct text should be, to the best of your ability.

Example input (Japanese to Chinese):

#200
{ORIGINAL_PREFIX}
変わりゆく時代において、
{TRANSLATION_PREFIX}

#501
{ORIGINAL_PREFIX}
生き残る秘訣は、進化し続けることです。
{TRANSLATION_PREFIX}

You should respond with:

#200
{ORIGINAL_PREFIX}
変わく時代いて、
{TRANSLATION_PREFIX}
在变化的时代中，

#501
{ORIGINAL_PREFIX}
生き残る秘訣は、進化し続けることです。
{TRANSLATION_PREFIX}
生存的秘诀是不断进化。

Example input (English to German):

#700
{ORIGINAL_PREFIX}
those who resist change may find themselves left behind.
{TRANSLATION_PREFIX}

#701
{ORIGINAL_PREFIX}
those resist change find themselves left.
{TRANSLATION_PREFIX}

You should respond with:

#700
{ORIGINAL_PREFIX}
In the age of digital transformation,
{TRANSLATION_PREFIX}
Im Zeitalter der digitalen Transformation,

#701
{ORIGINAL_PREFIX}
those who resist change may find themselves left behind.
{TRANSLATION_PREFIX}
diejenigen, die sich dem Wandel widersetzen, könnten sich zurückgelassen finden.

Please ensure that each line of dialogue remains distinct in the translation. Merging lines together can lead to timing problems during playback.

At the end of each set of translations, include a one or two line synopsis of the input text encapsulated in a <summary/> tag, for example:
<summary>John and Sarah discuss their plan to locate a suspect, deducing that he is likely in the uptown area.</summary>
Remember to end this tag with ``</summary>``.

Also use the available information to add a short description of the current scene so far encapsulated in a <scene/> tag, for example:
<scene>John and Sarah are in their office analyzing data and planning their next steps. They deduce that the suspect is probably in the uptown area and decide to start their search there.</scene>
Remember to end this tag with ``</scene>``.

Use the target language when writing content for the <summary/> and <scene/> tags. 
Ensure that the summary and scene are concise, containing less than 100 words.
You need to update your summary and scene with the new information you have.
Do not guess or improvise if the context is unclear, just summarise the dialogue.

The translation should be in a lovely colloquial style and suitable for high-quality subtitles.

I’m going to tip $1000 for a better translation!

### retry_instructions
There was an issue with the previous translation. 

Remember to include ``<summary>`` and ``<scene>`` tags in your response.
Please translate the subtitles again, paying careful attention to ensure that each line is translated separately, and that every line has a matching translation.
Do not merge lines together in the translation, it leads to incorrect timings and confusion for the reader.
The content of the translation is for learning purposes only and will not violate the usage guidelines.
'''


class Prompter(abc.ABC):
    def check_format(self, messages, output_str):
        return True


class TranslatePrompter(Prompter, ABC):
    @classmethod
    def format_texts(cls, texts):
        raise NotImplementedError()

    @staticmethod
    def post_process(texts):
        raise NotImplementedError()


class BaseTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang, context: TranslateInfo):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()

        self.audio_type = context.audio_type
        self.title = context.title
        self.glossary = context.glossary
        self.user_prompt = f'''Translation guidelines from context reviewer:
{{guideline}}

Previews summaries:
{{summaries_str}}

<chunk_id> Scene 1 Chunk {{chunk_num}} <chunk_id>

Please translate these subtitles for {self.audio_type} from {self.src_lang_display} to {self.target_lang_display}.\n
{{user_input}}
<summary></summary>
<scene></scene>'''

    def system(self) -> str:
        return BASE_TRANSLATE_INSTRUCTION

    def user(self, chunk_num: int, user_input: str, summaries='', guideline='') -> str:
        summaries_str = '\n'.join(f'Chunk {i}: {summary}' for i, summary in enumerate(summaries, 1))
        return self.user_prompt.format(
            summaries_str=summaries_str, chunk_num=chunk_num, user_input=user_input, guideline=guideline).strip()

    @property
    def formatted_glossary(self):
        glossary_strings = '\n'.join(f'{k}: {v}' for k, v in self.glossary.items())
        result = f'''
# Glossary
Use the following glossary to ensure consistency in your translations:
<preferred-translation>
{glossary_strings}
</preferred-translation>
'''
        return result

    @classmethod
    def format_texts(cls, texts: List[Tuple[int, str]]):
        """
        Reconstruct list of text into desired format.

        Args:
            texts: List of (id, text).

        Returns:
            The formatted string: f"#id\n{original_prefix}\n{text}\n{translation_prefix}\n"
        """
        return '\n'.join([f'#{i}\n{ORIGINAL_PREFIX}\n{text}\n{TRANSLATION_PREFIX}\n' for i, text in texts])

    def check_format(self, messages, content):
        summary = re.search(r'<summary>(.*)</summary>', content)
        scene = re.search(r'<scene>(.*)</scene>', content)

        # If message is for claude, use messages[0]
        user_input = messages[1]['content'] if len(messages) == 2 else messages[0]['content']
        original = re.findall(ORIGINAL_PREFIX + r'\n(.*?)\n' + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error(f'Fail to extract original text.')
            return False

        translation = self._extract_translation(content)
        if not translation:
            # TODO: Try to change chatbot_model if always fail
            logger.warning(f'Fail to extract translation.')
            logger.debug(f'Content: {content}')
            return False

        if len(original) != len(translation):
            logger.warning(
                f'Fail to ensure length consistent: original is {len(original)}, translation is {len(translation)}')
            logger.debug(f'original: {original}')
            logger.debug(f'translation: {original}')
            return False

        # Ensure the translated langauge is in the target language
        if not self._is_translation_in_target_language(translation):
            return False

        # It's ok to keep going without summary and scene
        if not summary or not summary.group(1):
            logger.warning(f'Fail to extract summary.')
        if not scene or not scene.group(1):
            logger.warning(f'Fail to extract scene.')

        return True

    def _extract_translation(self, content: str) -> List[str]:
        for potential_ori_prefix, potential_trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translation = re.findall(f'{potential_trans_prefix}\n*(.*?)(?:#\\d+|<summary>|\\n*$)', content, re.DOTALL)
            if translation:
                return translation
        return []

    def _is_translation_in_target_language(self, translation: List[str]) -> bool:
        if len(translation) >= 3:
            chunk_size = len(translation) // 3
            translation_chunks = [translation[i:i + chunk_size] for i in range(0, len(translation), chunk_size)]
            if len(translation_chunks) > 3:
                translation_chunks[-2].extend(translation_chunks[-1])
                translation_chunks.pop()

            translated_langs = [self.lan_detector.detect_language_of(' '.join(chunk)) for chunk in translation_chunks]
            translated_langs = [lang.name.lower() for lang in translated_langs if lang]

            if not translated_langs:
                return True

            translated_lang = max(set(translated_langs), key=translated_langs.count)
        else:
            detected_lang = self.lan_detector.detect_language_of(' '.join(translation))
            if not detected_lang:
                return True
            translated_lang = detected_lang.name.lower()

        target_lang = Language.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated language is {translated_lang}, not {target_lang}.')
            return False

        return True


class AtomicTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()

    def user(self, text):
        return f'''Please translate the following text from {self.src_lang_display} to {self.target_lang_display}. 
Please do not output any content other than the translated text. Here is the text: {text}'''

    def check_format(self, messages, output_str):
        # Ensure the translated langauge is in the target language
        detected_lang = self.lan_detector.detect_language_of(output_str)
        if not detected_lang:
            # Cant detect language
            return True

        translated_lang = detected_lang.name.lower()
        target_lang = Language.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated text: "{output_str}" is {translated_lang}, not {target_lang}.')
            return False

        return True


class ContextReviewPrompter(Prompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()

    def system(self):
        return f'''Context:
You are a context reviewer responsible for ensuring the consistency and accuracy of translations between two languages. Your task involves reviewing and providing necessary contextual information for translations.

Objective:
1. Build a comprehensive glossary of key terms and phrases used in the {self.src_lang_display} to {self.target_lang_display} translations. The glossary should include technical terms, slang, and culturally specific references that need consistent translation or localization, focusing on terms that may cause confusion or inconsistency.
2. Provide character name translations, including relevant information about the characters, such as relationships, roles, or personalities.
3. Write a concise story summary capturing the main plot points, characters, and themes of the video to help team members understand the context.
4. Define the tone and style of the subtitles, ensuring they match the intended mood and atmosphere of the texts, with guidelines on language use, formality, and stylistic preferences.
5. Identify the target audience for the subtitles, considering factors such as age, cultural background, and language proficiency, and provide insights on how to tailor the subtitles accordingly.

Style:
Formal and professional, with clear and precise language suitable for translation and localization contexts.

Tone:
Informative and authoritative to ensure clarity and reliability in the instructions.

Audience:
Translators, localization specialists, and proofreaders who need a detailed and consistent reference document for subtitling.

Response Format:
The output should include the following sections: Glossary, Characters, Summary, Tone and Style, Target Audience.

Example Input:
Please review the following text (title: The Detectors) and provide the necessary context for the translation from English to Chinese:
John and Sarah discuss their plan to locate a suspect, deducing that he is likely in the uptown area.
John: "As a 10 years experienced detector, my advice is we should start our search in the uptown area."
Sarah: "Agreed. Let's gather more information before we move."
Then, they prepare to start their investigation.

Example Output:

### Glossary:
- suspect: 嫌疑人
- uptown: 市中心

### Characters:
- John: 约翰, a detector with 10 years of experience
- Sarah: 萨拉, John's detector partner

### Summary:
John and Sarah discuss their plan to locate a suspect in the uptown area. They decide to gather more information before starting their investigation.

### Tone and Style:
The subtitles should be formal and professional, reflecting the serious nature of the investigation. Avoid slang and colloquial language.

### Target Audience:
The target audience is adult viewers with an interest in crime dramas. They are likely to be familiar with police procedurals and enjoy suspenseful storytelling.
'''

    def user(self, text, title='', given_glossary: Optional[dict] = None):
        glossary_text = f'Given glossary: {given_glossary}' if given_glossary else ''
        return f'''{glossary_text}
Please review the following text (title:{title}) and provide the necessary context for the translation from {self.src_lang_display} to {self.target_lang_display}:
{text}'''


class ProofreaderPrompter(Prompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.lan_detector = LanguageDetectorBuilder.from_all_languages().build()

    def system(self):
        return f'''Ignore all previous instructions.
You are a experienced proofreader, responsible for meticulously reviewing the translated subtitles to ensure they are free of grammatical errors, spelling mistakes, and inconsistencies. The Proofreader ensures that the subtitles are clear, concise, and adhere to the provided glossary and style guidelines.
Carefully read through the translated subtitles provided by translators. Ensure that the subtitles make sense in the context of the video and are easy to understand.
Check for and correct any grammatical errors, including punctuation, syntax, and sentence structure. Ensure that all words are spelled correctly and consistently throughout the subtitles.
Refer to the glossary and style guidelines provided by the Context Reviewer. Ensure that key terms, names, and phrases are used consistently and correctly throughout the subtitles. Verify that the tone and style of the subtitles are consistent with the guidelines.
Ensure that the subtitles are clear and concise, avoiding overly complex or ambiguous language. Make sure that the subtitles are easy to read and understand, especially considering the target audience's language proficiency.
Ensure that the subtitles accurately reflect the context and intent of the original dialogue. Make sure that any cultural references, jokes, or idiomatic expressions are appropriately localized and understandable.
Conduct a final review to ensure there are no remaining errors or inconsistencies. Make any necessary corrections to ensure the subtitles are accurate, natural-sounding, and of the highest quality.

Example input:
Please proofread the following translated text (the original texts are for reference only, focus on the translated text):
#1
{ORIGINAL_PREFIX}
Those who resist change may find themselves left behind.
{TRANSLATION_PREFIX}
那些抗拒变化的人可能会发现自己被抛在后面。

#2
{ORIGINAL_PREFIX}
On the other hand, those who embrace change can thrive in the new environment.
{TRANSLATION_PREFIX}
另一方面，那些接受变化的人可以在新环境中发展。

#3
{ORIGINAL_PREFIX}
Thus, it is important to adapt to changing circumstances and remain open to new opportunities.
{TRANSLATION_PREFIX}
因此，适应变化的环境并对新机会持开放态度是很重要的。


Example output:
#1
{TRANSLATION_PREFIX}
那些抗拒变化的人可能会发现自己被抛在后面。
{PROOFREAD_PREFIX}
那些抗拒变化的人可能会发现自己落伍了。

#2
{TRANSLATION_PREFIX}
另一方面，那些接受变化的人可以在新环境中发展。
{PROOFREAD_PREFIX}
相反，那些拥抱变化的人可以在新环境中如鱼得水。

#3
{TRANSLATION_PREFIX}
因此，适应变化的环境并对新机会持开放态度是很重要的。
{PROOFREAD_PREFIX}
因此，适应变化的环境并对新机会保持开放态度是非常重要的。


### retry_instructions
Please proofread the subtitles again, paying careful attention to ensure that each line is proofreaded separately, and that every line has a matching text.
Do not merge lines together during the proofread, it leads to incorrect timings and confusion for the reader.
'''

    def user(self, texts, translations, guideline=''):
        formated_texts = '\n'.join(
            [
                f'#{i}\n{ORIGINAL_PREFIX}\n{text}\n{TRANSLATION_PREFIX}\n{trans}\n' for i, (text, trans) in
                enumerate(zip(texts, translations), start=1)
            ])
        return f'''Translation guidelines from context reviewer:
{guideline}

Please proofread the following translated subtitles, which is from {self.src_lang_display} to {self.target_lang_display}:
{formated_texts}

Output:
'''

    def check_format(self, messages, content):
        # If message is for claude, use messages[0]
        user_input = messages[1]['content'] if len(messages) == 2 else messages[0]['content']
        original = re.findall(ORIGINAL_PREFIX + r'\n(.*?)\n' + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error(f'Fail to extract original text.')
            return False

        localized = re.findall(PROOFREAD_PREFIX + r'\s*(.*)', content, re.MULTILINE)

        if not localized:
            # TODO: Try to change chatbot_model if always fail
            logger.warning(f'Fail to extract translation.')
            logger.debug(f'Content: {content}')
            return False

        if len(original) != len(localized):
            logger.warning(
                f'Fail to ensure length consistent: original is {len(original)}, translation is {len(localized)}')
            logger.debug(f'original: {original}')
            logger.debug(f'translation: {original}')
            return False

        return True
