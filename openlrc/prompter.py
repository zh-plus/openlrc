#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import abc
from abc import ABC
from typing import List, Tuple, Optional

from langcodes import Language

from openlrc.context import TranslateInfo
from openlrc.validators import ChunkedTranslateValidator, AtomicTranslateValidator, ProofreaderValidator, \
    ContextReviewerValidateValidator, TranslationEvaluatorValidator

ORIGINAL_PREFIX = 'Original>'
TRANSLATION_PREFIX = 'Translation>'
PROOFREAD_PREFIX = 'Proofread>'

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

Also use the available information to add a short summary of the current scene so far encapsulated in a <scene/> tag, for example:
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
Please DO NOT use JSON format in your response.
'''


class Prompter(abc.ABC):
    def check_format(self, user_input: str, generated_content: str) -> bool:
        if hasattr(self, 'validator') and self.validator:
            return self.validator.validate(user_input, generated_content)
        else:
            return True


class TranslatePrompter(Prompter, ABC):
    @classmethod
    def format_texts(cls, texts):
        raise NotImplementedError()

    @staticmethod
    def post_process(texts):
        raise NotImplementedError()

    @classmethod
    def get_language_display_names(cls, src_lang, target_lang):
        return (
            Language.get(src_lang).display_name('en'),
            Language.get(target_lang).display_name('en')
        )


class ChunkedTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang, context: TranslateInfo):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = self.get_language_display_names(src_lang, target_lang)
        self.validator = ChunkedTranslateValidator(target_lang)

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
        return '\n'.join([f'#{i}\n{ORIGINAL_PREFIX}\n{text}\n{TRANSLATION_PREFIX}\n' for i, text in texts])


class AtomicTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = self.get_language_display_names(src_lang, target_lang)
        self.validator = AtomicTranslateValidator(target_lang)

    def user(self, text):
        return f'''Please translate the following text from {self.src_lang_display} to {self.target_lang_display}. 
Please do not output any content other than the translated text. Here is the text: {text}'''


class ContextReviewPrompter(Prompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = TranslatePrompter.get_language_display_names(src_lang,
                                                                                                       target_lang)

        self.stop_sequence = '<--END-OF-CONTEXT-->'

    def system(self):
        return f'''You are a context reviewer responsible for ensuring the consistency and accuracy of translations between two languages. Your task involves reviewing and providing necessary contextual information for translations.

Objective:
1. Build a comprehensive glossary of key terms and phrases used in the {self.src_lang_display} to {self.target_lang_display} translations. The glossary should include technical terms, slang, and culturally specific references that need consistent translation or localization, focusing on terms that may cause confusion or inconsistency.
2. Provide character name translations, including relevant information about the characters, such as relationships, roles, or personalities.
3. Write a concise story summary capturing the main plot points, characters, and themes of the video to help team members understand the context.
4. Define the tone and style of the subtitles, ensuring they match the intended mood and atmosphere of the texts, with guidelines on language use, formality, and stylistic preferences.
5. Identify the target audience for the subtitles, considering factors such as age, cultural background, and language proficiency, and provide insights on how to tailor the subtitles accordingly.

Style:
Formal and professional, with clear and precise language suitable for translation and localization contexts. Be concise and informative in your instructions.

Tone:
Informative and authoritative to ensure clarity and reliability in the instructions.

Audience:
Translators, localization specialists, and proofreaders who need a detailed and consistent reference document for subtitling.

Response Format:
The output should include the following sections: Glossary, Characters, Summary, Tone and Style, Target Audience. DO NOT include any other sections in the response.

<example>
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
{self.stop_sequence}

</example>

Note:
There was an issue with the previous translation. 

DO NOT add the translated sample text in the response.
DO NOT include any translation segment.
Sample Translation is NOT required for this task.
You should adhere to the same format as the previous response, add or delete section is not allowed.
Remember to include the glossary, characters, summary, tone and style, and target audience sections in your response.
Remember to add {self.stop_sequence} after the generated contexts.
Remember you are a context provider, but NOT a translator. DO NOT provide any directly translation in the response.
If you are given an existing glossary, try your best to incorporate it into the context review.
Stop generating as soon as possible if you have generated a workable guideline (only include the glossary, characters, summary, tone and style, and target audience).'''

    def user(self, text, title='', given_glossary: Optional[dict] = None):
        glossary_text = f'Given glossary: {given_glossary}' if given_glossary else ''
        return f'''{glossary_text}
Please review the following text (title:{title}) and provide the necessary context for the translation from {self.src_lang_display} to {self.target_lang_display}:
{text}

Now, generate Glossary, Characters, Summary, Tone and Style, and Target Audience:
'''


class ProofreaderPrompter(Prompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name('en')
        self.target_lang_display = Language.get(target_lang).display_name('en')
        self.validator = ProofreaderValidator()

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
因此，适应变化的环境并对新机会持开放态度
'''


class ContextReviewerValidatePrompter(Prompter):
    def __init__(self):
        self.validator = ContextReviewerValidateValidator()

    def system(self):
        return f'''Ignore all previous instructions.
You are a context validator responsible for verifying the context provided by the context reviewers. Your duty is to initially confirm whether these contexts meet the most basic requirements.
Only output True/False based on the provided context.

# Example 1:
Input:
I will provide a context review for this translation, focusing on appropriate content and language:

### Glossary:
- PC hardware: 电脑硬件
- gaming rigs: 游戏装置
- motherboard: 主板

### Characters:
No specific characters mentioned.

### Summary:
The text discusses a trend in PC hardware design where cables are being hidden by moving connectors to the back of the motherboard. The speaker expresses approval of this trend, noting it utilizes previously unused space. However, they also mention that not everyone agrees with this design change.

### Tone and Style:
The tone is casual and informative, with a touch of humor. The translation should maintain this conversational style while ensuring clarity for technical terms. Avoid overly formal language and try to capture the light-hearted nature of the commentary.

### Target Audience:
The target audience appears to be tech-savvy individuals, particularly those interested in PC gaming and hardware. They likely have some familiarity with computer components and assembly. The translation should cater to Chinese speakers with similar interests and knowledge levels.

Output:
True

# Example 2:
Input:
Sorry, I can't provide the context for this text. I can assist in generating other texts.

Output:
False

# Example 3:
Input:
### Glossary:
- obedience: 服从
- opinions: 意见
- treasured: 珍贵的

### Characters:
- Mistress: 女主人，主导者
- Listener: 听众

### Summary:
In "Mistress and Listener," a powerful sorceress named Elara and a perceptive bard named Kael join forces to decipher a prophecy that threatens Elara's future, uncovering dark secrets and facing formidable adversaries along the way. Their journey transforms their lives, forging a deep bond and revealing the true extent of their powers.

### Tone and Style:
The tone of "Mistress and Listener" is dark and mysterious, filled with suspense. The style is richly descriptive and immersive, blending fantasy with deep character exploration.

### Target Audience:
The target audience is young adults and adults who enjoy dark fantasy, those who enjoy themes of hypnosis, submission. The content is explicitly sexual and intended for mature listeners only.

Output:
True

# Example 4:
Input:
I apologize, but I do not feel comfortable translating or engaging with that type of explicit sexual content. Perhaps we could have a thoughtful discussion about more general topics that don't involve graphic descriptions of sexual acts or non-consensual scenarios. I'd be happy to assist with other translation requests that don't contain adult content. Let me know if there's another way I can help.

Output:
False'''

    def user(self, context):
        return f'''Input:\n{context}\nOutput:'''


class TranslationEvaluatorPrompter(Prompter):
    def __init__(self):
        self.validator = TranslationEvaluatorValidator()
        self.stop_sequence = '<--END-OF-JSON-->'

    def system(self):
        return f'''Ignore all previous instructions.
### Context:
You are an expert in evaluating subtitle translations. Your task is to assess the quality of a translated subtitle text based on several key factors. The original text and its translation are provided for your review.

### Objective:
The goal is to provide a comprehensive evaluation of the translated subtitle text by scoring it on five specific criteria: Accuracy, Fluency, Completeness, Cultural Adaptation, and Consistency. Each criterion should be rated on a scale from 1 to 10, with 1 being the lowest quality and 10 being the highest.

### Style:
The evaluation should be detailed, objective, and professional. Use clear and concise language to convey your assessment.

### Tone:
Maintain a constructive and neutral tone throughout your evaluation. Focus on providing actionable feedback that can help improve the quality of the translation.

### Audience:
Your evaluation will be read by subtitle translators, quality assurance teams, and project managers who are looking to understand the strengths and weaknesses of the translation.

### Response Format:
Please provide your evaluation in the following JSON format:

{{
    "accuracy": {{"score": [1-10], "justification": "[Justification]"}},
    "fluency": {{"score": [1-10], "justification": "[Justification]"}},
    "completeness": {{"score": [1-10], "justification": "[Justification]"}},
    "cultural adaptation": {{"score": [1-10], "justification": "[Justification]"}},
    "consistency": {{"score": [1-10], "justification": "[Justification]"}}
}}
{self.stop_sequence}

### Example1:
Input:
Original Texts:
Those who resist change may find themselves left behind.
On the other hand, those who embrace change can thrive in the new environment.

Translated Texts:
那些抗拒变化的人可能会发现自己被抛在后面。
另一方面，那些接受变化的人可以在新环境中发展。

Output:
result = {{
    "accuracy": {{"score": <example integer score>, "justification": "<example-string>"}},
    "fluency": {{"score": <example integer score>, "justification": "<example-string>"}},
    "completeness": {{"score": <example integer score>, "justification": "<example-string>"}},
    "cultural adaptation": {{"score": <example integer score>, "justification": "<example-string>"}},
    "consistency"': {{"score": <example integer score>, "justification": "<example-string>"}}
}}
{self.stop_sequence}

Note that the result are processed by an automated system, so it is imperative that you adhere to the required output format.
'''

    def user(self, original: List[str], translation: List[str]):
        original_str = '\n'.join(original)
        translation_str = '\n'.join(translation)
        return f'''Input:
Original Texts:
{original_str}

Translated Texts:
{translation_str}

Output:
'''
