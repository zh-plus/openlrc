#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import abc
import html
import re
from abc import ABC

from langcodes import Language

from openlrc.context import TranslateInfo
from openlrc.llama_resources import HY_MT2_PROMPT_PROFILE
from openlrc.validators import (
    AtomicTranslateValidator,
    BaseValidator,
    ChunkedTranslateValidator,
    ContextReviewerValidateValidator,
    HyMT2DelimiterTranslateValidator,
    LeanTranslateValidator,
    ProofreaderValidator,
    TranslationEvaluatorValidator,
)

ORIGINAL_PREFIX = "Original>"
TRANSLATION_PREFIX = "Translation>"
PROOFREAD_PREFIX = "Proofread>"

BASE_TRANSLATE_INSTRUCTION = f"""Ignore all previous instructions.
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
"""


class Prompter(abc.ABC):
    validator: "BaseValidator | None" = None

    def check_format(self, user_input: str, generated_content: str) -> bool:
        if self.validator:
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
        return (Language.get(src_lang).display_name("en"), Language.get(target_lang).display_name("en"))


class ChunkedTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang, context: TranslateInfo):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = self.get_language_display_names(src_lang, target_lang)
        self.validator = ChunkedTranslateValidator(target_lang)

        self.audio_type = context.audio_type
        self.title = context.title
        self.glossary = context.glossary
        self.user_prompt = f"""Translation guidelines from context reviewer:
{{guideline}}

Previous summaries:
{{summaries_str}}

<chunk_id> Scene 1 Chunk {{chunk_num}} <chunk_id>

Please translate these subtitles for {self.audio_type} from {self.src_lang_display} to {self.target_lang_display}.\n
{{user_input}}
<summary></summary>
<scene></scene>"""

    def system(self) -> str | None:
        return BASE_TRANSLATE_INSTRUCTION

    def user(self, chunk_num: int, user_input: str, summaries: list[str] | str = "", guideline: str = "") -> str:
        summaries_str = "\n".join(f"Chunk {i}: {summary}" for i, summary in enumerate(summaries, 1))
        return self.user_prompt.format(
            summaries_str=summaries_str, chunk_num=chunk_num, user_input=user_input, guideline=guideline
        ).strip()

    @property
    def formatted_glossary(self):
        if not self.glossary:
            return ""
        glossary_strings = "\n".join(f"{k}: {v}" for k, v in self.glossary.items())
        result = f"""
# Glossary
Use the following glossary to ensure consistency in your translations:
<preferred-translation>
{glossary_strings}
</preferred-translation>
"""
        return result

    @classmethod
    def format_texts(cls, texts: list[tuple[int, str]]):
        return "\n".join([f"#{i}\n{ORIGINAL_PREFIX}\n{text}\n{TRANSLATION_PREFIX}\n" for i, text in texts])


class AtomicTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = self.get_language_display_names(src_lang, target_lang)
        self.validator = AtomicTranslateValidator(target_lang)

    def user(self, text, *, context: str = "", guideline: str = ""):
        context_text = f"\nContext:\n{context}" if context else ""
        guideline_text = f"\nTranslation guideline:\n{guideline}" if guideline else ""
        return f"""Please translate the following text from {self.src_lang_display} to {self.target_lang_display}.
Please do not output any content other than the translated text.{context_text}{guideline_text}
Here is the text: {text}"""


class HyMT2AtomicTranslatePrompter(AtomicTranslatePrompter):
    """Atomic fallback prompt following Hy-MT2's official translation style."""

    def user(self, text, *, context: str = "", guideline: str = ""):
        context_text = f"[Background Information]\n{context}\n" if context else ""
        guideline_text = f"[Translation Guidance]\n{guideline}\n" if guideline else ""
        return (
            context_text + guideline_text + f"Translate the following text into {self.target_lang_display}. "
            "Note that you should only output the translated result without any additional explanation. "
            "Do not output XML/HTML tags, Markdown, code fences, or labels:\n"
            f"{text}"
        )


LEAN_TRANSLATE_INSTRUCTION = """You are a subtitle translator. Translate each numbered line from {src_lang} to {target_lang}.

Rules:
- Output ONLY the translation for each line, prefixed with its line number.
- Preserve the original line numbering exactly.
- Do not merge or split lines.
- Do not add explanations, notes, or commentary.
- Maintain natural, colloquial style suitable for subtitles.
- Use the provided context to ensure consistency.

Output format (one block per line):
#<id>
<translation>

Example:
#200
在变化的时代中，
#501
生存的秘诀是不断进化。
"""

LEAN_RETRY_INSTRUCTION = """Previous response had formatting issues. \
Please ensure each translated line starts with #<id> on its own line, \
followed by the translation on the next line. Do not add any extra text."""

HY_MT2_DELIMITER_RETRY_INSTRUCTION = """Previous response had delimiter formatting issues. \
Output only <seg id="N">translated text</seg> blocks for every original id. \
Keep each opening and closing <seg> delimiter exactly, translate only the text inside the tags, \
and do not add explanations, labels, Markdown, or code fences."""


class LeanTranslatePrompter(TranslatePrompter):
    """Prompter for :class:`LeanTranslator`.

    Produces a compact system prompt (~150 tokens) and a user prompt that
    carries three layers of fixed-budget context: global summary, terminology
    map, and a sliding window of recent translation pairs.
    """

    def __init__(self, src_lang: str, target_lang: str):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = self.get_language_display_names(src_lang, target_lang)
        # Validator is set per-chunk via update_expected_ids() before each call.
        self.validator = None

    def update_expected_ids(self, expected_ids: list[int]) -> None:
        """Refresh the validator with the line IDs of the current chunk."""
        self.validator = LeanTranslateValidator(expected_ids)

    def parse_translations(self, raw: str) -> dict[int, str]:
        return LeanTranslateValidator.parse_anchored_translations(raw)

    def retry_instruction(self) -> str:
        return LEAN_RETRY_INSTRUCTION

    def system(self) -> str | None:
        return LEAN_TRANSLATE_INSTRUCTION.format(src_lang=self.src_lang_display, target_lang=self.target_lang_display)

    def user(
        self,
        user_input: str,
        *,
        summary: str = "",
        characters: str = "",
        terminology: str = "",
        sliding_window: str = "",
    ) -> str:
        context_parts: list[str] = []
        if summary:
            context_parts.append(f"Summary: {summary}")
        if characters:
            context_parts.append(f"Characters:\n{characters}")
        if terminology:
            context_parts.append(f"Terminology:\n{terminology}")
        if sliding_window:
            context_parts.append(f"Recent translations:\n{sliding_window}")

        sections: list[str] = []
        if context_parts:
            sections.append("[Context]\n" + "\n\n".join(context_parts))
        sections.append(
            f"Please translate the following subtitles"
            f" from {self.src_lang_display} to {self.target_lang_display}:\n\n"
            f"{user_input}"
        )
        return "\n\n".join(sections)

    @classmethod
    def format_texts(cls, texts: list[tuple[int, str]]) -> str:  # type: ignore[override]
        """Format chunk lines as ``#id\\ntext`` blocks (no Original>/Translation> prefixes)."""
        return "\n".join(f"#{line_id}\n{text}" for line_id, text in texts)


class HyMT2DelimiterTranslatePrompter(LeanTranslatePrompter):
    """Hy-MT2 subtitle prompt using official-style delimiter preservation."""

    _SEG_ID_RE = re.compile(r"<seg\s+id=[\"']?(\d+)[\"']?\s*>", re.IGNORECASE)
    exact_id_alignment = True

    def system(self) -> None:
        return None

    def update_expected_ids(self, expected_ids: list[int]) -> None:
        self.validator = HyMT2DelimiterTranslateValidator(expected_ids)

    def parse_translations(self, raw: str) -> dict[int, str]:
        parsed = HyMT2DelimiterTranslateValidator.parse_delimited_translations(raw)
        if parsed:
            return parsed
        return LeanTranslateValidator.parse_anchored_translations(raw)

    def retry_instruction(self) -> str:
        return HY_MT2_DELIMITER_RETRY_INSTRUCTION

    @classmethod
    def format_texts(cls, texts: list[tuple[int, str]]) -> str:  # type: ignore[override]
        return "\n".join(f'<seg id="{line_id}">{html.escape(text, quote=False)}</seg>' for line_id, text in texts)

    def user(
        self,
        user_input: str,
        *,
        summary: str = "",
        characters: str = "",
        terminology: str = "",
        sliding_window: str = "",
        style: str = "",
        neighboring_context: str = "",
        story_so_far: str = "",
        current_scene: str = "",
    ) -> str:
        context_parts: list[str] = []
        if summary:
            context_parts.append(f"Background summary:\n{summary}")
        if characters:
            context_parts.append(f"Characters:\n{characters}")
        if terminology:
            context_parts.append(f"Reference the following translations:\n{terminology}")
        if sliding_window:
            context_parts.append(f"Recent translations:\n{sliding_window}")
        if style:
            context_parts.append(f"Required tone and style:\n{style}")
        if neighboring_context:
            context_parts.append(f"Read-only neighboring source subtitles:\n{neighboring_context}")
        if story_so_far:
            context_parts.append(f"Story so far after this source chunk:\n{story_so_far}")
        if current_scene:
            context_parts.append(f"Current scene for this source chunk:\n{current_scene}")

        sections: list[str] = []
        if context_parts:
            sections.append("[Background Information]\n" + "\n\n".join(context_parts))
        expected_ids = self._SEG_ID_RE.findall(user_input)
        id_list = ", ".join(f'id="{line_id}"' for line_id in expected_ids)
        count_rule = (
            f"- Output exactly {len(expected_ids)} <seg> blocks, one for each segment id: {id_list}.\n"
            if expected_ids
            else ""
        )
        sections.append(
            f"Please accurately translate the following subtitle text from {self.src_lang_display} "
            f"into {self.target_lang_display}. "
            "Note that you should only output the translated result without any additional explanation.\n\n"
            "Delimiter rules:\n"
            '- Retain every opening <seg id="N"> delimiter and every closing </seg> delimiter exactly.\n'
            "- Only translate the text between each opening and closing tag.\n"
            f"{count_rule}"
            "- Do not merge, split, omit, or reorder subtitle lines.\n"
            "- Do not add explanations, labels, Markdown, code fences, or any text outside the <seg> blocks.\n"
            "- Keep the translation natural and concise for subtitles.\n\n"
            "[Source Text]\n"
            f"{user_input}"
        )
        return "\n\n".join(sections)


def create_atomic_translate_prompter(
    src_lang: str, target_lang: str, prompt_profile: str = "default"
) -> AtomicTranslatePrompter:
    if prompt_profile == HY_MT2_PROMPT_PROFILE:
        return HyMT2AtomicTranslatePrompter(src_lang, target_lang)
    return AtomicTranslatePrompter(src_lang, target_lang)


def create_lean_translate_prompter(
    src_lang: str, target_lang: str, prompt_profile: str = "default"
) -> LeanTranslatePrompter:
    if prompt_profile == HY_MT2_PROMPT_PROFILE:
        return HyMT2DelimiterTranslatePrompter(src_lang, target_lang)
    return LeanTranslatePrompter(src_lang, target_lang)


class ContextReviewPrompterBase(Prompter, ABC):
    """Interface contract for prompters used by :class:`ContextReviewerAgent`.

    Subclasses must implement all abstract methods.  ``ContextReviewerAgent``
    relies on this interface for single-pass generation, chunked generation,
    and guideline merging.
    """

    expected_sections: list[str]
    stop_sequence: str

    @abc.abstractmethod
    def system(self) -> str: ...

    @abc.abstractmethod
    def user(self, text: str, title: str = "", given_glossary: dict | None = None) -> str: ...

    @abc.abstractmethod
    def user_partial(
        self, text: str, chunk_index: int, total_chunks: int, title: str = "", given_glossary: dict | None = None
    ) -> str: ...

    @abc.abstractmethod
    def merge_system(self) -> str: ...

    @abc.abstractmethod
    def merge_user(self, partial_guidelines: list[str], title: str = "") -> str: ...


class ContextReviewPrompter(ContextReviewPrompterBase):
    expected_sections = ["glossary", "characters", "summary", "tone and style", "target audience"]

    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = TranslatePrompter.get_language_display_names(
            src_lang, target_lang
        )

        self.stop_sequence = "<--END-OF-CONTEXT-->"

    def system(self):
        return f"""You are a context reviewer responsible for ensuring the consistency and accuracy of translations between two languages. Your task involves reviewing and providing necessary contextual information for translations.

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
There was an issue with the previous review. 

DO NOT add the translated sample text in the response.
DO NOT include any translation segment.
Sample Translation is NOT required for this task.
You should adhere to the same format as the previous response, add or delete section is not allowed.
Remember to include the glossary, characters, summary, tone and style, and target audience sections in your response.
Remember to add {self.stop_sequence} after the generated contexts.
Remember you are a context provider, but NOT a translator. DO NOT provide any directly translation in the response.
If you are given an existing glossary, try your best to incorporate it into the context review.
Stop generating as soon as possible if you have generated a workable guideline (only include the glossary, characters, summary, tone and style, and target audience).
I may give you a glossary. Please provide me with a new glossary that does not overlap with the one I give you."""

    def user(self, text, title="", given_glossary: dict | None = None):
        glossary_text = f"Given glossary: {given_glossary}" if given_glossary else ""
        return f"""{glossary_text}
Please review the following text (title:{title}) and provide the necessary context for the translation from {self.src_lang_display} to {self.target_lang_display}:
{text}

Now, generate Glossary, Characters, Summary, Tone and Style, and Target Audience:
"""

    def user_partial(self, text, chunk_index: int, total_chunks: int, title="", given_glossary: dict | None = None):
        glossary_text = f"Given glossary: {given_glossary}" if given_glossary else ""
        return f"""{glossary_text}
The following is section {chunk_index} of {total_chunks} from the subtitle file (title:{title}).
Note: This is only a portion of the full content. Focus on the terms, characters, and events present in this section.

Please review the following text and provide the necessary context for the translation from {self.src_lang_display} to {self.target_lang_display}:
{text}

Now, generate Glossary, Characters, Summary, Tone and Style, and Target Audience for this section:
"""

    def merge_system(self):
        return f"""You are a context reviewer. You will receive multiple partial translation guidelines \
generated from different sections of the same subtitle file ({self.src_lang_display} to {self.target_lang_display}). \
Merge them into a single comprehensive guideline following these rules:

Merge rules:
- Glossary: Union of all entries. If the same term appears with different translations, keep the more specific one.
- Characters: Union of all characters. Merge descriptions for the same character across sections.
- Summary: Synthesize a coherent summary covering all sections in chronological order.
- Tone and Style: Use the most representative description. If sections differ, note the variation.
- Target Audience: Use the most comprehensive description.

The final output must contain exactly these sections: Glossary, Characters, Summary, Tone and Style, Target Audience. \
DO NOT include any other sections.

<example>
Example Input:
### Partial guideline 1:
### Glossary:
- suspect: 嫌疑人
### Characters:
- John: 约翰, a detective
### Summary:
John begins investigating a case.
### Tone and Style:
Formal and professional.
### Target Audience:
Adult viewers interested in crime dramas.

---

### Partial guideline 2:
### Glossary:
- uptown: 市中心
- suspect: 嫌犯
### Characters:
- John: 约翰, a detective with 10 years of experience
- Sarah: 萨拉, John's partner
### Summary:
Sarah joins John and they plan to search the uptown area.
### Tone and Style:
Formal and serious.
### Target Audience:
Adult viewers who enjoy police procedurals.

Example Output:
### Glossary:
- suspect: 嫌疑人
- uptown: 市中心
### Characters:
- John: 约翰, a detective with 10 years of experience
- Sarah: 萨拉, John's partner
### Summary:
John begins investigating a case. Sarah joins him and they plan to search the uptown area for the suspect.
### Tone and Style:
Formal, professional, and serious, reflecting the nature of the investigation.
### Target Audience:
Adult viewers interested in crime dramas and police procedurals.
{self.stop_sequence}
</example>

Remember to add {self.stop_sequence} after the generated contexts."""

    def merge_user(self, partial_guidelines: list[str], title: str = ""):
        parts = "\n\n---\n\n".join(f"### Partial guideline {i + 1}:\n{g}" for i, g in enumerate(partial_guidelines))
        return f"""Title: {title}

{parts}

Now, merge the above into one comprehensive guideline with Glossary, Characters, Summary, Tone and Style, and Target Audience:
"""


class LeanContextReviewPrompter(ContextReviewPrompterBase):
    """Compact CR prompter for :class:`LeanTranslator`.

    Requests only Glossary, Characters, and Summary in YAML-like format.
    """

    expected_sections = ["glossary", "characters", "summary"]

    def __init__(self, src_lang: str, target_lang: str):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display, self.target_lang_display = TranslatePrompter.get_language_display_names(
            src_lang, target_lang
        )
        self.stop_sequence = "<--END-OF-CONTEXT-->"

    def system(self) -> str:
        return f"""You are a context reviewer for subtitle translation from {self.src_lang_display} to {self.target_lang_display}.

Generate exactly three sections in YAML format:

glossary:
  - <source term>: <translated term>
characters:
  - <name>: <translated name>, <brief description>
summary: <one-paragraph plot summary>

Rules:
- Glossary: key terms, slang, and culturally specific references that need consistent translation.
- Characters: name translations with relationships or roles.
- Summary: concise plot summary to help translators understand context.
- Do NOT include any other sections.
- Do NOT include translations of the source text.
- End your response with {self.stop_sequence}

Example output:
glossary:
  - suspect: 嫌疑人
  - uptown: 市中心
characters:
  - John: 约翰, a detective with 10 years of experience
  - Sarah: 萨拉, John's partner
summary: John and Sarah discuss their plan to locate a suspect in the uptown area.
{self.stop_sequence}"""

    def user(self, text: str, title: str = "", given_glossary: dict | None = None) -> str:
        glossary_text = f"Given glossary: {given_glossary}\n" if given_glossary else ""
        return f"""{glossary_text}\
Please review the following text (title: {title}) and provide context for translation from {self.src_lang_display} to {self.target_lang_display}:
{text}

Now, generate glossary, characters, and summary:
"""

    def user_partial(
        self, text: str, chunk_index: int, total_chunks: int, title: str = "", given_glossary: dict | None = None
    ) -> str:
        glossary_text = f"Given glossary: {given_glossary}\n" if given_glossary else ""
        return f"""{glossary_text}\
Section {chunk_index} of {total_chunks} from subtitle file (title: {title}).

Please review and provide context for translation from {self.src_lang_display} to {self.target_lang_display}:
{text}

Now, generate glossary, characters, and summary for this section:
"""

    def merge_system(self) -> str:
        return f"""You are a context reviewer. Merge multiple partial guidelines \
({self.src_lang_display} to {self.target_lang_display}) into one using YAML format.

Merge rules:
- glossary: Union of all entries. Keep the more specific translation for duplicates.
- characters: Union of all characters. Merge descriptions for the same character.
- summary: Synthesize a coherent summary covering all sections chronologically.

Output exactly three sections: glossary, characters, summary.
End with {self.stop_sequence}"""

    def merge_user(self, partial_guidelines: list[str], title: str = "") -> str:
        parts = "\n\n---\n\n".join(f"Partial guideline {i + 1}:\n{g}" for i, g in enumerate(partial_guidelines))
        return f"""Title: {title}

{parts}

Now, merge into one guideline with glossary, characters, and summary:
"""


class ProofreaderPrompter(Prompter):
    def __init__(self, src_lang, target_lang):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_display = Language.get(src_lang).display_name("en")
        self.target_lang_display = Language.get(target_lang).display_name("en")
        self.validator = ProofreaderValidator()

    def system(self):
        return f"""Ignore all previous instructions.
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
"""

    def user(self, texts: list[str], translations: list[str], guideline: str) -> str:
        paired = "\n".join(
            f"#{i}\n{ORIGINAL_PREFIX}\n{text}\n{TRANSLATION_PREFIX}\n{trans}"
            for i, (text, trans) in enumerate(zip(texts, translations), 1)
        )
        return (
            f"Translation guideline:\n{guideline}\n\n"
            f"Please proofread the following translated text "
            f"(the original texts are for reference only, focus on the translated text):\n{paired}"
        )


class ContextReviewerValidatePrompter(Prompter):
    def __init__(self):
        self.validator = ContextReviewerValidateValidator()

    def system(self):
        return """Ignore all previous instructions.
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
False"""

    def user(self, context):
        return f"""Input:\n{context}\nOutput:"""


class TranslationEvaluatorPrompter(Prompter):
    def __init__(self):
        self.validator = TranslationEvaluatorValidator()
        self.stop_sequence = "<--END-OF-JSON-->"

    def system(self):
        return f"""Ignore all previous instructions.
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
"""

    def user(self, original: list[str], translation: list[str]):
        original_str = "\n".join(original)
        translation_str = "\n".join(translation)
        return f"""Input:
Original Texts:
{original_str}

Translated Texts:
{translation_str}

Output:
"""
