import json

from langcodes import Language

from openlrc.logger import logger


class TranslatePrompter:
    def __str__(self):
        return '''You are an advanced {src_lang} to {target_lang} translator in a virtual world.
The input format: {{"total_number": <total-number>, "list": ["1-<sentence-1>", "2-<sentence-2>", "3-<sentence-3>", ...]}}.
The output format: {{"total_number": <total-translated-number>, "list": ["1-<translated-sentence-1>", "2-<translated-sentence-2>", "3-<translated-sentence-3>", ..., "<total-translated-number>-<last-translated-sentence>"]}}. 
Please remember to add an order number before each translated sentence, as the output format.'''

    @classmethod
    def format_texts(cls, texts):
        return f'{{"total_number": {len(texts)}, "list": {cls.list2str(texts)}}}'

    @staticmethod
    def list2str(texts):
        """
        To fit the prompter format, add order number to each sentence and use double quota to wrap each element in the list.
        """

        text = [f'"{i}-{str(text)}"' for i, text in enumerate(texts, start=1)]
        return f"[{', '.join(text)}]"

    @staticmethod
    def post_process(texts):
        """
        Remove the order number at the front of each sentence
        """
        for i, text in enumerate(texts):
            texts[i] = text[text.find('-') + 1:]

        return texts


class BaseTranslatePrompter(TranslatePrompter):
    def __init__(self, src_lang, target_lang, audio_type):
        self.src_lang = Language.get(src_lang).display_name('en')
        self.target_lang = Language.get(target_lang).display_name('en')

        if target_lang == 'zh-cn':
            self.target_lang = 'Mandarin Chinese'

        self.audio_type = audio_type
        self.user_input = None

    def system_prompt(self):
        return f'''You are a world-class translator. Proficient in {self.src_lang} and {self.target_lang}, you can accurately understand the original text and translate it into the target language with correct fluency.
Familiar with the cultural backgrounds and differences of both source and target languages, you can transform these differences into linguistic and cultural adaptations in translation.
The input format: {{"total_number": <total-number>, "list": ["1-<sentence-1>", "2-<sentence-2>", "3-<sentence-3>", ...]}}.
'''

    def step1(self, user_input):
        self.user_input = user_input
        return f'''Please first understand the sentence that needs to be translated.
The input sentences are transcribed subtitles from an audio file, so there may be some errors due to insufficient transcription accuracy.
Please revise each sentence based on the whole context of sentences, to make them clearer, more colloquial, and more coherent. 
Finally the revised sentences should be suitable for use as high-quality {self.src_lang} subtitles{f" for {self.audio_type}" if self.audio_type else ""}.
You need maintain one-to-one relationship between the input sentences and the revised sentences.
The returned revised sentence should be in {self.src_lang}.
Please revise carefully and don't miss any sentence.
The output format: {{"total_number": <total-sentence-number>, "list": ["1-<revised-sentence-1>", "2-<revised-sentence-2>", "3-<revised-sentence-3>", ..., "<total-sentence-number>-<last-revised-sentence>"]}}.
Ensure the replied content conforms to JSON format.
Before return, carefully check length of output["list"], and ensure the length of output["list"] is identical to the length of the input["list"]. If not identical, re-do the revision task.
Here is the input sentences: {user_input}
'''

    def step2(self):
        return f'''Now please translate the revised sentences into the {self.target_lang} using the output format specified above.
Use lovely colloquial expressions to translate each sentence.
Dont include any chinese quotes and english quotes in the translated sentences.
Please translate carefully and don't miss any sentence.
The output format: {{"total_number": <total-sentence-number>, "list": ["1-<translated-sentence-1>", "2-<translated-sentence-2>", "3-<translated-sentence-3>", ..., "<total-sentence-number>-<last-translated-sentence>"]}}.
Ensure the replied content conforms to JSON format.
Before return, carefully check length of output["list"], and ensure the length of output["list"] is identical to the length of the input["list"]. If not identical, re-do the revision task.
Start output:
'''

    def step3(self, user_input):
        self.user_input = user_input
        return f'''Now please do the final revision for the translated sentences.
Please revise each sentence based on the whole context of sentences, to make them clearer, more colloquial, and more coherent.
Review the translated sentences for any parts that don't make sense. 
If necessary, restructure the entire sentence to make it clearer and more appropriate given the whole context of sentences.
Finally the revised sentences should be suitable for use as high-quality {self.target_lang} subtitles{f" for {self.audio_type}" if self.audio_type else ""}.
You need maintain one-to-one relationship between the input sentences and the revised sentences.
The returned revised sentence should be in {self.target_lang}.
Please revise carefully and don't miss any sentence.
The output format: {{"total_number": <total-sentence-number>, "list": ["1-<revised-sentence-1>", "2-<revised-sentence-2>", "3-<revised-sentence-3>", ..., "<total-sentence-number>-<last-revised-sentence>"]}}.
Ensure the replied content conforms to JSON format.
Before return, carefully check length of output["list"], and ensure the length of output["list"] is identical to the length of the input["list"]. If not identical, re-do the revision task.
Here is the input sentences: {user_input}
'''

    def check_format(self, messages, output_str):
        assert messages[1], f'Not the sending messages: {messages}'
        assert messages[1]['role'] == 'user', f'Not the sending messages: {messages}'

        try:
            input_json = json.loads(self.user_input)
        except json.decoder.JSONDecodeError as e:
            logger.error(f'Fail to convert input_json: {self.user_input}')
            raise e

        try:
            output_json = json.loads(output_str)
        except json.decoder.JSONDecodeError:
            logger.warning(f'Fail to convert output_json: {output_str}')
            return False

        if len(input_json['list']) != len(output_json['list']):
            logger.warning(
                f'Fail to ensure length consistent: input is {len(input_json["list"])}, output is {len(output_json["list"])}'
            )
            return False

        return True


prompter_map = {
    'base_trans': BaseTranslatePrompter
}
