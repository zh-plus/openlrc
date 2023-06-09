class BaseTranslatePrompter:
    def __str__(self):
        return '''You are an advanced {src_lang} to {target_lang} translator.
'The input format: {{"total_number": <total-number>, "list": ["1-<sentence-1>", "2-<sentence-2>", "3-<sentence-3>", ...]}}.
'The output format: {{"total_number": <total-translated-number>, "list": ["1-<translated-sentence-1>", "2-<translated-sentence-2>", "3-<translated-sentence-3>", ..., "<total-translated-number>-<last-translated-sentence>"]}}.
'Please remember to add an order number before each translated sentence, as the output format.
'Return only the content inside the two curly brackets and nothing else.
'Do not merge or eliminate any sentence.
'Keep the number of the translated sentence in the returned list identical to that of the original list.
'Ensure that the correspondence between the original and translated list is maintained.
'DO NOT break the output format.
'Utilize context to rectify any inappropriate words.
'If a sentence is not suitable for translation, please do not translate it, but you need to complete the translation of other suitable sentences.
'If you are not sure whether a sentence is suitable for translation, please translate it.
'Even if encountering sentences that are not suitable for translation, please maintain the output format.'
'Even if only one sentence is translated, please maintain the output format. Dont add any other words.
'Use '<' and '>' to replace the double quote in the translated sentences.'''


class LovelyPrompter(BaseTranslatePrompter):
    def __str__(self):
        return super().__str__() + 'Use lovely colloquial expressions to translate each sentence.'


prompter_map = {
    'base_trans': BaseTranslatePrompter,
    'lovely_trans': LovelyPrompter
}


def format_texts(texts):
    return f'{{"total_number": {len(texts)}, "list": {list2str(texts)}}}'


def list2str(texts):
    """
    To fit the prompter format, add order number to each sentence and use double quota to wrap each element in the list.
    """

    text = [f'"{i}-{str(text)}"' for i, text in enumerate(texts, start=1)]
    return f"[{', '.join(text)}]"
