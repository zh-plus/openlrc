class SameLanguageException(Exception):
    """
    Raised when the source language and target language are the same.
    """

    def __init__(self):
        super().__init__(
            'Source language and target language are the same, no need to translate. '
            'If you want to translate, set force_translate=True.'
        )
