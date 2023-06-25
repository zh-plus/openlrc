#  Copyright (C) 2023. Hao Zheng
#  All rights reserved.

class SameLanguageException(Exception):
    """
    Raised when the source language and target language are the same.
    """

    def __init__(self):
        super().__init__(
            'Source language and target language are the same, no need to translate. '
            'If you want to translate, set force_translate=True.'
        )


class ChatBotException(Exception):
    """
    Raised when chatbot fails to generate response.
    """

    def __init__(self, message):
        super().__init__(message)


class FfmpegException(Exception):
    def __init__(self, message):
        super().__init__(message)
