#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
from openai.types.chat import ChatCompletion


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


class LengthExceedException(ChatBotException):
    """
    Raised when the length of generated response exceeds the limit.
    """

    def __init__(self, response: ChatCompletion):
        super().__init__(
            f'Failed to get completion. Exceed max token length. '
            f'Prompt tokens: {response.usage.prompt_tokens}, '
            f'Completion tokens: {response.usage.completion_tokens}, '
            f'Total tokens: {response.usage.total_tokens} '
            f'Reduce chunk_size may help.'
        )


class OpenaiFailureException(Exception):
    """
    Raised when OpenAI API fails to generate response.
    """

    def __init__(self):
        super().__init__('OpenAI API failed to generate response.')


class FfmpegException(Exception):
    def __init__(self, message):
        super().__init__(message)


class TranscribeException(Exception):
    def __init__(self, message):
        super().__init__(message)


class DependencyException(Exception):
    def __init__(self, message):
        super().__init__(f'Dependency not correctly installed: {message}')
