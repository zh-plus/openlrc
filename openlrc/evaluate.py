#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import abc
from typing import Union

from openlrc.agents import TranslationEvaluatorAgent
from openlrc.models import ModelConfig


class TranslationEvaluator(abc.ABC):
    """
    Base class for all evaluators.
    """

    @abc.abstractmethod
    def evaluate(self, src_texts, target_texts, src_lang, target_lang):
        """
        Evaluate the translated texts.
        :return: The evaluation result.
        """
        raise NotImplementedError()


class LLMTranslationEvaluator(TranslationEvaluator):
    """
    Evaluate the translated texts using large language models.
    """

    def __init__(self, chatbot_model: Union[str, ModelConfig] = 'gpt-4.1-nano'):
        self.agenet = TranslationEvaluatorAgent(chatbot_model=chatbot_model)

    def evaluate(self, src_texts, target_texts, src_lang=None, target_lang=None):
        return self.agenet.evaluate(src_texts, target_texts)


class EmbeddingTranslationEvaluator(TranslationEvaluator):
    """
    Evaluate the translated texts using embeddings.
    """

    def evaluate(self, src_texts, target_texts, src_lang, target_lang):
        pass
