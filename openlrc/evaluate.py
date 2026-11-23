#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import abc

from openlrc.agents import TranslationEvaluatorAgent, create_chatbot
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

    def __init__(self, chatbot_model: str | ModelConfig | None = None):
        self._chatbot = create_chatbot(chatbot_model)
        self.agent = TranslationEvaluatorAgent(chatbot=self._chatbot)

    def evaluate(self, src_texts, target_texts, src_lang=None, target_lang=None):
        return self.agent.evaluate(src_texts, target_texts)

    def close(self):
        """Close the underlying chatbot connection."""
        self._chatbot.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EmbeddingTranslationEvaluator(TranslationEvaluator):
    """
    Evaluate the translated texts using embeddings.
    """

    def evaluate(self, src_texts, target_texts, src_lang, target_lang):
        pass
