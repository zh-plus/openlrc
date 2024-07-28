#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import abc

from openlrc.agents import TranslationEvaluatorAgent
from openlrc.logger import logger


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

    def __init__(self, chatbot_model: str = 'gpt-4o-mini'):
        self.agenet = TranslationEvaluatorAgent(chatbot_model=chatbot_model)
        self.recommended_model = {
            'gpt-4',
            'claude-3-sonnet',
            'claude-3-opus',
            'gemini-1.5-pro'
        }

        for m in self.recommended_model:
            if chatbot_model.startswith(m):
                self.agenet = TranslationEvaluatorAgent(chatbot_model=chatbot_model)
                break
        else:
            logger.warning(f'Chatbot model {chatbot_model} is not in the recommended list for evaluating translations.')

    def evaluate(self, src_texts, target_texts, src_lang=None, target_lang=None):
        return self.agenet.evaluate(src_texts, target_texts)


class EmbeddingTranslationEvaluator(TranslationEvaluator):
    """
    Evaluate the translated texts using embeddings.
    """

    def evaluate(self, src_texts, target_texts, src_lang, target_lang):
        pass
