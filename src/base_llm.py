from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate_answer(self, question: str, question_type: str, context: str = "", max_tokens: int = 100) -> str:
        """
        Abstract method to generate an answer based on a question and context.
        """
        pass

    @abstractmethod
    def generate_plot_creation_code(self, user_question: str, df, df_description) -> str:
        """
        Abstract method to generate plot creation code based on a dataframe and user question.
        """
        pass
