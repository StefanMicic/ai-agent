import os
from openai import OpenAI
from dotenv import load_dotenv
from src.base_llm import BaseLLM
from src.logging_config import logger
from src.prompts import SYSTEM_PROMPT, CONTEXT_PROMPT, SYSTEM_PROMPT_PLOT, CONTEXT_PROMPT_PLOT, SYSTEM_PROMPT_INTENT, CONTEXT_PROMPT_INTENT
import pandas as pd

load_dotenv()

class OpenaiLLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """
        Initializes the LLM class with the OpenAI API.

        Args:
            model_name (str): Name of the model to use. Default is "gpt-4o-mini".
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name: str = model_name
    
    def generate_answer(self, question: str, question_type: str, context: str = "", max_tokens: int = 100) -> str:
        """
        Generates an answer to a given question based on the question type.

        Args:
            question (str): The question for which the answer is to be generated.
            question_type (str): The type of the question, which determines the prompt template.
            context (str, optional): The context to be used in the prompt. Default is an empty string.
            max_tokens (int, optional): The maximum number of tokens for the generated answer. Default is 100.

        Returns:
            str: The generated answer.
        """
        if question_type == "general":
            system = SYSTEM_PROMPT
            prompt = CONTEXT_PROMPT.format(context=context, question=question)
        elif question_type == "intent":
            system = SYSTEM_PROMPT_INTENT
            prompt = CONTEXT_PROMPT_INTENT.format(question=question)
        else:
            return "Could not provide answer."
        
        logger.info(f"Getting Openai LLM answer for type {question_type}...")
        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=0.1
        )
        answer = chat_completion.choices[0].message.content
        logger.info(f"Openai LLM answer retrieved.")
        return answer
    
    def generate_plot_creation_code(self, user_question: str, df: pd.DataFrame, df_description: str) -> str:
        """
        Generates code for creating a plot based on the user's question and the DataFrame description.

        Args:
            user_question (str): The question asked by the user.
            df (pd.DataFrame): The DataFrame containing the data to be visualized.
            df_description (str): A description of the DataFrame and its contents.

        Returns:
            str: The generated code for creating a plot.
        """
        rows_num: int = len(df)
        cols_num: int = len(df.columns)
        cols_description: str = ""
        for col in df.columns:
            col_type = df.dtypes[col]
            cols_description += f"{col} ({col_type})\n"

        system_message: str = SYSTEM_PROMPT_PLOT.format(
            rows_num=rows_num, cols_num=cols_num, df_description=df_description, cols_description=cols_description)
        prompt_problem: str = CONTEXT_PROMPT_PLOT.format(user_question=user_question)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_problem},
        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content.strip()
