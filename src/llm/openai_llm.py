import os
import json
from openai import OpenAI
import pandas as pd
from src.api.constants import CHAT_HISTORY_DIR
from src.llm.base_llm import BaseLLM
from src.config.logging_config import logger
from src.prompts.prompts import SYSTEM_PROMPT, CONTEXT_PROMPT, SYSTEM_PROMPT_PLOT, CONTEXT_PROMPT_PLOT, SYSTEM_PROMPT_INTENT, CONTEXT_PROMPT_INTENT, SYSTEM_PROMPT_CSV_SELECTION, CONTEXT_PROMPT_CSV_SELECTION
from dotenv import load_dotenv

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
        self.history_file = os.path.join(CHAT_HISTORY_DIR, "openai.json")
        self.chat_history = self.load_chat_history()

    def load_chat_history(self):
        """Loads chat history from a JSON file."""
        if not os.path.exists(CHAT_HISTORY_DIR):
            os.makedirs(CHAT_HISTORY_DIR)
        
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON format in {self.history_file}, resetting chat history.")
                    return []
        return []

    def save_chat_history(self):
        """Saves chat history to a JSON file."""
        with open(self.history_file, "w", encoding="utf-8") as file:
            json.dump(self.chat_history, file, ensure_ascii=False, indent=4)
    
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
        self.chat_history = self.load_chat_history()
        if question_type == "general":
            system = SYSTEM_PROMPT
            prompt = CONTEXT_PROMPT.format(context=context, question=question)
        elif question_type == "intent":
            system = SYSTEM_PROMPT_INTENT
            prompt = CONTEXT_PROMPT_INTENT.format(question=question)
        else:
            return "Could not provide answer."
        
        logger.info(f"Getting Openai LLM answer for type {question_type}...")
        messages = [{"role": "system", "content": system}]
        if question_type != "intent":
            messages += self.chat_history[-10:]
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=0.1
        )
        answer = chat_completion.choices[0].message.content
        logger.info(f"Openai LLM answer retrieved.")
        if question_type == "intent":
            return answer
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        self.save_chat_history()
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
    
    def select_relevant_csv_file(self, file_descriptions: list, user_question: str) -> str:
        """
        Selects the most relevant CSV file from a list based on the user's question.

        Args:
            file_list (list): A list of tuples, where each tuple contains a file name and its description.
            user_question (str): The question asked by the user.

        Returns:
            str: The filename of the most relevant CSV file, or "No relevant file found" if none match.
        """
        system_message: str = SYSTEM_PROMPT_CSV_SELECTION
        prompt_problem: str = CONTEXT_PROMPT_CSV_SELECTION.format(file_descriptions=file_descriptions, user_question=user_question)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_problem},
        ]
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content.strip()
