import os
from openai import OpenAI
from dotenv import load_dotenv
from src.logging_config import logger

load_dotenv()

class OpenaiLLM:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """
        Initializes the LLM class with the OpenAI API.

        Args:
            model_name (str): Name of the model to use. Default is "gpt-4o-mini".
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
    
    def get_template_ida_answering(self, context: str, question: str):
        system = (
            "You are a helpful AI assistant that provides data-driven business insights and strategic "
            "recommendations. Your responses should be concise, actionable, and aligned with the given business context. "
            "Prioritize retention strategies, proactive outreach, and customer engagement improvements based on available data. "
            "When answering questions, ensure that responses stay relevant to the given business context and avoid speculative or unrelated topics."
        )

        prompt = (
            f"Given the following business insights and strategic direction, answer the user's question. "
            f"{context}\n\nUser Question: {question}"
        )

        return system, prompt
    
    def get_template_sales_answering(self, context: str, question: str):
        system = (
            "You are an AI assistant specializing in sales strategy and customer engagement. Your responses should provide "
            "practical sales techniques, lead conversion insights, and value-based selling approaches. Ensure that answers align "
            "with best practices in sales, customer retention, and revenue growth."
        )

        prompt = (
            f"Given the following sales insights, provide a strategic response to the user's question. "
            f"{context}\n\nUser Question: {question}"
        )

        return system, prompt
    
    def get_template_company_answering(self, context: str, question: str):
        system = (
            "You are an AI assistant providing company insights, mission alignment, and strategic recommendations. "
            "Ensure responses reflect the company's core values, vision, and operational strengths while addressing user queries."
        )

        prompt = (
            f"Given the following company background and strategic direction, provide an insightful answer to the user's question. "
            f"{context}\n\nUser Question: {question}"
        )

        return system, prompt
    
    def generate_answer(self, type: str, question: str, question_type: str, context: str = "", max_tokens: int = 100) -> str:
        if type == "ida":
            system, prompt = self.get_template_ida_answering(context=context, question=question)
        elif type == "sales":
            system, prompt = self.get_template_sales_answering(context=context, question=question)
        elif type == "company":
            system, prompt = self.get_template_company_answering(context=context, question=question)
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
    
    def generate_plot_creation_code(self, user_question: str, df, df_description) -> str:
        system_message = (
            "Act as a data scientist and Python programmer. Write code that will solve my problem.\n"
            f"I have a table with {len(df)} rows and {len(df.columns)} columns.\n"
            f"The description of the table and columns is as follows: {df_description}.\n"
            "The columns and their types are as follows:\n"
        )
        for col in df.columns:
            col_type = df.dtypes[col]
            system_message += f"{col} ({col_type})\n"

        prompt_problem = (
            "Solve the following problem:\n"
            "Create a plot in Python with matplotlib package and save it as image named img.png at path ../graphs that fulfills this request:\n"
            f"{user_question}\n\n"
            "While writing the code, please follow these guidelines:\n"
            "1. The answer must be without explanations or comments.\n"
            "2. Do not import additional libraries.\n"
            "3. The table is stored in the variable df.\n"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_problem},
        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )

        return chat_completion.choices[0].message.content.strip()