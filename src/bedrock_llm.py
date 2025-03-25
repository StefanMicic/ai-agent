import os
import boto3
from dotenv import load_dotenv
import json
from src.base_llm import BaseLLM
from src.logging_config import logger
from src.prompts import SYSTEM_PROMPT, CONTEXT_PROMPT, SYSTEM_PROMPT_PLOT, CONTEXT_PROMPT_PLOT, SYSTEM_PROMPT_INTENT, CONTEXT_PROMPT_INTENT
import pandas as pd

load_dotenv()

class BedrockLlamaLLM(BaseLLM):
    def __init__(self, model_name: str = "meta.llama3-8b-instruct-v1:0") -> None:
        """
        Initializes the LLM class with AWS Bedrock using credentials from environment variables.

        Args:
            model_name (str): Name of the Bedrock model to use. Default is "meta.llama3-8b-instruct-v1:0".
        """
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
        )
        self.model_name: str = model_name
    
    def generate_answer(self, question: str, question_type: str, context: str = "", max_tokens: int = 100) -> str:
        """
        Generates an answer to a given question based on the question type using the AWS Bedrock Llama model.

        Args:
            question (str): The question for which the answer is to be generated.
            question_type (str): The type of the question, which determines the prompt format.
            context (str, optional): The context to be used in the prompt. Default is an empty string.
            max_tokens (int, optional): The maximum number of tokens for the generated answer. Default is 100.

        Returns:
            str: The generated answer.
        """
        if question_type == "general":
            prompt = SYSTEM_PROMPT + "\n\n" + CONTEXT_PROMPT.format(context=context, question=question)
        elif question_type == "intent":
            prompt = SYSTEM_PROMPT_INTENT + "\n\n" + CONTEXT_PROMPT_INTENT.format(question=question)
        else:
            return "Could not provide answer."
        
        formatted_prompt = f"""
            <|begin_of_text|><|start_header_id|>user<|end_header_id|>
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": max_tokens
        }

        request = json.dumps(native_request)
        response = self.client.invoke_model(modelId=self.model_name, body=request)
        model_response = json.loads(response["body"].read())

        response_text = model_response.get("generation", "No generation returned.")
        logger.info("Bedrock Llama answer retrieved.")
        return response_text
    
    def generate_plot_creation_code(self, user_question: str, df: pd.DataFrame, df_description: str) -> str:
        """
        Generates code for creating a plot based on the user's question and the DataFrame description using the AWS Bedrock Llama model.

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

        formatted_prompt = f"""
            <|begin_of_text|><|start_header_id|>user<|end_header_id|>
            {system_message}\n\n{prompt_problem}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
        native_request = {
            "prompt": formatted_prompt
        }

        request = json.dumps(native_request)

        response = self.client.invoke_model(modelId=self.model_name, body=request)
        model_response = json.loads(response["body"].read())

        response_text = model_response.get("generation", "No generation returned.")
        logger.info("Bedrock Llama answer retrieved.")
        return response_text
