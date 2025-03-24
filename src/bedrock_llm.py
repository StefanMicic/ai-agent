import os
import boto3
from dotenv import load_dotenv
import json
from src.logging_config import logger

load_dotenv()

class BedrockLlamaLLM:
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
        self.model_name = model_name
    
    def get_template_ida_answering(self, context: str, question: str):
        system = (
            "You are a helpful AI assistant that provides data-driven business insights and strategic "
            "recommendations. Your responses should be concise, actionable, and aligned with the given business context. "
            "Prioritize retention strategies, proactive outreach, and customer engagement improvements based on available data. "
            "When answering questions, ensure that responses stay relevant to the given business context and avoid speculative or unrelated topics."
        )
        return f"{system}\n\nGiven the following context, answer the user's question:\n{context}\n\nUser Question: {question}"

    def get_template_sales_answering(self, context: str, question: str):
        system = (
            "You are an AI assistant specializing in sales strategy and customer engagement. Your responses should provide "
            "practical sales techniques, lead conversion insights, and value-based selling approaches. Ensure that answers align "
            "with best practices in sales, customer retention, and revenue growth."
        )
        return f"{system}\n\nGiven the following context, answer the user's question:\n{context}\n\nUser Question: {question}"
    
    def get_template_company_answering(self, context: str, question: str):
        system = (
            "You are an AI assistant providing company insights, mission alignment, and strategic recommendations. "
            "Ensure responses reflect the company's core values, vision, and operational strengths while addressing user queries."
        )
        return f"{system}\n\nGiven the following context, answer the user's question:\n{context}\n\nUser Question: {question}"
    
    def generate_answer(self, type: str, question: str, question_type: str, context: str = "", max_tokens: int = 100) -> str:
        if type == "ida":
            prompt = self.get_template_ida_answering(context, question)
        elif type == "sales":
            prompt = self.get_template_sales_answering(context, question)
        elif type == "company":
            prompt = self.get_template_company_answering(context, question)
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