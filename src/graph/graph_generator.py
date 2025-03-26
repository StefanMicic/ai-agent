import pandas as pd
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from src.llm.openai_llm import OpenaiLLM
from src.llm.bedrock_llm import BedrockLlamaLLM
from src.config.logging_config import logger

class GraphGenerator:
    def __init__(self, csv_file: str, description_file: str, llm_type: str, retry_limit: int = 3) -> None:
        """
        Initializes the GraphGenerator class for generating plots based on user queries.

        Args:
            csv_file (str): Path to the CSV file containing the data.
            description_file (str): Path to the file containing the description of the data.
            llm_type (str): Type of LLM to use, either "openai" or "llama".
            retry_limit (int, optional): Number of retries for generating the plot. Default is 3.
        """
        self.csv_file: str = csv_file
        self.description_file: str = description_file
        self.retry_limit: int = retry_limit
        self.df: pd.DataFrame = None
        self.df_description: str = None
        self.load_data()
        self.load_description()
        self.code_block: Any = None
        self.code_blocks: list = []
        
        if llm_type == "openai":
            self.llm = OpenaiLLM()
        elif llm_type == "llama":
            self.llm = BedrockLlamaLLM()

    def load_data(self) -> None:
        """
        Loads the sales data from the specified CSV file.

        Raises:
            Exception: If there is an error loading the data from the CSV file.
        """
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Data loaded successfully from {self.csv_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def load_description(self) -> None:
        """
        Loads the description from the specified text file.

        Raises:
            Exception: If there is an error loading the description from the file.
        """
        try:
            with open(self.description_file, 'r') as file:
                self.df_description = file.read()
            logger.info(f"Description loaded successfully from {self.description_file}")
        except Exception as e:
            logger.error(f"Error loading description: {e}")

    def generate_plot(self, plot_question: str) -> str:
        """
        Generate a plot based on the user query.

        Args:
            plot_question (str): The question related to the plot to be generated.

        Returns:
            str: A message indicating whether the plot was generated successfully or an error occurred.
        """
        generated_code = self.llm.generate_plot_creation_code(plot_question, self.df, self.df_description)
        if generated_code:
            try:
                tmp_code: str = ""
                for line in generated_code.split("\n"):
                    if not line.startswith("```"):
                        tmp_code += line.strip() + "\n"
                tmp_code = tmp_code.replace("python", "").strip()
                exec(tmp_code, {"df": self.df, "pd": pd, "np": np, "plt": plt})
                return "Plot generated successfully."
            except Exception as e:
                logger.error(f"Error generating plot: {e}")
                return f"Error generating plot: {e}"
        return "No valid code generated for plot."
