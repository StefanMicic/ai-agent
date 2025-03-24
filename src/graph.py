import pandas as pd
import numpy as np
from src.logging_config import logger
from typing import Any
import matplotlib.pyplot as plt
from src.openai_llm import OpenaiLLM
from src.bedrock_llm import BedrockLlamaLLM

class GraphGenerator:
    def __init__(self, csv_file: str, description_file: str, llm_type: str, retry_limit: int = 3):
        self.csv_file = csv_file
        self.description_file = description_file
        self.retry_limit = retry_limit
        self.df = None
        self.df_description = None
        self.load_data()
        self.load_description()
        self.code_block = None
        self.code_blocks = []
        if llm_type == "openai":
            self.llm = OpenaiLLM()
        elif llm_type == "llama":
            self.llm = BedrockLlamaLLM()

    def load_data(self):
        """Loads sales data from the CSV file."""
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Data loaded successfully from {self.csv_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def load_description(self):
        """Loads the description from a text file."""
        try:
            with open(self.description_file, 'r') as file:
                self.df_description = file.read()
            logger.info(f"Description loaded successfully from {self.description_file}")
        except Exception as e:
            logger.error(f"Error loading description: {e}")

    def generate_plot(self, plot_question: str) -> Any:
        """Generate a graph plot based on the user query."""
        generated_code = self.llm.generate_plot_creation_code(plot_question, self.df, self.description_file)
        if generated_code:
            try:
                tmp_code = ""
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
