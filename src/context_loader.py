import os
from typing import List
from src.logging_config import logger

class ContextLoader:
    def __init__(self, general_answering_data_directory: str, ida_data_directory: str) -> None:
        """
        Initialize the context loader with directories containing categorized files.

        Args:
            general_answering_data_directory (str): Path to the folder containing files for general answering.
            ida_data_directory (str): Path to the folder containing files for insight, direction, action answering.
        """
        self.general_answering_data_directory: str = general_answering_data_directory
        self.ida_data_directory: str = ida_data_directory

    def _load_txt_files_content(self, file_paths: List[str]) -> str:
        """
        Load and concatenate content from the given text files.

        Args:
            file_paths (List[str]): List of file paths to be read.

        Returns:
            str: Combined content of all the files.
        """
        content: List[str] = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                content.append(file.read())
        return "\n".join(content)

    def get_context(self, file_names: List[str]) -> str:
        """
        Retrieve the context based on the specified context type.

        Args:
            file_names (List[str]): List of file names to be loaded (without the '.txt' extension).

        Returns:
            str: The combined context from the specified files.

        Raises:
            ValueError: If an unsupported context type is provided.
        """
        file_paths = []
        for fn in file_names:
            if fn == "ida":
                file_paths.extend([self.ida_data_directory + "/" + ida_fn for ida_fn in os.listdir(self.ida_data_directory)])
            else:
                file_paths.append(self.general_answering_data_directory + "/" + fn + ".txt")
        context: str = self._load_txt_files_content(file_paths)
        return context
