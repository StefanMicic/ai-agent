from typing import List
from pydantic import BaseModel

class GeneralAnsweringRequest(BaseModel):
    question: str
    collections_names: List[str] = ["sales", "company"]
    llm_type: str = "openai"

class IdaAnsweringRequest(BaseModel):
    question: str
    ida_file_name: str
    llm_type: str = "openai"
