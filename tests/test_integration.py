import pytest
from fastapi.testclient import TestClient
from app import app
from src.context.context_loader import ContextLoader
from src.llm.openai_llm import OpenaiLLM
from src.api.constants import GENERAL_ANSWERING_DATA_DIR

client = TestClient(app)

filename = "sales.txt"
SALES_TEXT_CONTENT = ""
with open(GENERAL_ANSWERING_DATA_DIR + "/" + filename, "r", encoding="utf-8") as file:
    SALES_TEXT_CONTENT = filename + "\n\n" + file.read()

@pytest.fixture
def context_loader():
    """Fixture to provide a real ContextLoader instance pointing to test directories."""
    return ContextLoader(
        general_answering_data_directory=GENERAL_ANSWERING_DATA_DIR,
        ida_data_directory="",
        graph_data_directory=""
    )

@pytest.fixture
def openai_llm():
    """Fixture to provide a real OpenaiLLM instance"""
    return OpenaiLLM()

def test_general_answering_with_valid_question(context_loader, openai_llm):
    """Tests if the LLM correctly returns intent=1, and context loader loads only `sales.txt`."""
    request_data = {
        "question": "What are sales?",
        "collections_names": ["sales"],
        "llm_type": "openai"
    }

    question = "What are the sales trends for the last quarter?"

    intent = openai_llm.generate_answer(question=question, question_type="intent", context="").strip()
    assert intent == "1"
    
    loaded_context = context_loader.get_context(["sales"])
    assert loaded_context.strip() == SALES_TEXT_CONTENT

    response = client.post("/general_answering", json=request_data)
    assert response.status_code == 200 
