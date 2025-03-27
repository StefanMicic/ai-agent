import os
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.llm.bedrock_llm import BedrockLlamaLLM

@pytest.fixture
def llm():
    with patch("boto3.client") as mock_client:
        mock_client.return_value.invoke_model = MagicMock()
        mock_client.return_value.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps({"generation": "Test Answer"}))
        }
        return BedrockLlamaLLM()
    
def test_load_chat_history(llm):
    history = llm.load_chat_history()
    assert isinstance(history, list)

@patch("boto3.client")
def test_generate_answer(mock_boto_client, llm):
    mock_boto_client.return_value.invoke_model.return_value = {
        "body": MagicMock(read=lambda: json.dumps({"generation": "Test Answer"}))
    }
    response = llm.generate_answer("What is AI?", "general")
    assert response == "Test Answer" 
