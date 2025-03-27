# ai-agent

This project implements a Retrieval-Augmented Generation (RAG) system using different LLM models such as Bedrock and OpenAI to interact with various data formats (e.g., CSV, DOCX). The system is built with FastAPI for easy interaction with the models and other components. The project also includes endpoints for answering questions, generating plot creation code, and scheduling actions.

## Project Structure

```plaintext
project_root/
├── data/                       # Contains all data used for answering
├── src/
│   ├── llm/                
│   │   ├── base_llm.py       # Base class for LLM interactions
│   │   ├── bedrock_llm.py     # Interacts with Bedrock LLM
│   │   ├── openai_llm.py      # Interacts with OpenAI LLM
│   ├── context/            
│   │   ├── context_loader.py  # Loads and processes context data for models
│   ├── config/             
│   │   ├── logging_config.py  # Configures logging settings for the project
│   ├── prompts/            
│   │   ├── prompts.py        # Defines LLM prompt templates
│   ├── graph/           
│   │   ├── graph_generator.py # Handles graph generation and processing
│   ├── api/                
│   │   ├── models.py         # Defines input request models for FastAPI
│   │   ├── constants.py      # Stores constants such as file paths
├── app.py                   # FastAPI app entry point
├── create_ida.py            # Python script for converting txt files from data/general_answering_data into insight-direction-action format. Result files are saved in data/insight_direction_action_data
├── requirements.txt         # List of dependencies for the project
├── .env                     # Environment variables for the project
├── README.md                # Project documentation
```

## Endpoints

### `POST /general_answering`

This endpoint generates answers based on user input using the Bedrock LLM.

#### Request body:
```json
{
  "question": "What are sales",
  "collections_names": [
    "sales",
    "company",
    "ida"
  ],
  "llm_type": "openai"
}
```

- **llm_type**: Which llm to use (openai, llama).
- **collections_names**: Which files to use for answering (sales => sales.txt, company => company.txt, ida => all files from data/insight_direction_action_data).
- **question**: There are 3 types of question which are classified using llm: 
    * First is general which will load context data (files selected in collection_names) and use selected llm to answer. 
    * Second type is graph creation which will use llm to generate python code for graph creation, based on user request and in that case image is returned. 
    * Third is action creation but this is only hardcoded on some string return message.

## Installation and Running

### 1. Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/StefanMicic/ai-agent.git
```

### 2. Install Dependencies

Ensure you have Python 3.7 or higher installed. Create a virtual environment and install the required dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set env variables

- Create .env file
- Insert these variables:
    * OPENAI_API_KEY=
    * AWS_ACCESS_KEY_ID=
    * AWS_SECRET_ACCESS_KEY=
    * AWS_REGION=

### 4. Running the Application

To run the application locally with FastAPI:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8000/docs`.

## Running Tests

To run the unit tests for the project, use `pytest`:

```bash
PYTHONPATH=. pytest tests/
```

This will run all the tests in the `tests` directory. Ensure that `pytest` is installed (`pip install -r requirements.txt`).

---
