from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from src.context.context_loader import ContextLoader
from src.config.logging_config import logger
from src.llm.openai_llm import OpenaiLLM
from src.llm.bedrock_llm import BedrockLlamaLLM
from src.graph.graph_generator import GraphGenerator
from src.api.constants import GRAPH_IMAGE_PATH, GENERAL_ANSWERING_DATA_DIR, IDA_DATA_DIR, GRAPH_DATA_DIR

load_dotenv()

app = FastAPI()

class GeneralAnsweringRequest(BaseModel):
    question: str
    collections_names: List[str] = ["sales", "company"]
    llm_type: str = "openai"

class IdaAnsweringRequest(BaseModel):
    question: str
    ida_file_name: str
    llm_type: str = "openai"

context_loader = ContextLoader(general_answering_data_directory=GENERAL_ANSWERING_DATA_DIR, ida_data_directory=IDA_DATA_DIR, graph_data_directory=GRAPH_DATA_DIR)
logger.info("ContextLoader initialized.")

openai_llm = OpenaiLLM()
logger.info("OpenaiLLM initialized.")

bedrock_llama_llm = BedrockLlamaLLM()
logger.info("BedrockLlamaLLM initialized.")

@app.post("/general_answering")
async def general_answering(request: GeneralAnsweringRequest):
    try:
        logger.info("general_answering request received.")
        
        question = request.question
        logger.info(f"Received question: {question[:200]}...")
        collections_names = request.collections_names
        logger.info(f"Received collections_names: {collections_names}...")
        llm_type = request.llm_type
        logger.info(f"Received llm_type: {llm_type}...")

        if llm_type == "openai":
            llm = openai_llm
        elif llm_type == "llama":
            llm = bedrock_llama_llm
        else:
            return {
                'answer': "Please provide valid llm_type!"
            }
        
        logger.info("Understanding intent...")
        intent = llm.generate_answer(question=question, question_type="intent", context="").strip()
        logger.info(f"Found intent: {intent}")

        if intent == "1":
            context = context_loader.get_context(file_names=collections_names)
            if not context:
                return {
                    'answer': "Please provide valid collections_names!"
                }
            answer = llm.generate_answer(question=question, question_type="general", context=context)
            logger.info(f"Answer generated: {answer[:100]}...")
            return {
                'answer': answer
            }
        elif intent == "2":
            file_descriptions = context_loader.get_graph_context()
            found_filename = llm.select_relevant_csv_file(file_descriptions=file_descriptions, user_question=question)
            logger.info(f"Found graph filename: {found_filename}")
            csv_file = GRAPH_DATA_DIR + "/" + found_filename.split("_")[0] + ".csv"
            description_file = GRAPH_DATA_DIR + "/" + found_filename
            graph_generator = GraphGenerator(csv_file=csv_file, description_file=description_file, llm_type=llm_type)
            graph_generator.generate_plot(plot_question=question)
            return FileResponse(GRAPH_IMAGE_PATH, media_type="image/png")
        elif intent == "3":
            return {
                'answer': "I will create task you requested!"
            }
        else:
            raise Exception("Something went wrong, when finding intent.")
    except Exception as e:
        logger.error(f"An error occurred while generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

