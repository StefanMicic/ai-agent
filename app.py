from src.logging_config import logger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from src.context_loader import ContextLoader
from src.logging_config import logger
from src.openai_llm import OpenaiLLM
from src.bedrock_llm import BedrockLlamaLLM
from src.graph_generator import GraphGenerator

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

general_answering_data_directory = "./data/general_answering_data"
ida_data_directory = "./data/insight_direction_action_data"
df_path = "./data/graph_data/sales.csv"
df_description_path = "./data/graph_data/sales_csv_description.txt"

context_loader = ContextLoader(general_answering_data_directory=general_answering_data_directory, ida_data_directory=ida_data_directory)
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

        if not question:
            return {
                'answer': "Please provide question!"
            }
        if not collections_names:
            return {
                'answer': "Please provide collections_names!"
            } 
        if not llm_type:
            return {
                'answer': "Please provide llm_type!"
            }

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
            graph_generator = GraphGenerator(csv_file=df_path, description_file=df_description_path, llm_type=llm_type)
            graph_generator.generate_plot(plot_question=question)
            return FileResponse("./graphs/img.png", media_type="image/png")
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

