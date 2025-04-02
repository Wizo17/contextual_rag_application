import sys
import os
import psutil

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.services.llm_session import LLMSession
from src.services.indexer import Indexer
from src.services.indexer2 import Indexer2
from src.config.config import LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL
from src.utils.logger import setup_mlflow
from src.utils.logger import logger

setup_mlflow()

def main():
    logger.info("Contextual RAG Example")
    logger.info(f"Logical cores : {psutil.cpu_count(logical=True)}")
    logger.info(f"Physical cores : {psutil.cpu_count(logical=False)}")
    logger.info(f"CPU Usage : {psutil.cpu_percent()}%")

    llm_session = LLMSession(LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL)
    # indexer = Indexer()
    indexer = Indexer2()

    indexer.load_index()
    indexer.load_chunks()

    queries = [
        "Que contient les recours des Drs Thierry et Jean-Bernard S ?",
    ]

    for query in queries:
        doc_res = indexer.query_index(query)
        # logger.info(doc_res)

        answer = llm_session.get_response_from_documents(query, doc_res)
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {answer}\n\n")
        

if __name__ == "__main__":
    main()

