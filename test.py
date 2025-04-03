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
        "Cas de médecin généraliste condamné",
        "Liste de cas avec ophtalmologue",
        "Dans le Dossier n° 5301, quels arguments le Dr A a-t-il soulevés pour contester les sanctions qui lui ont été infligées du 17 avril 2018 ?",
        "Dans le Dossier n° 5301, quel était le motif du rejet de l'appel de la caisse ?"
        "Que contient la plainte du 29 novembre 2016 avec M. B contre Dr A ?",
    ]

    for query in queries:
        doc_res = indexer.query_index(query)
        # logger.info(doc_res)

        answer = llm_session.get_response_from_documents(query, doc_res)
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {answer}\n\n")
        

if __name__ == "__main__":
    main()

