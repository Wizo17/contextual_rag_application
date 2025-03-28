import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.services.llm_session import LLMSession
from src.services.indexer import Indexer
from src.config.config import LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL
from src.utils.logger import setup_mlflow

setup_mlflow()

def main():
    print("Contextual RAG Example")

    llm_session = LLMSession(LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL)
    indexer = Indexer()

    indexer.load_index()
    indexer.load_chunks()

    queries = [
        "Que contient la proposition de loi contre les fraudes aux moyens de paiement scripturaux du 19 mars 2025 ?",
        "Fournis moi le contenu de l'article 3 de la loi contre les fraudes aux moyens de paiement scripturaux du 19 mars 2025 ?",
        "Fournis moi un résumé de la proposition de loi pour réformer l'accueil des gens du voyage du 27 mars 2025.",
        "Une question complètement aléatoire",
    ]

    for query in queries:
        doc_res = indexer.query_index(query)
        # print(doc_res)

        answer = llm_session.get_response_from_documents(query, doc_res)
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        

if __name__ == "__main__":
    main()

