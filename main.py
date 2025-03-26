import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.services.llm_session import LLMSession
from src.services.indexer import Indexer
from src.config.config import LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL

def main():
    print("Contextual RAG Example")

    llm_session = LLMSession(LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL)
    indexer = Indexer()
    # indexer.process_docs()
    # indexer.build_index()
    # indexer.save_chunks()

    indexer.load_index()
    indexer.load_chunks()

    # query = "Où exercait le Dr Rémi R dans la décision du 18 décembre 2015?"
    query = "Quels sont les faits qui sont reprochés à Dr Rémi R décision du 18 décembre 2015?"

    doc_res = indexer.query_index(query)
    # print(doc_res)

    answer = llm_session.get_response_from_documents(query, doc_res)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
        

if __name__ == "__main__":
    main()

