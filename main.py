import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.embedding.embedder import Embedder
from src.services.llm_session import LLMSession
from src.config.config import CHUNK_SIZE, LLM_CONTEXTUAL_MODEL, LLM_CONTEXTUAL_PROVIDER, OVERLAP_SIZE, INDEX_PATH, LEXICAL_STORE_PATH
from src.preprocessing.document_processor import load_documents
from src.preprocessing.chunk_processor import chunk_text_gpt2
from src.retrieval.faiss_vector_store import FaissVectorStore
from src.retrieval.bm25_lexical_store import BM25LexicalStore

def main():
    print("C/RAG Example")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "raw")
    documents = load_documents(data_path, limit=1)

    context_llm_session = LLMSession(LLM_CONTEXTUAL_PROVIDER, LLM_CONTEXTUAL_MODEL)
    embedder = Embedder()

    processed_docs = []
    embedding_size = 0 # 1536 for text-embedding-3-small

    # Process document
    for doc in documents:
        content = doc["content"]
        chunks = chunk_text_gpt2(content, CHUNK_SIZE, OVERLAP_SIZE)
        context_chunk_list = []
        embedding_list = []

        for chunk in chunks:
            print("################################ PROCESS CHUNCK ################################")
            context = context_llm_session.get_context(chunk, content)
            context_chunk_list.append(context)

            emb = embedder.get_embedding(context, "openai")
            embedding_list.append(emb)

            embedding_size = emb.shape[0]
        
        processed_docs.append({
            "file_path": doc["file_path"],
            "document_id": os.path.basename(doc["file_path"]),
            "content": doc["content"],
            "chunks": chunks,
            "context_chunks": context_chunk_list,
            "embedding": embedding_list,
            "embedding_size": embedding_size
        })

    # Index building
    vector_store = FaissVectorStore(embedding_size, INDEX_PATH, False)
    for doc in processed_docs:
        for emb in doc['embedding']:
            vector_store.add_elements(emb)
    vector_store.save_index()
    
    lexical_store = BM25LexicalStore(preload=False)
    lexical_store.add_documents(processed_docs)
    lexical_store.save_store()


        

if __name__ == "__main__":
    main()

