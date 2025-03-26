import os
import json
from config.config import LLM_CONTEXTUAL_MODEL, LLM_CONTEXTUAL_PROVIDER, DOCUMENT_PATH, DOCUMENT_LIMIT, CHUNK_SIZE, OVERLAP_SIZE, INDEX_PATH, EMBEDDING_DIM, EMBEDDING_PROVIDER, RETRIEVAL_TOP_K, RERANK_TOP_K, CHUNKS_PATH, CONTEXT_CHUNKS_PATH
from services.llm_session import LLMSession
from embedding.embedder import Embedder
from reranking.reranker import Reranker
from preprocessing.document_processor import load_documents
from preprocessing.chunk_processor import chunk_text_gpt2
from retrieval.faiss_vector_store import FaissVectorStore
from retrieval.bm25_lexical_store import BM25LexicalStore
from utils.logger import logger

class Indexer:
    # TODO Write docstring

    def __init__(self):
        # TODO Write docstring
        self.context_llm_session = LLMSession(LLM_CONTEXTUAL_PROVIDER, LLM_CONTEXTUAL_MODEL)
        self.embedder = Embedder()
        self.reranker = Reranker()

        self.vector_store = None
        self.lexical_store = None

        self.embedding_size = 0

        self.global_chunks_list = []
        self.global_context_chunks_list = []
        self.global_embedding_list = []
        self.global_store_docs = []

    def process_docs(self, limit:int = DOCUMENT_LIMIT):
        # TODO Write docstring
        logger.info(f"Process docs in {DOCUMENT_PATH}")

        try:
            documents = load_documents(DOCUMENT_PATH, limit=limit)

            for doc in documents:
                logger.info(f"Process doc: {doc['file_path']}")
                content = doc["content"]
                chunks = chunk_text_gpt2(content, CHUNK_SIZE, OVERLAP_SIZE)
                context_chunk_list = []
                embedding_list = []
                count = 0

                for chunk in chunks:
                    count += 1
                    logger.info(f"Process chunk no: {count}")
                    context = self.context_llm_session.get_context(chunk, content)
                    context_chunk_list.append(context)

                    emb_result = self.embedder.get_embedding(context, EMBEDDING_PROVIDER)
                    embedding_list.append(emb_result)

                    self.embedding_size = emb_result.shape[0]
                    if self.embedding_size != EMBEDDING_DIM:
                        raise AssertionError(f"Embedding dim: {self.embedding_size} not equals env EMBEDDING_DIM: {EMBEDDING_DIM}.")

                    # Use for storing chunk
                    self.global_store_docs.append({
                        "file_path": doc["file_path"],
                        "document_id": os.path.basename(doc["file_path"]),
                        "content": chunk
                    })

                    # Fill global data
                    self.global_chunks_list.append(chunk)
                    self.global_context_chunks_list.append(context)
                    self.global_embedding_list.append(emb_result)

            return True    
        except Exception as e:
            logger.error(f"An error occured during processing docs: {e}")
            return False

    def build_index(self, update: bool = False):
        # TODO Write docstring
        if not self.global_embedding_list or not self.global_store_docs:
            logger.error(f"Error durring building index.")
            return False
        
        try:
            if not update:
                self.vector_store = FaissVectorStore(EMBEDDING_DIM, INDEX_PATH, False)
                self.lexical_store = BM25LexicalStore(preload=False)

            logger.info(f"Building vector index with Faiss")
            for emb in self.global_embedding_list:
                self.vector_store.add_elements(emb)
            self.vector_store.save_index()
            
            logger.info(f"Building lexical index with BM25")
            self.lexical_store.add_documents(self.global_store_docs)
            self.lexical_store.save_store()

            return True
        except Exception as e:
            logger.error(f"An error occured in building index: {e}")
            return False
        
    def load_index(self):
        # TODO Write docstring
        try:
            self.vector_store = FaissVectorStore(EMBEDDING_DIM, INDEX_PATH, True)
            self.lexical_store = BM25LexicalStore(preload=True)
            logger.info(f"Index loading successful")

            return True
        except Exception as e:
            logger.error(f"An error occured during loading index: {e}")
            return False

    def query_index(self, query: str):
        # TODO Write docstring
        if query.strip() == "":
            raise ValueError("Query can't be empty.")
        
        try:
            # Querying vector store
            emb_q1 = self.embedder.get_embedding(query, EMBEDDING_PROVIDER)
            indices, distances = self.vector_store.search(emb_q1, RETRIEVAL_TOP_K)

            # TODO Change logic
            # Retrieve chunk
            retrieved_chunks = []
            for idx in indices[0]:
                retrieved_chunks.append(self.global_chunks_list[idx])

            # Retrieve lexical content
            lexical_result = self.lexical_store.search(query, RETRIEVAL_TOP_K)
            retrieved_lex = []
            for item in lexical_result:
                retrieved_lex.append(item["content"])

            # Build final chunk
            corpus_list = retrieved_chunks + retrieved_lex

            # Rerank result
            chunk_rank = []
            rank_result = self.reranker.rerank_results(query, corpus_list, RERANK_TOP_K)
            for item in rank_result:
                chunk_rank.append(corpus_list[item["corpus_id"]])

            return chunk_rank
        except Exception as e:
            logger.error(f"An error occured during querying: {e}")
            return []
        
    def save_chunks(self):
        # TODO Write docstring
        try:
            data = {"chunks": self.global_chunks_list}
            with open(CHUNKS_PATH, 'w') as f:
                json.dump(data, f)
                logger.info(f"Base chunk saved {CHUNKS_PATH}")

            data = {"chunks": self.global_context_chunks_list}
            with open(CONTEXT_CHUNKS_PATH, 'w') as f:
                json.dump(data, f)
                logger.info(f"Context chunk saved {CONTEXT_CHUNKS_PATH}")
            
            return True
        except Exception as e:
            logger.error(f"An error has occurred while merging embedding: {e}")
            return False

    def load_chunks(self):
        # TODO Write docstring
        try:
            if os.path.exists(CHUNKS_PATH):
                with open(CHUNKS_PATH, 'r') as f:
                    data = json.load(f)
                    self.global_chunks_list = data["chunks"]
                    logger.info(f"Base chunk loaded from {CONTEXT_CHUNKS_PATH}")

            if os.path.exists(CONTEXT_CHUNKS_PATH):
                with open(CONTEXT_CHUNKS_PATH, 'r') as f:
                    data = json.load(f)
                    self.global_context_chunks_list = data["chunks"]
                    logger.info(f"Context chunk loaded from {CONTEXT_CHUNKS_PATH}")

            return True
        except Exception as e:
            logger.error(f"An error has occurred while merging embedding: {e}")
            return False
