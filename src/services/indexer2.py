import os
import json
from uuid import uuid4
from langchain_core.documents import Document
from config.config import LLM_CONTEXTUAL_MODEL, LLM_CONTEXTUAL_PROVIDER, DOCUMENT_PATH, DOCUMENT_LIMIT, CHUNK_SIZE, OVERLAP_SIZE, INDEX_PATH, EMBEDDING_MODEL, EMBEDDING_PROVIDER, RETRIEVAL_TOP_K, RERANK_TOP_K, CHUNKS_PATH, CONTEXT_CHUNKS_PATH
from services.llm_session import LLMSession
from reranking.reranker import Reranker
from preprocessing.document_processor import load_documents
from preprocessing.chunk_processor import chunk_text_gpt2
from retrieval.faiss_langchain_vector_store import FaissLangchainVectorStore
from retrieval.bm25_lexical_store import BM25LexicalStore
from utils.logger import logger

class Indexer2:
    # TODO Write docstring

    def __init__(self):
        self.context_llm_session = LLMSession(LLM_CONTEXTUAL_PROVIDER, LLM_CONTEXTUAL_MODEL)
        self.reranker = Reranker()

        self.vector_store = None
        self.lexical_store = None

        self.global_chunks_list = []
        self.global_context_chunks_list = []
        self.global_store_docs = []
        self.global_documents = []
        self.global_uuid = []

    def process_docs(self, limit:int = DOCUMENT_LIMIT):
        # TODO Write docstring
        
        logger.info(f"Process docs in {DOCUMENT_PATH}")

        try:
            documents = load_documents(DOCUMENT_PATH, limit=limit)

            for doc in documents:
                logger.info(f"Process doc: {doc['file_path']}")
                content = doc["content"]
                chunks = chunk_text_gpt2(content, CHUNK_SIZE, OVERLAP_SIZE)
                count = 0

                for chunk in chunks:
                    count += 1
                    logger.info(f"Process chunk no: {count}")
                    context = self.context_llm_session.get_context(chunk, content)

                    store_element = context # chunk or context (context + chunk)

                    # Use for storing chunk
                    self.global_store_docs.append({
                        "file_path": doc["file_path"],
                        "document_id": os.path.basename(doc["file_path"]),
                        "content": store_element
                    })

                    # Fill global data
                    self.global_chunks_list.append(chunk)
                    self.global_context_chunks_list.append(context)
                    self.global_documents.append(
                        Document(
                            page_content=store_element,
                            metadata={"source": os.path.basename(doc["file_path"])},
                        )
                    )
                    self.global_uuid.append(str(uuid4()))

            return True    
        except Exception as e:
            logger.error(f"An error occured during processing docs: {e}")
            return False

    def build_index(self, update: bool = False):
        # TODO Write docstring
        
        if not self.global_documents or not self.global_store_docs:
            logger.error(f"Error durring building index.")
            return False
        
        try:
            if not update:
                self.vector_store = FaissLangchainVectorStore(EMBEDDING_PROVIDER, EMBEDDING_MODEL, INDEX_PATH, False)
                self.lexical_store = BM25LexicalStore(preload=False)

            logger.info(f"Building vector index with Faiss")
            self.vector_store.add_elements(self.global_documents, self.global_uuid)
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
            self.vector_store = FaissLangchainVectorStore(EMBEDDING_PROVIDER, EMBEDDING_MODEL, INDEX_PATH, True)
            self.lexical_store = BM25LexicalStore(preload=True)
            logger.info(f"Index loading successful")

            return True
        except Exception as e:
            logger.error(f"An error occured during loading index: {e}")
            return False

    def query_index(self, query: str, keeps_double_entries: bool = False):
        # TODO Write docstring
        
        if query.strip() == "":
            raise ValueError("Query can't be empty.")
        
        try:
            # Querying vector store
            result = self.vector_store.search(query, RETRIEVAL_TOP_K)
            # {res.page_content} {res.metadata}

            # Retrieve chunk
            retrieved_chunks = []
            for res in result:
                retrieved_chunks.append(res.page_content)

            # Retrieve lexical content
            lexical_result = self.lexical_store.search(query, RETRIEVAL_TOP_K)
            retrieved_lex = []
            for item in lexical_result:
                retrieved_lex.append(item["content"])

            # Build final chunk
            if keeps_double_entries:
                corpus_list = retrieved_chunks + retrieved_lex
            else:
                corpus_list = list(set(retrieved_chunks + retrieved_lex))

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
