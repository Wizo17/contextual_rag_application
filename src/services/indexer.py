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
    """
    Indexer class for processing documents, generating embeddings, and managing vector and lexical stores.

    This class is responsible for loading documents from a specified directory, chunking the text,
    generating contextual embeddings using a language model, and storing the embeddings in both
    vector and lexical stores. It also handles the reranking of search results based on relevance.

    Attributes:
        context_llm_session (LLMSession): Session for interacting with the language model.
        embedder (Embedder): Object for generating embeddings from text.
        reranker (Reranker): Object for reranking search results based on relevance.
        vector_store (FaissVectorStore): Store for managing vector embeddings.
        lexical_store (BM25LexicalStore): Store for managing lexical search.
        embedding_size (int): Dimensionality of the embeddings generated.
        global_chunks_list (list): List to hold all processed text chunks.
        global_context_chunks_list (list): List to hold contextual chunks.
        global_embedding_list (list): List to hold generated embeddings.
        global_store_docs (list): List to hold documents for storage.
    """

    def __init__(self):
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
        """
        Process documents from the specified directory, chunk the text, generate embeddings,
        and store the results in global lists for further processing.

        This method loads documents, processes each document by chunking its content, 
        generating contextual embeddings using a language model, and storing the embeddings 
        along with the document metadata in global lists. It also ensures that the embeddings 
        conform to the expected dimensionality.

        Args:
            limit (int, optional): The maximum number of documents to process. 
                                   If not specified, defaults to DOCUMENT_LIMIT.

        Returns:
            bool: True if the documents were processed successfully, False otherwise.
        """
        
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
        """Build the vector and lexical indexes from the processed documents.

        This method constructs the vector index using FAISS and the lexical index using BM25.
        It adds the embeddings and documents to their respective stores and saves the indexes
        to disk. If the update parameter is set to True, it will update the existing indexes
        instead of creating new ones.

        Args:
            update (bool, optional): If True, updates the existing indexes. Defaults to False.

        Returns:
            bool: True if the indexes were built successfully, False otherwise.
        """
        
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
        """Load the vector and lexical indexes from disk.

        This method initializes the vector store and lexical store by loading the previously saved
        indexes from their respective file paths. It ensures that the application can retrieve
        embeddings and documents efficiently for subsequent queries.

        Returns:
            bool: True if the indexes were loaded successfully, False otherwise.
        """
        
        try:
            self.vector_store = FaissVectorStore(EMBEDDING_DIM, INDEX_PATH, True)
            self.lexical_store = BM25LexicalStore(preload=True)
            logger.info(f"Index loading successful")

            return True
        except Exception as e:
            logger.error(f"An error occured during loading index: {e}")
            return False

    def query_index(self, query: str, keeps_double_entries: bool = False):
        """Query the vector and lexical indexes to retrieve relevant chunks based on the input query.

        This method takes a user query, retrieves the corresponding embeddings from the vector store,
        and searches the lexical store for relevant documents. It combines the results from both stores,
        optionally removing duplicate entries, and reranks the final results using a reranker model.

        Args:
            query (str): The search query to be processed.
            keeps_double_entries (bool, optional): If True, retains duplicate entries in the results.
                Defaults to False.

        Returns:
            list: A list of the top-ranked chunks relevant to the query. Returns an empty list if an error occurs.
        
        Raises:
            ValueError: If the query is an empty string.
        """
        
        if query.strip() == "":
            raise ValueError("Query can't be empty.")
        
        try:
            # Querying vector store
            emb_q1 = self.embedder.get_embedding(query, EMBEDDING_PROVIDER)
            indices, distances = self.vector_store.search(emb_q1, RETRIEVAL_TOP_K)

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
        """Save the current state of the chunks to disk.

        This method serializes the global chunks list and the global context chunks list
        to their respective JSON files. It ensures that the current state of the chunks can
        be persisted and reloaded later.

        Returns:
            bool: True if the chunks were successfully saved, False otherwise.

        Raises:
            Exception: If an error occurs while saving the chunks to disk.
        """
        
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
        """Load the current state of the chunks from disk.

        This method reads the serialized chunks from their respective JSON files. 
        If the files exist, it loads the chunks into the global lists for further processing. 
        It ensures that the current state of the chunks can be restored after a restart.

        Returns:
            bool: True if the chunks were successfully loaded, False otherwise.

        Raises:
            Exception: If an error occurs while loading the chunks from disk.
        """
        
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
