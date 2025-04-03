import os
import shutil
import json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langchain_core.documents import Document
from config.config import LLM_CONTEXTUAL_MODEL, LLM_CONTEXTUAL_PROVIDER, DOCUMENT_PATH_INPUT, DOCUMENT_LIMIT, CHUNK_SIZE, OVERLAP_SIZE, INDEX_PATH, EMBEDDING_MODEL, EMBEDDING_PROVIDER, RETRIEVAL_TOP_K, RERANK_TOP_K, CHUNKS_PATH, CONTEXT_CHUNKS_PATH, DOCUMENT_CHUNKS_PATH, UUIDS_CHUNKS_PATH, DOCUMENT_PATH_OUTPUT, PROCESSING_DOC_MAX_WORKERS
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

        self.lock = threading.Lock() # Use lock for concurrency

    def process_docs(self, limit:int = DOCUMENT_LIMIT):
        # TODO Write docstring
        # TODO Use process_single_doc
        
        logger.info(f"Process docs in {DOCUMENT_PATH_INPUT}")

        try:
            documents = load_documents(DOCUMENT_PATH_INPUT, limit=limit)

            for doc in documents:
                logger.info(f"Process doc: {doc['file_path']}")
                content = doc["content"]
                chunks = chunk_text_gpt2(content, CHUNK_SIZE, OVERLAP_SIZE)
                count = 0
                as_error = False
                sub_store_docs_list = []
                sub_chunks_list = []
                sub_context_chunks_list = []
                sub_document_list = []
                sub_uuid_list = []

                for chunk in chunks:
                    count += 1
                    logger.info(f"Process chunk no: {count}")
                    context = self.context_llm_session.get_context(chunk, content)
                    if context is None:
                        as_error = True
                        break

                    store_element = f"CONTEXT:\n{context}\nCHUNK:\n{chunk}" # chunk or context (context + chunk)
                    # logger.debug(f"\nCONTEXT:\n{store_element}")
                    # logger.debug(f"\nCHUNK:\n{chunk}")

                    # Use for storing chunk
                    sub_store_docs_list.append({
                        "file_path": doc["file_path"],
                        "document_id": os.path.basename(doc["file_path"]),
                        "content": store_element
                    })

                    sub_chunks_list.append(chunk)
                    sub_context_chunks_list.append(context)
                    sub_document_list.append(
                        Document(
                            page_content=store_element,
                            metadata={"source": os.path.basename(doc["file_path"])},
                        )
                    )
                    sub_uuid_list.append(str(uuid4()))

                if not as_error:
                    # Update global data
                    self.global_store_docs.extend(sub_store_docs_list)
                    self.global_chunks_list.extend(sub_chunks_list)
                    self.global_context_chunks_list.extend(sub_context_chunks_list)
                    self.global_documents.extend(sub_document_list)
                    self.global_uuid.extend(sub_uuid_list)

                    # Move doc
                    os.makedirs(DOCUMENT_PATH_OUTPUT, exist_ok=True)
                    shutil.move(doc["file_path"], DOCUMENT_PATH_OUTPUT)
                    logger.info(f"Document move from {doc["file_path"]} to {DOCUMENT_PATH_OUTPUT}")


            return True    
        except Exception as e:
            logger.error(f"An error occured during processing docs: {e}")
            return False

    def build_index(self, update: bool = False):
        # TODO Write docstring
        
        if not self.global_documents or not self.global_store_docs:
            logger.error(f"Size of documents: {len(self.global_documents)}")
            logger.error(f"Size of store doc: {len(self.global_store_docs)}")
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
            # # Save store documents
            # store_data = [
            #     {"file_path": doc.file_path, "document_id": doc.document_id, "content": doc.content}
            #     for doc in self.global_store_docs
            # ]
            # with open("data/index/doc_store.json", 'w', encoding='utf-8') as f:
            #     json.dump({"documents": store_data}, f, ensure_ascii=False, indent=4)
            #     logger.info(f"Documents saved to {"data/index/doc_store.json"}")

            # Save documents
            documents_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.global_documents
            ]
            with open(DOCUMENT_CHUNKS_PATH, 'w', encoding='utf-8') as f:
                json.dump({"documents": documents_data}, f, ensure_ascii=False, indent=4)
                logger.info(f"Documents saved to {DOCUMENT_CHUNKS_PATH}")
            
            # Save UUIDs
            with open(UUIDS_CHUNKS_PATH, 'w', encoding='utf-8') as f:
                json.dump({"uuids": self.global_uuid}, f, ensure_ascii=False, indent=4)
                logger.info(f"UUIDs saved to {UUIDS_CHUNKS_PATH}")
            
            return True
        except Exception as e:
            logger.error(f"An error occurred while saving data: {e}")
            return False

    def load_chunks(self):
        # TODO Write docstring
        
        try:
            # Load store documents
            # store_data = [
            #     {"file_path": doc.file_path, "document_id": doc.document_id, "content": doc.content}
            #     for doc in self.global_store_docs
            # ]
            # with open("data/index/doc_store.json", 'w', encoding='utf-8') as f:
            #     json.dump({"documents": store_data}, f, ensure_ascii=False, indent=4)
            #     logger.info(f"Documents saved to {"data/index/doc_store.json"}")

            # if os.path.exists("data/index/doc_store.json"):
            #     with open("data/index/doc_store.json", 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            #         self.global_documents = [
            #             Document(page_content=doc["page_content"], metadata=doc["metadata"])
            #             for doc in data.get("documents", [])
            #         ]
            #         logger.info(f"Documents loaded from {DOCUMENT_CHUNKS_PATH}") 

            # Load documents
            if os.path.exists(DOCUMENT_CHUNKS_PATH):
                with open(DOCUMENT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.global_documents = [
                        Document(page_content=doc["page_content"], metadata=doc["metadata"])
                        for doc in data.get("documents", [])
                    ]
                    logger.info(f"Documents loaded from {DOCUMENT_CHUNKS_PATH}")
            
            # Load UUIDs
            if os.path.exists(UUIDS_CHUNKS_PATH):
                with open(UUIDS_CHUNKS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.global_uuid = data.get("uuids", [])
                    logger.info(f"UUIDs loaded from {UUIDS_CHUNKS_PATH}")
            
            return True
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            return False
        
    def process_single_doc(self, doc):
        # TODO Write docstring
        
        logger.info(f"Processing document: {doc['file_path']}")
        content = doc["content"]
        chunks = chunk_text_gpt2(content, CHUNK_SIZE, OVERLAP_SIZE)
        as_error = False
        sub_store_docs_list = []
        sub_chunks_list = []
        sub_context_chunks_list = []
        sub_document_list = []
        sub_uuid_list = []

        for count, chunk in enumerate(chunks, start=1):
            logger.info(f"Processing chunk {count}")
            context = self.context_llm_session.get_context(chunk, content)
            if context is None:
                as_error = True
                break
            store_element = f"CONTEXT:\n{context}\nCHUNK:\n{chunk}"

            # Storage chunk data
            with self.lock:  # Lock writing
                sub_store_docs_list.append({
                    "file_path": doc["file_path"],
                    "document_id": os.path.basename(doc["file_path"]),
                    "content": store_element
                })

                sub_chunks_list.append(chunk)
                sub_context_chunks_list.append(context)
                sub_document_list.append(
                    Document(
                        page_content=store_element,
                        metadata={"source": os.path.basename(doc["file_path"])},
                    )
                )
                sub_uuid_list.append(str(uuid4()))

        if not as_error:
            # Update global data
            self.global_store_docs.extend(sub_store_docs_list)
            self.global_chunks_list.extend(sub_chunks_list)
            self.global_context_chunks_list.extend(sub_context_chunks_list)
            self.global_documents.extend(sub_document_list)
            self.global_uuid.extend(sub_uuid_list)

            # Move doc
            os.makedirs(DOCUMENT_PATH_OUTPUT, exist_ok=True)
            shutil.move(doc["file_path"], DOCUMENT_PATH_OUTPUT)
            logger.info(f"Document move from {doc["file_path"]} to {DOCUMENT_PATH_OUTPUT}")

    def parallel_process_docs(self):
        # TODO Write docstring

        try:
            documents = load_documents(DOCUMENT_PATH_INPUT, limit=DOCUMENT_LIMIT)
            logger.info(f"Processing {len(documents)} documents in parallel")

            with ThreadPoolExecutor(max_workers = PROCESSING_DOC_MAX_WORKERS) as executor:
                futures = {executor.submit(self.process_single_doc, doc): doc for doc in documents}

                for future in as_completed(futures):
                    try:
                        future.result()  # Throw except if process isn't complete
                    except Exception as e:
                        logger.error(f"Error processing document {futures[future]['file_path']}: {e}")

            return True
        except Exception as e:
            logger.error(f"An error occurred during parallel document processing: {e}")
            return False

    def execute_pipeline(self, strict:bool = False):
        # TODO Write docstring
        
        logger.info("Starting indexing pipeline")

        if not self.load_index():
            logger.error("Failed to load index.")
            if strict:
                return False

        if not self.load_chunks():
            logger.error("Failed to load chunks.")
            if strict:
                return False

        if not self.parallel_process_docs():
            logger.error("Error during document processing. Aborting.")
            return False

        if not self.build_index():
            logger.error("Failed to build index. Aborting.")
            return False

        if not self.save_chunks():
            logger.error("Failed to save chunks. Aborting.")
            return False
        
        logger.info("Indexing pipeline completed successfully")
        return True
