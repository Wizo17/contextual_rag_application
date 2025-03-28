import os
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config.config import OPENAI_API_KEY, RETRIEVAL_TOP_K, INDEX_PATH, EMBEDDING_PROVIDER, EMBEDDING_MODEL
from utils.logger import logger

class FaissLangchainVectorStore:
    # TODO Write docstring

    def __init__(self, provider: str = EMBEDDING_PROVIDER, model: str = EMBEDDING_MODEL, index_file_path: str = INDEX_PATH, preload: bool = True):
        self.provider = provider
        self.model = model
        self.index_file_path = index_file_path

        model_providers = {
            "openai": lambda: OpenAIEmbeddings(model=self.model, openai_api_key=OPENAI_API_KEY),
            "ollama": lambda: OllamaEmbeddings(model=self.model),
            "huggingface": lambda: HuggingFaceEmbeddings(model=self.model),
        }

        if self.provider not in ["openai", "ollama", "huggingface"]:
            raise Exception(f"Invalid embedding provider: {self.provider}")

        self.embeddings = model_providers[self.provider]()

        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        if preload:
            self.load_index()

    def add_elements(self, documents: list[Document], uuids: list[str]):
        # TODO Write docstring

        try:
            self.vector_store.add_documents(documents=documents, ids=uuids)

            logger.info(f"Elements successfuly added in Faiss Vector Store")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while adding elements in Faiss Vector Store: {e}")
            return False
        
    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K, filter: dict = None, with_score: bool = False):
        # TODO Write docstring

        try:
            if with_score:
                return self.vector_store.similarity_search_with_score(
                    query,
                    k=top_k,
                    filter=filter,
                )
            else:
                return self.vector_store.similarity_search(
                    query,
                    k=top_k,
                    filter=filter,
                )
        except Exception as e:
            logger.error(f"An error has occurred while searching elements in Faiss Vector Store: {e}")
            return False
        
    def delete_elements(self, uuids: list[str]):
        # TODO Write docstring

        try:
            res = self.vector_store.delete(ids=uuids)

            logger.info(f"Elements successfuly deleted in Faiss Store")
            return res
        except Exception as e:
            logger.error(f"An error has occurred while deleting elements in Faiss Vector Store: {e}")
            return False
        
    def save_index(self):
        # TODO Write docstring

        try:
            self.vector_store.save_local(self.index_file_path)

            logger.info(f"Faiss Store successfuly saved")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while saving Faiss Vector Store: {e}")
            return False
        
    def load_index(self):
        # TODO Write docstring

        try:
            self.vector_store = FAISS.load_local(
                self.index_file_path, self.embeddings, allow_dangerous_deserialization=True
            )

            logger.info(f"Faiss Store successfuly loaded")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while loading Faiss Vector Store: {e}")
            return False
    
    def delete_index(self):
        # TODO Write docstring

        try:
            if self.index_file_path and os.path.exists(self.index_file_path):
                os.remove(self.index_file_path)
                logger.info(f"Index deleted : {self.index_file_path}")
                return True
            else:
                logger.error(f"There is no index at the location: {self.index_file_path}")
                return False
        except Exception as e:
            logger.error(f"An error has occurred while deleting index: {e}")
            return False
        