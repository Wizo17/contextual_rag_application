import os
import json
from rank_bm25 import BM25Okapi
import numpy as np
from config.config import RETRIEVAL_TOP_K, LEXICAL_STORE_PATH
from utils.logger import logger

class BM25LexicalStore:
    # TODO Write docstring

    def __init__(self, store_path: str = LEXICAL_STORE_PATH, preload: bool = True):
        # TODO Write docstring
        self.store_path = store_path
        self.documents = []
        self.bm25 = None

        if preload:
            self.load_store()

    def load_store(self):
        # TODO Write docstring
        try:
            result = False

            if os.path.exists(self.store_path):
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    self.build_bm25_index()  # Rebuild bm25 index
                    result = True

            logger.info(f"Store successfully loaded")
            return result
        except Exception as e:
            logger.error(f"An error has occurred while loading store: {e}")
            return False

    def save_store(self):
        # TODO Write docstring
        try:
            result = False

            data = {"documents": self.documents}
            with open(self.store_path, 'w') as f:
                json.dump(data, f)
                logger.info(f"BM25 store saved successfuly in {self.store_path}!")
                result = True

            logger.info(f"Store successfully saved")
            return result
        except Exception as e:
            logger.error(f"An error has occurred while saving store: {e}")
            return False

    def build_bm25_index(self):
        # TODO Write docstring
        if not self.documents:
            return False
        
        try:
            # Tokenize documents for BM25 (based on a simple word breakdown)
            tokenized_corpus = [doc['content'].split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Index bm25 built")

            logger.info(f"Bm25 index successfully built")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while building bm25 index: {e}")
            return False

    def add_document(self, doc: dict):
        # TODO Write docstring
        try:
            self.documents.append(doc)
            self.build_bm25_index()  # Rebuild BM25

            logger.info(f"Document successfully adding in bm25 index")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while adding document in bm25 index: {e}")
            return False
    
    def add_documents(self, docs: list):
        # TODO Write docstring
        try:
            for doc in docs:
                self.documents.append(doc)

            self.build_bm25_index()  # Rebuild BM25

            logger.info(f"Document list successfully adding in bm25 index")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while adding document list in bm25 index: {e}")
            return False

    def delete_document(self, doc_id: str):
        # TODO Write docstring
        try:
            # Delete doc by ID
            self.documents = [doc for doc in self.documents if doc["document_id"] != doc_id]
            self.build_bm25_index()  # Rebuild BM25

            logger.info(f"Document successfully deleted")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while deleting document: {e}")
            return False
    
    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        if not self.bm25:
            raise ValueError("BM25 index is not built.")
        
        try:
            # Tokenise query
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            ranked_docs = np.argsort(scores)[::-1]  # Sort by descending scores

            return [self.documents[i] for i in ranked_docs[:top_k]]
        except Exception as e:
            logger.error(f"An error has occurred while searching document: {e}")
            return []

    def get_document_by_id(self, doc_id: str):
        # TODO Write docstring
        try:
            for doc in self.documents:
                if doc["document_id"] == doc_id:
                    return doc
                
            return None
        except Exception as e:
            logger.error(f"An error has occurred while finding document: {e}")
            return None
    