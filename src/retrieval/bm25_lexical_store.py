import os
import json
from rank_bm25 import BM25Okapi
import numpy as np
from config.config import RETRIEVAL_TOP_K, LEXICAL_STORE_PATH

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
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r') as f:
                data = json.load(f)
                self.documents = data["documents"]
                self.build_bm25_index()  # Rebuild bm25 index

    def save_store(self):
        # TODO Write docstring
        data = {"documents": self.documents}
        with open(self.store_path, 'w') as f:
            json.dump(data, f)

    def build_bm25_index(self):
        # TODO Write docstring
        if not self.documents:
            return
        
        # Tokenize documents for BM25 (based on a simple word breakdown)
        tokenized_corpus = [doc['content'].split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_document(self, doc: dict):
        # TODO Write docstring
        self.documents.append(doc)
        self.build_bm25_index()  # Rebuild BM25
    
    def add_documents(self, docs: list):
        # TODO Write docstring
        for doc in docs:
            self.documents.append(doc)

        self.build_bm25_index()  # Rebuild BM25

    def delete_document(self, doc_id: str):
        # TODO Write docstring
        # Delete doc by ID
        self.documents = [doc for doc in self.documents if doc["document_id"] != doc_id]
        self.build_bm25_index()  # Rebuild BM25
    
    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        if not self.bm25:
            raise ValueError("BM25 index is not built.")
        
        # Tokenise query
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_docs = np.argsort(scores)[::-1]  # Sort by descending scores
        return [self.documents[i] for i in ranked_docs[:top_k]]

    def get_document_by_id(self, doc_id: str):
        # TODO Write docstring
        for doc in self.documents:
            if doc["document_id"] == doc_id:
                return doc
        return None
    