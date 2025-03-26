import os
import json
from rank_bm25 import BM25Okapi
import numpy as np
from config.config import RETRIEVAL_TOP_K, LEXICAL_STORE_PATH
from utils.logger import logger

class BM25LexicalStore:
    """A class for storing and retrieving documents using BM25 lexical search.

    This class implements a document store that uses the BM25 ranking algorithm for lexical search.
    It allows storing documents, building a BM25 index, and retrieving relevant documents based on
    text queries. The store can be persisted to and loaded from disk.

    Attributes:
        store_path (str): Path to the file where the document store is saved.
        documents (list): List of documents stored in the lexical store.
        bm25 (BM25Okapi): The BM25 index built from the stored documents.

    Note:
        BM25 is a bag-of-words retrieval function that ranks documents based on the query terms
        appearing in each document, regardless of their proximity within the document.
    """

    def __init__(self, store_path: str = LEXICAL_STORE_PATH, preload: bool = True):
        self.store_path = store_path
        self.documents = []
        self.bm25 = None

        if preload:
            self.load_store()

    def load_store(self):
        """Load the document store from disk.

        This method loads previously stored documents and their metadata from the store_path file.
        If the file exists, it loads the documents and rebuilds the BM25 index.

        Returns:
            bool: True if the store was successfully loaded, False otherwise.

        Raises:
            Exception: If an error occurs while loading or parsing the store file.
        """

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
        """Save the document store to disk.

        This method saves the current state of the document store to the store_path file.
        The documents are serialized to JSON format for persistent storage.

        Returns:
            bool: True if the store was successfully saved, False otherwise.

        Raises:
            Exception: If an error occurs while saving or writing to the store file.
        """
        
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
        """Build the BM25 index from the current documents.

        This method tokenizes the content of the documents stored in the 
        document store and constructs a BM25 index for efficient retrieval. 
        The BM25 algorithm is a probabilistic model used for information retrieval 
        that ranks documents based on their relevance to a given query.

        Returns:
            bool: True if the index was successfully built, False otherwise.
        """
        
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
        """Add a single document to the BM25 index.

        This method appends a document to the internal document store and rebuilds the BM25 index
        to include the newly added document. It ensures that the index is always up-to-date with the
        current set of documents.

        Args:
            doc (dict): The document to be added, which should contain the necessary fields for indexing.

        Returns:
            bool: True if the document was successfully added and the index rebuilt, False otherwise.
        """
        
        try:
            self.documents.append(doc)
            self.build_bm25_index()  # Rebuild BM25

            logger.info(f"Document successfully adding in bm25 index")
            return True
        except Exception as e:
            logger.error(f"An error has occurred while adding document in bm25 index: {e}")
            return False
    
    def add_documents(self, docs: list):
        """Add a list of documents to the BM25 index.

        This method appends multiple documents to the internal document store and rebuilds the BM25 index
        to include the newly added documents. It ensures that the index is always up-to-date with the
        current set of documents.

        Args:
            docs (list): A list of documents to be added, where each document should contain the necessary fields for indexing.

        Returns:
            bool: True if the documents were successfully added and the index rebuilt, False otherwise.
        """
        
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
        """Delete a document from the BM25 index by its document ID.

        This method removes a document from the internal document store based on the provided
        document ID. After deletion, it rebuilds the BM25 index to ensure it reflects the current
        set of documents.

        Args:
            doc_id (str): The ID of the document to be deleted.

        Returns:
            bool: True if the document was successfully deleted and the index rebuilt, False otherwise.
        """
        
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
        """Search for documents in the BM25 index based on a query.

        This method retrieves the top_k most relevant documents from the BM25 index
        that match the provided query. It tokenizes the query and calculates the
        relevance scores for each document in the index, returning the documents
        sorted by their scores.

        Args:
            query (str): The search query to find relevant documents.
            top_k (int, optional): The number of top results to return. Defaults to RETRIEVAL_TOP_K.

        Returns:
            list: A list of the top_k most relevant documents, sorted by relevance score.
                  Returns an empty list if an error occurs during the search.
        """
        
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
        """Retrieve a document from the BM25 index by its document ID.

        This method searches through the stored documents and returns the document
        that matches the provided document ID. If no document is found with the given
        ID, it returns None.

        Args:
            doc_id (str): The ID of the document to retrieve.

        Returns:
            dict or None: The document matching the provided ID, or None if no document
            with that ID exists.
        """
        
        try:
            for doc in self.documents:
                if doc["document_id"] == doc_id:
                    return doc
                
            return None
        except Exception as e:
            logger.error(f"An error has occurred while finding document: {e}")
            return None
    