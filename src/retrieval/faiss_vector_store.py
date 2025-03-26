import faiss
import numpy as np
import os
from config.config import RETRIEVAL_TOP_K, INDEX_PATH
from utils.logger import logger

# logging.getLogger('faiss').setLevel(logging.WARNING)

class FaissVectorStore:
    # TODO Write docstring

    def __init__(self, embedding_dim: int, index_file_path: str = INDEX_PATH, preload: bool = True):
        # TODO Write docstring
        self.embedding_dim = embedding_dim
        self.index_file_path = index_file_path

        # FAISS index initialization
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"FAISS index created on CPU successfully!")

        if preload:
            self.load_index()
    
    def add_elements(self, embeddings: np.ndarray, metadata: list = []):
        # TODO Write docstring
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        try:
            # Ensure that embeddings are a 2D numpy array (N, embedding_dim)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            self.index.add(embeddings)
            logger.info(f"Element successfully added in faiss index")

            return True
        except Exception as e:
            logger.error(f"An error has occurred while adding elements in faiss index: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        # Ensure that the query vector is in the right format (2D)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        try:
            distances, indices = self.index.search(query_embedding, top_k)

            return indices, distances
        except Exception as e:
            logger.error(f"An error has occurred while searching: {e}")
            return [], []
    
    def save_index(self):
        # TODO Write docstring
        try:
            result = False
            if self.index_file_path:
                faiss.write_index(self.index, self.index_file_path)
                logger.info(f"Faiss Index saved successfuly in {self.index_file_path}!")
                result = True
            else:
                logger.error(f"Faiss Index not saved!")
                result = False

            return result
        except Exception as e:
            logger.error(f"An error has occurred while saving index: {e}")
            return False
    
    def load_index(self):
        # TODO Write docstring
        try:
            if self.index_file_path and os.path.exists(self.index_file_path):
                self.index = faiss.read_index(self.index_file_path)
                logger.info(f"Faiss index read")

                return True
            else:
                raise FileNotFoundError(f"FAISS index doesn't exist: {self.index_file_path}")
        except Exception as e:
            logger.error(f"An error has occurred while loading index: {e}")
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
