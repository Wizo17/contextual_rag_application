import faiss
import numpy as np
import os
from config.config import RETRIEVAL_TOP_K, INDEX_PATH

class FaissVectorStore:
    # TODO Write docstring

    def __init__(self, embedding_dim: int, index_file_path: str = INDEX_PATH, preload: bool = True):
        # TODO Write docstring
        self.embedding_dim = embedding_dim
        self.index_file_path = index_file_path

        # FAISS index initialization
        self.index = faiss.IndexFlatL2(embedding_dim)

        if index_file_path and os.path.exists(index_file_path) and preload:
            # Load index if exist
            self.index = faiss.read_index(index_file_path)
    
    def add_elements(self, embeddings: np.ndarray, metadata: list = []):
        # TODO Write docstring
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # Ensure that embeddings are a 2D numpy array (N, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        # Ensure that the query vector is in the right format (2D)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)
        return indices, distances
    
    def save_index(self):
        # TODO Write docstring
        if self.index_file_path:
            faiss.write_index(self.index, self.index_file_path)
    
    def load_index(self):
        # TODO Write docstring
        if self.index_file_path and os.path.exists(self.index_file_path):
            self.index = faiss.read_index(self.index_file_path)
        else:
            raise FileNotFoundError(f"FAISS index doesn't exist: {self.index_file_path}")
        
    def delete_index(self):
        # TODO Write docstring
        if self.index_file_path and os.path.exists(self.index_file_path):
            os.remove(self.index_file_path)
            print(f"Index deleted : {self.index_file_path}")
        else:
            print(f"There is no index at the location: {self.index_file_path}")
