import faiss
import numpy as np
import os
from config.config import RETRIEVAL_TOP_K, INDEX_PATH
from utils.logger import logger

# logging.getLogger('faiss').setLevel(logging.WARNING)

class FaissVectorStore:
    """
    A class for storing and retrieving embeddings using FAISS (Facebook AI Similarity Search).

    This class provides functionality to create a FAISS index for efficient similarity search
    of vector embeddings. It allows adding new embeddings, searching for the nearest neighbors
    based on a query embedding, and saving/loading the index to/from disk.

    Attributes:
        embedding_dim (int): The dimensionality of the embeddings.
        index_file_path (str): The file path where the FAISS index is saved.
        index (faiss.IndexFlatL2): The FAISS index used for similarity search.
    """

    def __init__(self, embedding_dim: int, index_file_path: str = INDEX_PATH, preload: bool = True):
        self.embedding_dim = embedding_dim
        self.index_file_path = index_file_path

        # FAISS index initialization
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"FAISS index created on CPU successfully!")

        if preload:
            self.load_index()
    
    def add_elements(self, embeddings: np.ndarray, metadata: list = []):
        """Add embeddings to the FAISS index.

        This method adds a set of embeddings to the FAISS index. It ensures that the embeddings
        are in the correct format and handles the addition of both single and multiple embeddings.

        Args:
            embeddings (np.ndarray): A 2D numpy array of shape (N, embedding_dim) containing the embeddings to be added.
            metadata (list, optional): A list of metadata associated with the embeddings. Defaults to an empty list.

        Returns:
            bool: True if the embeddings were successfully added, False otherwise.
        """
        
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
        """Search for the top_k most similar embeddings in the FAISS index based on a query embedding.

        This method takes a query embedding, searches the FAISS index for the top_k nearest neighbors,
        and returns their indices and distances. The query embedding is reshaped to ensure it is in the
        correct format for the search operation.

        Args:
            query_embedding (np.ndarray): A 1D numpy array representing the query embedding.
            top_k (int, optional): The number of top results to return. Defaults to RETRIEVAL_TOP_K.

        Returns:
            tuple: A tuple containing:
                - indices (np.ndarray): The indices of the top_k nearest neighbors in the FAISS index.
                - distances (np.ndarray): The distances of the top_k nearest neighbors from the query embedding.
        """
        
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
        """Save the FAISS index to disk.

        This method serializes the current state of the FAISS index and writes it to the specified
        file path. It ensures that the index can be reloaded later without needing to rebuild it.

        Returns:
            bool: True if the index was successfully saved, False otherwise.
        """
        
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
        """Load the FAISS index from disk.

        This method reads the FAISS index from the specified file path. If the index file exists,
        it loads the index into memory, allowing for subsequent search operations. If the index file
        does not exist, a FileNotFoundError is raised.

        Returns:
            bool: True if the index was successfully loaded, False otherwise.
        
        Raises:
            FileNotFoundError: If the index file does not exist at the specified path.
        """
        
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
        """Delete the FAISS index from disk.

        This method removes the FAISS index file from the specified file path. It checks if the index file
        exists before attempting to delete it. If the file does not exist, an error is logged.

        Returns:
            bool: True if the index was successfully deleted, False otherwise.
        """
        
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
