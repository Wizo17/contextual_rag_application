from sentence_transformers import CrossEncoder
from config.config import RERANK_TOP_K
from utils.logger import logger

class Reranker:
    """A class for reranking search results using a cross-encoder model.

    This class provides functionality to rerank a list of search results based on their 
    relevance to a query using a cross-encoder model from sentence-transformers. It uses
    the 'ms-marco-MiniLM-L-6-v2' model which is specifically trained for passage reranking.

    Attributes:
        cross_encoder (CrossEncoder): The cross-encoder model used for reranking.
    """

    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank_results(self, query: str, results: list[str], top_k: int = RERANK_TOP_K):
        """Rerank a list of search results based on their relevance to a query.

        This method uses the cross-encoder model to rerank search results by computing
        relevance scores between the query and each result, then returning the top_k
        most relevant results.

        Args:
            query (str): The search query to compare results against.
            results (list[str]): List of text results to rerank.
            top_k (int, optional): Number of top results to return. Defaults to RERANK_TOP_K.

        Returns:
            list[str]: The top_k most relevant results, reranked by relevance score.
            Empty list if an error occurs during reranking.
        """
        
        try:
            reranked = self.cross_encoder.rank(
                query, 
                results, 
                top_k=top_k
            )
            
            logger.info(f"Reranking successfully")
            return reranked
        except Exception as e:
            logger.error(f"An error has occurred while reranking: {e}")
            return []
    