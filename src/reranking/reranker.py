from sentence_transformers import CrossEncoder
from config.config import RERANK_TOP_K
from utils.logger import logger

class Reranker:
    # TODO Write docstring

    def __init__(self):
        # TODO Write docstring
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # def scored_results(self, query: str, results: list[str]):
    #     # TODO Write docstring
    #     data = []
    #     for doc in results:
    #         data.append(query, doc)

    #     scores = self.cross_encoder.predict(data)
    #     return scores

    def rerank_results(self, query: str, results: list[str], top_k: int = RERANK_TOP_K):
        # TODO Write docstring
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
    