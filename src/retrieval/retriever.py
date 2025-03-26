import faiss
import elasticsearch
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
from config.config import RETRIEVAL_TOP_K

class Retriever:

    def __init__(self, embedding_dim: int):
        # 3. Document stores
        self.faiss_store = FAISSDocumentStore(
            faiss_index_factory_str='Flat', 
            embedding_dim=embedding_dim
        )
        
        self.es_client = elasticsearch.Elasticsearch()
        self.es_store = ElasticsearchDocumentStore(
            host='localhost', 
            port=9200, 
            index='medical_legal_docs'
        )
        
        # 4. Retrievers
        self.vector_retriever = EmbeddingRetriever(document_store=self.faiss_store)
        self.bm25_retriever = BM25Retriever(document_store=self.es_store)

    def retrieve_vector_documents(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k)
        return vector_results
        
    def retrieve_lexical_documents(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        # TODO Write docstring
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        return bm25_results
