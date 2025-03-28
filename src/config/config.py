import os
from dotenv import load_dotenv

load_dotenv()

# Size of each chunk of data
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
# Size of overlap between chunks
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE"))
# Number of top results to retrieve
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K"))
# Number of top results to rerank
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K"))
# Path to the index file
INDEX_PATH = os.getenv("INDEX_PATH")
# Path to the lexical store
LEXICAL_STORE_PATH = os.getenv("LEXICAL_STORE_PATH")
# Path to the documents
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")
# Path to the chunks
CHUNKS_PATH = os.getenv("CHUNKS_PATH")
# Path to the context chunks
CONTEXT_CHUNKS_PATH = os.getenv("CONTEXT_CHUNKS_PATH")
# Limit on the number of documents to process
DOCUMENT_LIMIT = int(os.getenv("DOCUMENT_LIMIT"))

# Activate mflow logs
MLFLOW_ENABLE = os.getenv("MLFLOW_ENABLE")
# mlflow host
MFFLOW_HOST = os.getenv("MFFLOW_HOST")
# mlflow port
MFFLOW_PORT = os.getenv("MFFLOW_PORT")

# Provider for embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
# Model used for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Dimension of the embeddings
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

# Provider for contextual language model
LLM_CONTEXTUAL_PROVIDER = os.getenv("LLM_CONTEXTUAL_PROVIDER")
# Model used for contextual language processing
LLM_CONTEXTUAL_MODEL = os.getenv("LLM_CONTEXTUAL_MODEL")
# Context length for the contextual language model
LLM_CONTEXTUAL_CONTEXT_LENGTH = int(os.getenv("LLM_CONTEXTUAL_CONTEXT_LENGTH"))

# Provider for generative language model
LLM_GENERATIVE_PROVIDER = os.getenv("LLM_GENERATIVE_PROVIDER")
# Model used for generative language processing
LLM_GENERATIVE_MODEL = os.getenv("LLM_GENERATIVE_MODEL")
# Context length for the generative language model
LLM_GENERATIVE_CONTEXT_LENGTH = int(os.getenv("LLM_GENERATIVE_CONTEXT_LENGTH"))

# API key for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# API key for Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# API key for Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Token for Hugging Face Hub
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

