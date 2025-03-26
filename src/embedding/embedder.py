import openai
import numpy as np
# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import login
from utils.logger import logger
from config.config import HUGGINGFACE_HUB_TOKEN, OPENAI_API_KEY, EMBEDDING_MODEL

class Embedder:
    """A class for generating embeddings from text using various embedding models.

    This class provides functionality to generate vector embeddings from text input using
    different embedding models like OpenAI's text embeddings, LegalBERT, BioBERT, etc.
    It handles the initialization of models and tokenizers, and provides methods to 
    generate embeddings using the specified model type.

    Attributes:
        None

    Note:
        Currently only OpenAI embeddings are enabled. Other model options are commented out
        but can be enabled by uncommenting the relevant code sections.
    """

    def __init__(self):
        # login(token=HUGGINGFACE_HUB_TOKEN)

        openai.api_key = OPENAI_API_KEY

        # legal-bert-base-uncased
        # self.legalbert_tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        # self.legalbert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')

        # biobert_v1.1_pubmed
        # self.biobert_tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
        # self.biobert_model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')

        # meta-llama/Llama-2-7b-hf
        # self.llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        # self.llama_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

        # mistralai/Mistral-7B-v0.1
        # self.mistral_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        # self.mistral_model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')

        # tiiuae/falcon-7b
        # self.tiiuae_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        # self.tiiuae_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")

    def get_embedding(self, text: str, model_type: str):
        """Generate embeddings for the given text using the specified model type.

        Args:
            text (str): The input text to generate embeddings for.
            model_type (str): The type of embedding model to use (e.g., "openai", "legalbert", etc.).

        Returns:
            numpy.ndarray: The generated embeddings as a numpy array.
            None: If an error occurs during embedding generation.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        
        try:
            if model_type == "openai":
                response = openai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )
                embeddings = np.array(response.data[0].embedding)
            
            # elif model_type == "legalbert":
            #     inputs = self.legalbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            #     outputs = self.legalbert_model(**inputs)
            #     embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

            # elif model_type == "biobert":
            #     inputs = self.biobert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            #     outputs = self.biobert_model(**inputs)
            #     embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

            # elif model_type == "llama":
            #     inputs = self.llama_tokenizer(text, return_tensors='pt', truncation=True)
            #     outputs = self.llama_model(**inputs)
            #     embeddings = self.llama_model.get_input_embeddings()(inputs['input_ids']).mean(dim=1).detach().numpy()

            # elif model_type == "mistral":
            #     inputs = self.mistral_tokenizer(text, return_tensors='pt', truncation=True)
            #     outputs = self.mistral_model(**inputs)
            #     embeddings = self.mistral_model.get_input_embeddings()(inputs['input_ids']).mean(dim=1).detach().numpy()

            # elif model_type == "tiiuae":
            #     inputs = self.tiiuae_tokenizer(text, return_tensors='pt', truncation=True)
            #     outputs = self.tiiuae_model(**inputs)
            #     embeddings = self.tiiuae_model.get_input_embeddings()(inputs['input_ids']).mean(dim=1).detach().numpy()

            else:
                raise ValueError(f"Embedding model '{model_type}' is not supported!")

            logger.info(f"Embedding successfully created")
            return embeddings
        except Exception as e:
            logger.error(f"An error has occured during pdf load file: {e}")
            None

    def fuse_embeddings(self, embeddings: list[np.ndarray], method: str = 'mean'):
        """Fuse multiple embeddings into a single embedding using different methods.

        Args:
            embeddings (list[np.ndarray]): List of embeddings to fuse together.
            method (str, optional): Method to use for fusion. Can be 'mean', 'sum' or 'concat'. Defaults to 'mean'.

        Returns:
            numpy.ndarray: The fused embedding.
            None: If an error occurs during fusion.

        Raises:
            ValueError: If the list of embeddings is empty or if the fusion method is not supported.
        """

        try:
            if not embeddings:
                raise ValueError("List of embeddings is empty.")
            
            if method == 'mean':
                fused_emb = np.mean(embeddings, axis=0)
        
            elif method == 'sum':
                fused_emb = np.sum(embeddings, axis=0)
            
            elif method == 'concat':
                fused_emb = np.concatenate(embeddings, axis=0)
            
            else:
                logger.error(f"Method '{method}' not supported. Use 'mean', 'sum' or 'concat'.")
                raise ValueError(f"Method '{method}' not supported. Use 'mean', 'sum' or 'concat'.")

            logger.info(f"Embedding successfully merged")
            return fused_emb
        except Exception as e:
            logger.error(f"An error has occurred while merging embedding: {e}")
            return None
        