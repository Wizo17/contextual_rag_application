import openai
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import login
from utils.logger import logger
from config.config import HUGGINGFACE_HUB_TOKEN, OPENAI_API_KEY

class Embedder:
    # TODO Write docstring

    def __init__(self):
        login(token=HUGGINGFACE_HUB_TOKEN)

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
        # TODO Write docstring
        
        try:
            if model_type == "openai":
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
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

            return embeddings
        except Exception as e:
            logger.error(f"An error occured during pdf load file: {e}")
            None

    def fuse_embeddings(self, embeddings: list[np.ndarray], method: str = 'mean'):
        # TODO Write docstring

        if not embeddings:
            raise ValueError("List of embeddings is empty.")
        
        if method == 'mean':
            fused_emb = np.mean(embeddings, axis=0)
    
        elif method == 'sum':
            fused_emb = np.sum(embeddings, axis=0)
        
        elif method == 'concat':
            fused_emb = np.concatenate(embeddings, axis=0)
        
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'mean', 'sum' or 'concat'.")

        return fused_emb
        
 