from langchain.text_splitter import TokenTextSplitter
from transformers import GPT2TokenizerFast
from utils.logger import logger

def chunk_text(text: str, chunk_size: int, overlap: int):
    # TODO Write docstring

    try:
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        logger.info(f"Spliting text with chunk_size={chunk_size} and overlap={overlap}")
        return text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"An error occured during split text: {e}")
        return []

def chunk_text_gpt2(text: str, chunk_size: int, overlap: int):
    # TODO Write docstring

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = TokenTextSplitter(
            encoding_name="gpt2",
            chunk_size=chunk_size, 
            chunk_overlap=overlap, 
        )

        logger.info(f"Spliting text with chunk_size={chunk_size} and overlap={overlap}")
        return text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"An error occured during split text: {e}")
        return []
