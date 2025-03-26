from langchain.text_splitter import TokenTextSplitter
from transformers import GPT2TokenizerFast
from utils.logger import logger

def chunk_text(text: str, chunk_size: int, overlap: int):
    """Split text into chunks using a token-based text splitter.

    This function splits the input text into smaller chunks based on token count,
    with configurable chunk size and overlap between chunks.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of tokens per chunk.
        overlap (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
        Empty list if an error occurs during processing.
    """

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
    """Split text into chunks using GPT-2 tokenizer and token-based text splitter.

    This function splits the input text into smaller chunks using the GPT-2 tokenizer,
    with configurable chunk size and overlap between chunks. The GPT-2 tokenizer provides
    more accurate token counting for transformer-based models.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of tokens per chunk.
        overlap (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
        Empty list if an error occurs during processing.
    """

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
