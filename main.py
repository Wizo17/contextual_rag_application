import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.config.config import CHUNK_SIZE, OVERLAP_SIZE
from src.preprocessing.document_processor import load_documents
from src.preprocessing.chunk_processor import chunk_text, chunk_text_gpt2

def main():
    print("C/RAG Example")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "raw")
    documents = load_documents(data_path, limit=1)

    for doc in documents:
        chunks = chunk_text(doc, CHUNK_SIZE, OVERLAP_SIZE)

        for chunk in chunks:
            print("##################################################################################################")
            print(chunk)

if __name__ == "__main__":
    main()
