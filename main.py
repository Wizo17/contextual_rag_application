import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

from src.services.indexer import Indexer
from src.utils.logger import setup_mlflow

setup_mlflow()

def main():
    print("Contextual RAG Example")

    print("Index construction")

    indexer = Indexer()
    indexer.process_docs()
    indexer.build_index()
    indexer.save_chunks()

    print("Successful index construction")
        

if __name__ == "__main__":
    main()

