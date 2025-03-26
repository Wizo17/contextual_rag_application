import sys
import os

# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

# Import required libraries and custom modules
import streamlit as st
from src.utils.logger import logger

# Set up the main title of the Streamlit application
st.title('Contextual RAG Example')

