import sys
import os


# Add the src directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")

# Import required libraries and custom modules
import streamlit as st
from src.utils.logger import logger
from src.services.llm_session import LLMSession
from src.services.indexer import Indexer
from src.services.indexer2 import Indexer2
from src.config.config import LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL



# Page configuration
st.set_page_config(page_title="Contextual RAG Example", layout="centered")

# Set up the main title of the Streamlit application
st.title("Contextual RAG Example")



# Init session state
if "init_mlflow" not in st.session_state:
    from src.utils.logger import setup_mlflow
    setup_mlflow()
    st.session_state.init_mlflow = True

if "llm_session" not in st.session_state:
    st.session_state.llm_session = LLMSession(LLM_GENERATIVE_PROVIDER, LLM_GENERATIVE_MODEL)

if "indexer" not in st.session_state:
    # st.session_state.indexer = Indexer()
    st.session_state.indexer = Indexer2()
    st.session_state.indexer.load_index()
    st.session_state.indexer.load_chunks()

if "messages" not in st.session_state:
    st.session_state.messages = []



# Show old messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources button
        if message["role"] == "assistant" and message.get("sources"):
            with st.popover("📄 Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i + 1}:** {source}")

# User input area
user_input = st.chat_input("Posez votre question...")



if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    
    # Show history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response
    doc_res = st.session_state.indexer.query_index(user_input.strip())
    response = st.session_state.llm_session.get_response_from_documents(user_input, doc_res)
    
    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)

        # Show sources
        if doc_res:
            with st.popover("📄 Sources"):
                for i, source in enumerate(doc_res):
                    st.markdown(f"**Source {i + 1}:** {source}")
        
    # Add response to history
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": doc_res})
