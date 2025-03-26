from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from config.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from templates.prompts import ADD_CONTEXT_HUMAN_PROMPT, ADD_CONTEXT_SYSTEM_PROMPT, BASIC_QUESTION_SYSTEM_PROMPT, BASIC_QUESTION_HUMAN_PROMPT
from utils.logger import logger

class LLMSession:
    """
    A class to manage interactions with various LLM providers.

    This class initializes the specified LLM provider and model, allowing for
    the invocation of language model functionalities. Supported providers include
    OpenAI, Anthropic, Ollama, and Google.

    Attributes:
        provider (str): The name of the LLM provider.
        model (str): The model name to be used with the provider.
        llm: An instance of the language model corresponding to the specified provider.
    """

    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

        model_providers = {
            "openai": lambda: ChatOpenAI(model=self.model, openai_api_key=OPENAI_API_KEY),
            "anthropic": lambda: ChatAnthropic(model=self.model, anthropic_api_key=ANTHROPIC_API_KEY),
            "ollama": lambda: ChatOllama(model=self.model),
            "google": lambda: ChatGoogleGenerativeAI(model=self.model, google_api_key=GOOGLE_API_KEY),
        }

        if self.provider not in ["openai", "ollama", "anthropic", "google"]:
            raise Exception(f"Invalid LLM provider: {self.provider}")

        self.llm = model_providers[self.provider]()
    
    def get_context(self, chunk, document):
        """Generate context for a given chunk and document.

        This method constructs a context message for the language model by combining
        a predefined system prompt with a human message that includes the provided
        chunk and document. It invokes the language model to generate a response
        based on this context.

        Args:
            chunk (str): The text chunk to be included in the context.
            document (str): The document to be included in the context.

        Returns:
            str: The generated context response from the language model.
                 Returns None if an error occurs during invocation.
        """
        
        try:
            input_message = [
                SystemMessage(content=ADD_CONTEXT_SYSTEM_PROMPT),
                HumanMessage(content=ADD_CONTEXT_HUMAN_PROMPT.format(
                    chunk = chunk, 
                    document = document
                ))
            ]

            return self.llm.invoke(input_message).content
        except Exception as e:
            logger.error(f"An error has occurred while invoking model - get_context: {e}")
            return None
    
    def get_response_from_documents(self, query: str, documents: list):
        """Generate a response from the language model based on a query and a list of documents.

        This method constructs a context message for the language model by combining
        a predefined system prompt with a human message that includes the provided
        query and documents. It invokes the language model to generate a response
        based on this context.

        Args:
            query (str): The query to be included in the context.
            documents (list): A list of documents to be included in the context.

        Returns:
            str: The generated response from the language model.
                 Returns None if an error occurs during invocation.
        """
        
        try:
            input_message = [
                SystemMessage(content=BASIC_QUESTION_SYSTEM_PROMPT),
                HumanMessage(content=BASIC_QUESTION_HUMAN_PROMPT.format(
                    query = query, 
                    documents = "\nChunk: ".join(documents)
                ))
            ]

            return self.llm.invoke(input_message).content
        except Exception as e:
            logger.error(f"An error has occurred while invoking model - get_response_from_documents: {e}")
            return None
