from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from config.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
from templates.prompts import ADD_CONTEXT_HUMAN_PROMPT, ADD_CONTEXT_SYSTEM_PROMPT

class LLMSession:
    """A class to manage Language Learning Model (LLM) sessions across different providers.

    This class provides a unified interface to interact with various LLM providers
    including OpenAI, Anthropic, Ollama, and Google. It handles the initialization
    of the appropriate LLM client based on the specified provider and model.

    Attributes:
        provider (str): The LLM provider name (openai, anthropic, ollama, or google)
        model (str): The specific model name to use with the provider
        llm: The initialized LLM client instance
    """

    def __init__(self, provider, model):
        """Initialize a new LLM session.

        Args:
            provider (str, optional): The LLM provider to use. Defaults to value from global config.
            model (str, optional): The model name to use. Defaults to value from global config.

        Raises:
            Exception: If an invalid provider is specified.
        """
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


    def invoke(self, message):
        """Send a message to the LLM and get a structured response.

        Args:
            message (str): The input message or prompt to send to the LLM.

        Returns:
            # TODO Complete
        """
        return self.llm.invoke(message)
    
    def get_context(self, chunk, document):
        # TODO Write docstring

        input_message = [
            SystemMessage(content=ADD_CONTEXT_SYSTEM_PROMPT),
            HumanMessage(content=ADD_CONTEXT_HUMAN_PROMPT.format(
                chunk = chunk, 
                document = document
            ))
        ]

        return self.invoke(input_message).content
