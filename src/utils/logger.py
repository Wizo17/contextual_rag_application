import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow():
    from config.config import MLFLOW_ENABLE, MFFLOW_HOST, MFFLOW_PORT

    if MLFLOW_ENABLE.lower() == "yes":
        logger.info("Starting mlflow session")
        import mlflow
        mlflow.set_tracking_uri(f"{MFFLOW_HOST}:{MFFLOW_PORT}")
        mlflow.set_experiment("Contextual RAG Example")
        mlflow.langchain.autolog()
