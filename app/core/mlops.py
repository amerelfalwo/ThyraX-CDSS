import mlflow
import mlflow.sklearn
from app.core.config import settings
import logging

# Ensure tracking URI is set to a local SQLite database
mlflow.set_tracking_uri("sqlite:///./mlflow.db")

logger = logging.getLogger(__name__)

def load_production_model(model_name: str):
    """
    Fetches the latest production-ready model directly from the MLflow Model Registry.
    """
    try:
        # Expected URI format for aliases: models:/<model_name>@Production
        model_uri = f"models:/{model_name}@Production"
        logger.info(f"Attempting to load MLflow model from: {model_uri}")
        
        # Load the model directly from MLflow
        model = mlflow.sklearn.load_model(model_uri)
        return model
        
    except Exception as e:
        logger.error(f"Failed to load '{model_name}' from MLflow. Ensure MLflow server is running and model exists.")
        logger.error(f"Exception: {e}")
        # In a real environment, you might fallback to local disk here, but per requirements we strictly load from MLflow.
        raise RuntimeError(f"MLflow model load failed: {e}")
