import mlflow
import mlflow.sklearn
from joblib import load
from pathlib import Path
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def register_existing_model():
    model_path = Path("models/compressed/disease_compressed.joblib")
    if not model_path.exists():
        print("Model file not found. Skipping registry.")
        return
        
    model = load(model_path)
    model_name = "thyrax_xgboost"
    
    mlflow.set_experiment("ThyraX_Initial_Import")
    
    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgboost_model",
            registered_model_name=model_name
        )
        print(f"Model logged in run {run.info.run_id}")
        
    client = MlflowClient()
    # Find the latest version
    model_version_details = client.search_model_versions(f"name='{model_name}'")
    if model_version_details:
        latest_version = model_version_details[0].version
        # Assign the 'Production' alias to it
        client.set_registered_model_alias(model_name, "Production", latest_version)
        print(f"Assigned 'Production' alias to version {latest_version} of {model_name}")

if __name__ == "__main__":
    register_existing_model()
