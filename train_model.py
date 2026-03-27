import mlflow
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///./mlflow.db"
mlflow.set_tracking_uri("sqlite:///./mlflow.db")

mlflow.set_experiment("ThyraX_Clinical_Assessment")

print("🚀 Starting Model Training with MLflow...")

X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    print(f" Active Run ID: {run.info.run_id}")
    
    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,
        "learning_rate": 0.1
    }
    
    mlflow.log_params(params)
    mlflow.log_param("model_type", "XGBoost Classifier")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="ThyraX_Disease_Classifier")
    
    print("-" * 30)
    print(f" Model trained successfully!")
    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print(" Model and metrics securely logged to MLflow Registry.")