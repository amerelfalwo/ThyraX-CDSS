# ThyraX CDSS

Clinical Decision Support System for thyroid assessment, ultrasound AI analysis, and guideline-aware assistant workflows.

## Overview

ThyraX combines multiple components in one stack:

- FastAPI backend for clinical, imaging, patient, and agent endpoints.
- Streamlit frontend for physician-facing workflows.
- MLflow and DVC support for experiment tracking and model lifecycle.
- Internal API key protection for all sensitive AI routes.

Core capabilities:

- Clinical assessment using a production XGBoost model.
- AI OCR extraction from lab report images.
- Ultrasound validation and ONNX prediction pipeline.
- Longitudinal patient dashboard with visit history.
- Agent chat with retrieval + structured medical tooling.

## Architecture

### Backend (FastAPI)

- Clinical router: disease prediction and rule-based next-step routing.
- Labs router: OCR extraction of thyroid lab values from images.
- Image router: ultrasound validation and segmentation/classification prediction.
- Patient router: patient creation, visit storage, and dashboard retrieval.
- Agent router: conversational medical assistant endpoint.

### Security

- All protected routes require header: X-AI-Service-Key.
- Public health endpoint: GET /health.
- If INTERNAL_SERVICE_KEY is not configured, protected routes return server error by design.

### MLOps

- MLflow for experiment/model tracking and registry.
- DVC for large data/model artifact versioning.

### Frontend

- Streamlit application with tabs for dashboard, clinical+OCR, imaging, and assistant chat.

## API Endpoints

| Method | Endpoint                        | Auth | Description                                |
| ------ | ------------------------------- | ---- | ------------------------------------------ |
| GET    | /health                         | No   | Service health check                       |
| POST   | /clinical/assess                | Yes  | Clinical disease prediction + routing      |
| POST   | /labs/extract                   | Yes  | OCR extraction from lab report image       |
| POST   | /image/validate                 | Yes  | Ultrasound gatekeeper validation           |
| POST   | /image/predict                  | Yes  | ONNX imaging pipeline prediction           |
| POST   | /patient/create                 | Yes  | Create patient profile                     |
| POST   | /patient/{patient_id}/visit     | Yes  | Save visit data                            |
| GET    | /patient/{patient_id}/dashboard | Yes  | Patient timeline and latest recommendation |
| POST   | /agent/chat                     | Yes  | AI assistant endpoint                      |

## Project Structure

```text
ThyraX-CDSS/
├── app/
│   ├── agent/
│   ├── core/
│   ├── disease/
│   ├── models/
│   ├── routers/
│   ├── schemas/
│   └── segmentation/
├── data/
├── frontend/
├── media/
├── models/
├── scripts/
├── docker-compose.yml
├── Dockerfile
├── main.py
├── pyproject.toml
└── train_model.py
```

## Requirements

- Python 3.12+
- uv (recommended package manager)
- Optional: Docker + Docker Compose

## Environment Variables

Create a .env file at repository root:

```env
GOOGLE_API_KEY_LABS=your_google_api_key
GOOGLE_API_KEY_VISION=your_google_api_key
GOOGLE_API_KEY_AGENT=your_google_api_key
INTERNAL_SERVICE_KEY=your_internal_service_key
DATABASE_URL=sqlite+aiosqlite:///./thyrax.db
CHROMA_PERSIST_DIR=./data/vector_store
```

Notes:

- INTERNAL_SERVICE_KEY is mandatory for protected endpoints.
- If GOOGLE_API_KEY_VISION is empty, ultrasound validation currently defaults to pass.

## Local Setup

Install dependencies:

```bash
uv sync
```

Optional model/registry preparation:

```bash
uv run python train_model.py
uv run python scripts/register_mlflow.py
```

Run services in separate terminals:

1. API

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

2. Frontend

```bash
uv run streamlit run frontend/app.py
```

3. MLflow UI (optional)

```bash
uv run mlflow ui --backend-store-uri sqlite:///./mlflow.db --port 5000
```

Service URLs:

- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501
- MLflow: http://localhost:5000

## Project Scripts (Missing Part)

The repository includes helper scripts that are important for full workflow but were easy to miss:

1. Model training script

```bash
uv run python train_model.py
```

- trains an XGBoost multi-class model
- logs params/metrics to MLflow
- registers model as ThyraX_Disease_Classifier

2. Register existing production model

```bash
uv run python scripts/register_mlflow.py
```

- loads local model file from models/compressed/disease_compressed.joblib
- registers it in MLflow as thyrax_xgboost
- sets MLflow alias Production to latest version

3. Agent endpoint test script

```bash
uv run python test_agent.py
```

- sends a realistic /agent/chat request
- verifies tools invocation and response path
- useful for quick smoke test after deployment

## Agent and RAG Notes

The assistant depends on ChromaDB collections configured in app settings:

- CHROMA_GUIDELINES_COLLECTION=pdf_documents
- CHROMA_SIMILAR_CASES_COLLECTION=similar_cases

If these collections are not populated yet, agent responses will still run but retrieval quality will be limited.

Recommended minimum for agent readiness:

1. Ensure .env has valid GOOGLE_API_KEY_AGENT and INTERNAL_SERVICE_KEY.
2. Ensure database is initialized by running the API once.
3. Ensure vector data exists under data/vector_store.
4. Run test_agent.py to verify end-to-end behavior.

## Docker Setup

Build and run:

```bash
docker compose up --build -d
```

What this starts:

- api service on port 8000 (FastAPI)
- frontend service on port 8501 (Streamlit)
- mlflow service on port 5000 (MLflow UI)

Useful lifecycle commands:

```bash
# Start existing containers
docker compose up -d

# Rebuild a single service
docker compose up -d --build api

# Check container status
docker compose ps
```

Stop services:

```bash
docker compose down
```

Useful logs:

```bash
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f mlflow
```

Default container ports:

- 8000: thyrax_api
- 8501: thyrax_frontend
- 5000: thyrax_mlflow

Persistent mounted paths:

- ./thyrax.db -> /app/thyrax.db
- ./data/vector_store -> /app/data/vector_store
- ./models/compressed -> /app/models/compressed
- ./mlflow.db -> /app/mlflow.db
- ./mlruns -> /app/mlruns

## Cloud Deployment Script (Azure VM)

If you already prepared the cloud bootstrap script, use:

```bash
bash scripts/setup_azure_vm.sh
```

The script will:

- update the VM packages
- install Docker Engine and Docker Compose plugin
- add your user to the docker group
- create a .env template if it does not exist
- print run commands and public URLs

After running the script:

```bash
# 1) Activate docker group in current session
newgrp docker

# 2) Fill your secrets
nano .env

# 3) Start services
docker compose up --build -d
```

Open these inbound ports on your cloud firewall/NSG:

- 8000 for FastAPI
- 8501 for Streamlit
- 5000 for MLflow

## Example Authenticated Request

```bash
curl -X POST http://localhost:8000/clinical/assess \
     -H "X-AI-Service-Key: your_internal_service_key" \
     -H "Content-Type: application/json" \
     -d '{
          "patient_id": 1,
          "age": 45,
          "on_thyroxine": false,
          "thyroid_surgery": false,
          "query_hyperthyroid": false,
          "TSH": 2.5,
          "T3": 1.2,
          "TT4": 8.0,
          "FTI": 5.5,
          "T4U": 1.1,
          "nodule_present": true
     }'
```

## Tech Stack

- Backend: FastAPI, SQLAlchemy Async, Uvicorn
- ML: XGBoost, ONNX Runtime, scikit-learn
- LLM/Agent: LangChain ecosystem + Google Generative AI
- Vector Store: ChromaDB + sentence-transformers
- Frontend: Streamlit
- MLOps: MLflow, DVC, Evidently

## Production Notes

- Restrict CORS origins before internet-facing deployment.
- Keep INTERNAL_SERVICE_KEY out of source control.
- Run behind reverse proxy + TLS for production traffic.
- Use managed database and persistent object storage for scale.
