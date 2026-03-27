# ThyraX CDSS - Clinical Decision Support System

ThyraX is a comprehensive Clinical Decision Support System (CDSS) for Thyroid Cancer diagnosis, engineered for production use. It strictly follows the Human-in-the-Loop principle, integrating a robust FastAPI backend, a modern Streamlit frontend, and a complete MLOps pipeline for model tracking and versioning.

## System Architecture

The project is structured into three main components:

### 1. Backend Service (FastAPI)
A modular API architecture divided into highly specialized routers:
- Labs Router: Premium AI OCR extraction using Gemini Vision to map medical laboratory report images to structured JSON data.
- Clinical Router: Core assessment engine. Evaluates laboratory features (TSH, T3, TT4, FTI, T4U) through our XGBoost disease model and provides rule-based agentic routing.
- Image Router: End-to-end ultrasound pipeline featuring an AI gatekeeper for strict image validation and an ONNX-based segmentation/classification pipeline for TI-RADS staging.
- Patient Router: Handles patient creation and longitudinal tracking of laboratory metrics and clinical history stored in SQLite/PostgreSQL.
- Chat Router: LangChain ReAct medical agent integrated with RAG (18+ medical guidelines, ACR TI-RADS, PubMed) and SQL patient history.

### 2. Security Layer
All AI endpoints are protected behind an Internal API Key authentication system using the `X-AI-Service-Key` header. This ensures only authorized microservices (the main website backend) can access paid AI features. A public `GET /health` endpoint is available for uptime monitoring without authentication.

### 3. MLOps Pipeline (MLflow and DVC)
- DVC manages local raw data sets and handles the versioning of large model binaries.
- MLflow orchestrates model tracking, metric logging, and model registry. The FastAPI backend dynamically fetches the XGBoost model aliased as "Production" from the local SQLite MLflow registry at runtime.

### 4. Frontend Application (Streamlit)
A professional interface designed for clinical environments featuring 4 dedicated tabs:
- Dashboard: Visualizes longitudinal patient metrics (such as TSH trends) and complete visit history.
- Clinical and OCR: Dual-column interface for uploading lab reports with AI auto-fill and manually submitting clinical disease assessments.
- Ultrasound AI: Image upload UI displaying the gatekeeper validation status alongside the final segmentation AI results.
- AI Assistant: A persistent chat interface natively prompting the downstream LangChain ReAct agent.

## API Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| GET | /health | No | Public health check for uptime monitoring |
| POST | /labs/extract | Yes | Premium Gemini Vision OCR for lab reports |
| POST | /clinical/assess | Yes | XGBoost disease prediction and agentic routing |
| POST | /image/validate | Yes | AI gatekeeper for ultrasound image validation |
| POST | /image/predict | Yes | ONNX segmentation and TI-RADS classification |
| POST | /patient/create | Yes | Create a new patient profile |
| POST | /patient/{id}/visit | Yes | Save confirmed visit data |
| GET | /patient/{id}/dashboard | Yes | Full patient history and longitudinal data |
| POST | /agent/chat | Yes | LangChain ReAct medical agent with RAG |

## Project Structure

```text
thyrax-cdss/
├── app/
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── mlops.py
│   │   └── security.py
│   ├── disease/
│   │   └── schema.py
│   ├── segmentation/
│   │   └── model.py
│   ├── schemas/
│   │   ├── clinical.py
│   │   ├── patient.py
│   │   ├── labs.py
│   │   ├── image.py
│   │   └── chat.py
│   ├── agent/
│   │   ├── agent.py
│   │   └── tools.py
│   └── routers/
│       ├── clinical.py
│       ├── labs.py
│       ├── image.py
│       ├── patient.py
│       └── chat.py
├── frontend/
│   └── app.py
├── models/
│   └── compressed/
├── data/
│   └── raw_data/
├── scripts/
│   └── register_mlflow.py
├── train_model.py
├── main.py
└── pyproject.toml
```

## Setup and Execution Guide

### Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY_LABS=your_google_ai_key
GOOGLE_API_KEY_VISION=your_google_ai_key
GOOGLE_API_KEY_AGENT=your_google_ai_key
DATABASE_URL=sqlite+aiosqlite:///./thyrax.db
INTERNAL_SERVICE_KEY=your_secret_service_key
```

### Dependency Installation

```bash
uv sync
```

### MLOps Initialization

```bash
uv run dvc init
uv run dvc add models/compressed/ data/raw_data/
uv run python train_model.py
uv run python scripts/register_mlflow.py
```

### Running the System

You will need three separate terminal sessions to launch the entire architecture:

Terminal 1 - MLflow Tracking UI
```bash
uv run mlflow ui --backend-store-uri sqlite:///./mlflow.db --port 5000
```

Terminal 2 - FastAPI Backend
```bash
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Terminal 3 - Streamlit Frontend
```bash
uv run streamlit run frontend/app.py
```

Once all services are running:
- ThyraX CDSS Frontend: http://localhost:8501
- FastAPI Swagger Docs: http://localhost:8000/docs
- MLflow Tracking UI: http://localhost:5000

### Authenticating API Requests

All protected endpoints require the `X-AI-Service-Key` header. Example:

```bash
curl -X POST http://localhost:8000/clinical/assess \
     -H "X-AI-Service-Key: your_secret_service_key" \
     -H "Content-Type: application/json" \
     -d '{"patient_id": 1, "TSH": 2.5, "T3": 1.2, "TT4": 8.0, "FTI": 5.5, "T4U": 1.1, "nodule_present": true}'
```

Requests without a valid key will receive a `403 Forbidden` response. The `/health` endpoint does not require authentication.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Disease Model | XGBoost (MLflow Registry) |
| Image Pipeline | ONNX Runtime, OpenCV |
| AI Agent | LangChain, Google Gemini 2.5 Pro |
| Vision OCR | Google Gemini 2.5 Flash |
| RAG Knowledge Base | ChromaDB, all-MiniLM-L6-v2 |
| Patient Database | SQLAlchemy Async, SQLite |
| MLOps | MLflow, DVC, Evidently |
| Frontend | Streamlit |
| Package Manager | uv |

## Docker Deployment

The entire system is containerized using a single `Dockerfile` and orchestrated via `docker-compose.yml` into three dedicated services.

### Services

| Container | Port | Role |
|-----------|------|------|
| `thyrax_api` | 8000 | FastAPI backend serving all AI endpoints |
| `thyrax_frontend` | 8501 | Streamlit clinical UI |
| `thyrax_mlflow` | 5000 | MLflow model tracking and registry |

### Persistent Volumes

The following directories are mounted as volumes so that data survives container restarts:

| Volume | Purpose |
|--------|---------|
| `./thyrax.db` | SQLite patient database |
| `./mlflow.db` + `./mlruns` | MLflow experiment and model tracking state |
| `./data/vector_store` | ChromaDB vector embeddings (RAG knowledge base) |
| `./models/compressed` | ONNX and XGBoost model binaries (not baked into the image) |

### Build and Launch

```bash
docker compose up --build -d
```

To stop all services:

```bash
docker compose down
```

To view service logs:

```bash
docker compose logs -f api
docker compose logs -f frontend
```

## Azure VM Deployment

A fully automated setup script is included for provisioning a fresh Ubuntu VM on Microsoft Azure.

### One-Step Provisioning

On the fresh VM, run:

```bash
bash scripts/setup_azure_vm.sh
```

This script automatically:
1. Updates and upgrades Ubuntu apt packages
2. Installs Docker Engine, CLI, containerd, and the Compose plugin from the Official Docker apt repository (not snap)
3. Adds the current user to the docker group (no sudo required)
4. Creates a `.env` template with all required keys
5. Prints color-coded next steps and service URLs

### After Running the Script

```bash
# Step 1 - Activate docker group without logging out
newgrp docker

# Step 2 - Insert your API keys
nano .env

# Step 3 - Launch the full cluster
docker compose up --build -d
```

### Azure NSG Inbound Rules

Open the following ports in the VM's Network Security Group on the Azure portal:

| Port | Service |
|------|---------|
| 8000 | FastAPI Backend |
| 8501 | Streamlit Frontend |
| 5000 | MLflow Tracking UI |

