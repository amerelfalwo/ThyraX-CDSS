"""
ThyraX CDSS — Unified Clinical Decision Support System API.

Combines:
  - Disease prediction (clinical tabular data → XGBoost)
  - Image pipeline (Ultrasound → ONNX Segmentation → Classification)
  - Agentic clinical routing (Phase 1 & 2)
  - LangChain AI Agent with RAG + SQL tools (Phase 4)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.disease.schema import ThyroidInput
from app.disease.model import predict_thyroid
from app.core.database import init_db
from app.routers import clinical, chat, labs, image, patient


# ═══════════════════════════════════════════════════════════════
# Lifespan — startup & shutdown
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup."""
    await init_db()
    print("✅ Database tables created / verified.")
    yield
    print("🛑 ThyraX CDSS shutting down.")


# ═══════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="ThyraX CDSS API",
    description=(
        "Comprehensive Clinical Decision Support System for Thyroid Cancer Diagnosis.\n\n"
        "## Endpoints\n"
        "- **`/predict/disease`** — Raw disease model prediction\n"
        "- **`/predict/image`** — Ultrasound segmentation + classification\n"
        "- **`/assess/clinical`** — Full CDSS workflow (Phase 1 & 2)\n"
        "- **`/agent/chat`** — AI medical assistant with RAG tools (Phase 4)\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# Register Routers
# ═══════════════════════════════════════════════════════════════

app.include_router(clinical.router)
app.include_router(chat.router)


app.include_router(labs.router)
app.include_router(patient.router)
app.include_router(image.router)


# ═══════════════════════════════════════════════════════════════
# Legacy Disease Model Endpoint
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ThyraX AI Engine",
    }


@app.post("/predict/disease")
def predict_clinical_disease(data: ThyroidInput):
    try:
        input_data = data.model_dump()
        result = predict_thyroid(input_data)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")