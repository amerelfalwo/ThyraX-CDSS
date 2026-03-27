"""
Phase 1 & 2 — Clinical Assessment & Medically-Driven Agentic Routing.

POST /assess/clinical
  - Runs the disease model to determine functional status
  - Routes the patient based on clinical rules:
      hyperthyroid  →  recommend Radionuclide Scan
      hypothyroid/normal + nodule  →  flag cancer suspicion, request Ultrasound
      normal + no nodule  →  routine follow-up
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

import pandas as pd
from app.core.mlops import load_production_model
from app.core.database import get_db
from app.core.security import verify_internal_api_key

from app.schemas.clinical import ClinicalAssessmentRequest, ClinicalAssessmentResponse

router = APIRouter(prefix="/clinical", tags=["Clinical Assessment"], dependencies=[Depends(verify_internal_api_key)])


# ═══════════════════════════════════════════════════════════════
# Agentic Routing Logic
# ═══════════════════════════════════════════════════════════════

def _route_clinical_decision(functional_status: str, nodule_present: bool, probabilities: dict) -> dict:
    """
    Medically-driven routing based on the disease model output.

    Returns a dict with: risk_level, recommendation, next_step, next_step_details
    """
    if functional_status == "hyperthyroid":
        return {
            "risk_level": "moderate",
            "recommendation": (
                "Patient shows signs of HYPERTHYROIDISM. "
                "The recommended next step is a Radionuclide (Iodine-123) Scan "
                "to evaluate for autonomously functioning thyroid nodules (Hot Nodules). "
                "Hot nodules are RARELY malignant (<1% risk). "
                "Cancer workup is NOT immediately indicated unless cold nodules are identified on the scan."
            ),
            "next_step": "radionuclide_scan",
            "next_step_details": {
                "action": "Order Radionuclide Scan (I-123 uptake)",
                "rationale": "Differentiate hot vs. cold nodules in hyperthyroid state",
                "cancer_pipeline_triggered": False,
                "urgency": "routine",
            },
        }

    if functional_status in ("hypothyroid", "normal") and nodule_present:
        risk = "high" if functional_status == "hypothyroid" else "elevated"
        return {
            "risk_level": risk,
            "recommendation": (
                f"Patient is {'HYPOTHYROID' if functional_status == 'hypothyroid' else 'EUTHYROID (normal)'} "
                f"with a PALPABLE NODULE detected on physical examination. "
                f"Cold nodules in {'hypothyroid' if functional_status == 'hypothyroid' else 'euthyroid'} patients "
                f"carry a HIGHER malignancy risk (5-15%). "
                f"The recommended next step is a HIGH-RESOLUTION THYROID ULTRASOUND "
                f"to evaluate the nodule characteristics per ACR TI-RADS criteria."
            ),
            "next_step": "upload_ultrasound",
            "next_step_details": {
                "action": "Upload thyroid ultrasound image for AI analysis",
                "endpoint": "/predict/image",
                "rationale": "Evaluate cold nodule for malignancy using segmentation + classification pipeline",
                "cancer_pipeline_triggered": True,
                "urgency": "priority",
            },
        }

    # normal + no nodule
    return {
        "risk_level": "low",
        "recommendation": (
            "Patient thyroid function is NORMAL with NO palpable nodule detected. "
            "No immediate imaging is required. Recommend routine clinical follow-up "
            "with repeat thyroid function tests in 6-12 months, or sooner if symptoms develop."
        ),
        "next_step": "routine_followup",
        "next_step_details": {
            "action": "Schedule follow-up in 6-12 months",
            "rationale": "Normal function, no structural abnormality",
            "cancer_pipeline_triggered": False,
            "urgency": "routine",
        },
    }


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/assess", response_model=ClinicalAssessmentResponse)
async def assess_clinical(req: ClinicalAssessmentRequest, db: AsyncSession = Depends(get_db)):
    """
    Phase 1: Run disease model  →  Phase 2: Agentic routing.
    Persists the visit record for future history queries.
    """
    # ── 1. Run the disease prediction model ──
    disease_input = {
        "TSH": req.TSH,
        "T3": req.T3,
        "TT4": req.TT4,
        "FTI": req.FTI,
        "T4U": req.T4U,
        # Defaulting remaining disease parameters not provided by SaaS simplified endpoint
        "age": 50,
        "on_thyroxine": 0,
        "thyroid_surgery": 0,
        "query_hyperthyroid": 0,
    }

    try:
        model = load_production_model("thyrax_xgboost")
        LABEL_MAP = {0: 'normal', 1: 'hypothyroid', 2: 'hyperthyroid'}
        df = pd.DataFrame([disease_input])
        
        pred = int(model.predict(df)[0])
        probs = model.predict_proba(df)[0]
        prob_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}
        
        functional_status = LABEL_MAP[pred]
        probabilities = prob_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease model error: {e}")

    # ── 2. Agentic routing ──
    routing = _route_clinical_decision(functional_status, req.nodule_present, probabilities)

    # ── 3. Return response (DB saving decoupled) ──
    return ClinicalAssessmentResponse(
        status="success",
        patient_id=req.patient_id,
        functional_status=functional_status,
        probabilities=probabilities,
        risk_level=routing["risk_level"],
        clinical_recommendation=routing["recommendation"],
        next_step=routing["next_step"],
        next_step_details=routing["next_step_details"],
    )
