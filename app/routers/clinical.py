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
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.disease.model import predict_thyroid
from app.core.database import get_db
from app.models.patient import Patient, Visit

router = APIRouter(prefix="/assess", tags=["Clinical Assessment"])


# ═══════════════════════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════════════════════

class ClinicalAssessmentRequest(BaseModel):
    """Labs + clinical context submitted by the doctor."""
    # ── Patient info ──
    patient_id: Optional[int] = Field(None, description="Existing patient ID. If None, a new patient record is created.")
    patient_name: str = Field(..., min_length=1, description="Patient name")
    age: int = Field(..., ge=0, le=120)
    gender: Optional[str] = Field(None, description="M / F / Other")

    # ── Lab results (same features the disease model expects) ──
    TT4: float = Field(..., ge=0, description="Total T4 (µg/dL)")
    TSH: float = Field(..., ge=0, description="TSH (µIU/mL)")
    T3: float = Field(..., ge=0, description="T3 (ng/mL)")
    FTI: float = Field(..., ge=0, description="Free Thyroxine Index")
    T4U: float = Field(..., ge=0, description="T4 Uptake")

    # ── Clinical flags ──
    on_thyroxine: int = Field(0, ge=0, le=1)
    thyroid_surgery: int = Field(0, ge=0, le=1)
    query_hyperthyroid: int = Field(0, ge=0, le=1)
    nodule_present: bool = Field(False, description="Physical examination: is a palpable nodule present?")

    # ── Optional doctor notes ──
    notes: Optional[str] = None


class ClinicalAssessmentResponse(BaseModel):
    status: str
    patient_id: int
    visit_id: int

    # ── Disease model output ──
    functional_status: str
    probabilities: dict

    # ── Agentic routing ──
    risk_level: str
    clinical_recommendation: str
    next_step: str
    next_step_details: dict

    disclaimer: str = (
        "⚕️ DISCLAIMER: This AI-generated assessment is a clinical decision "
        "support tool ONLY. The final diagnosis and treatment decisions must "
        "be made by a qualified physician. This system does NOT replace "
        "professional medical judgment."
    )


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
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/clinical", response_model=ClinicalAssessmentResponse)
async def assess_clinical(req: ClinicalAssessmentRequest, db: AsyncSession = Depends(get_db)):
    """
    Phase 1: Run disease model  →  Phase 2: Agentic routing.
    Persists the visit record for future history queries.
    """
    # ── 1. Run the disease prediction model ──
    disease_input = {
        "TT4": req.TT4,
        "TSH": req.TSH,
        "T3": req.T3,
        "FTI": req.FTI,
        "T4U": req.T4U,
        "age": req.age,
        "on_thyroxine": req.on_thyroxine,
        "thyroid_surgery": req.thyroid_surgery,
        "query_hyperthyroid": req.query_hyperthyroid,
    }

    try:
        prediction = predict_thyroid(disease_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease model error: {e}")

    functional_status: str = prediction["label"]
    probabilities: dict = prediction["probabilities"]

    # ── 2. Agentic routing ──
    routing = _route_clinical_decision(functional_status, req.nodule_present, probabilities)

    # ── 3. Persist patient + visit ──
    try:
        if req.patient_id:
            patient = await db.get(Patient, req.patient_id)
            if not patient:
                raise HTTPException(status_code=404, detail=f"Patient ID {req.patient_id} not found")
        else:
            patient = Patient(name=req.patient_name, age=req.age, gender=req.gender)
            db.add(patient)
            await db.flush()  # get the auto-generated ID

        visit = Visit(
            patient_id=patient.id,
            tsh=req.TSH,
            t3=req.T3,
            t4=req.TT4,    # TT4 stored as t4
            tt4=req.TT4,
            fti=req.FTI,
            t4u=req.T4U,
            on_thyroxine=bool(req.on_thyroxine),
            thyroid_surgery=bool(req.thyroid_surgery),
            query_hyperthyroid=bool(req.query_hyperthyroid),
            nodule_present=req.nodule_present,
            functional_status=functional_status,
            clinical_recommendation=routing["recommendation"],
            next_step=routing["next_step"],
            notes=req.notes,
        )
        db.add(visit)
        await db.commit()
        await db.refresh(visit)
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # ── 4. Return response ──
    return ClinicalAssessmentResponse(
        status="success",
        patient_id=patient.id,
        visit_id=visit.id,
        functional_status=functional_status,
        probabilities=probabilities,
        risk_level=routing["risk_level"],
        clinical_recommendation=routing["recommendation"],
        next_step=routing["next_step"],
        next_step_details=routing["next_step_details"],
    )
