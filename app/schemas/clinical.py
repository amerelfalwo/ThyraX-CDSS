from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ClinicalAssessmentRequest(BaseModel):
    patient_id: int
    TSH: float
    T3: float
    TT4: float
    FTI: float
    T4U: float
    nodule_present: bool

class ClinicalAssessmentResponse(BaseModel):
    status: str
    patient_id: int
    
    # ── Disease model output ──
    functional_status: str
    probabilities: Dict[str, float]

    # ── Agentic routing ──
    risk_level: str
    clinical_recommendation: str
    next_step: str
    next_step_details: Dict[str, Any]

    disclaimer: str = (
        "⚕️ DISCLAIMER: This AI-generated assessment is a clinical decision "
        "support tool ONLY. The final diagnosis and treatment decisions must "
        "be made by a qualified physician. This system does NOT replace "
        "professional medical judgment."
    )
