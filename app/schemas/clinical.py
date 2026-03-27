from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ClinicalAssessmentRequest(BaseModel):
    patient_id: int = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    on_thyroxine: int = Field(..., ge=0, le=1, description="Is patient on thyroxine? 1=yes, 0=no")
    thyroid_surgery: int = Field(..., ge=0, le=1, description="History of thyroid surgery? 1=yes, 0=no")
    query_hyperthyroid: int = Field(..., ge=0, le=1, description="Clinical suspicion of hyperthyroidism? 1=yes, 0=no")
    TSH: float = Field(..., ge=0, description="Thyroid Stimulating Hormone (µIU/mL)")
    T3: float = Field(..., ge=0, description="Triiodothyronine (ng/mL)")
    TT4: float = Field(..., ge=0, description="Total T4 (µg/dL)")
    FTI: float = Field(..., ge=0, description="Free Thyroxine Index")
    T4U: float = Field(..., ge=0, description="T4 Uptake")
    nodule_present: bool = Field(..., description="Palpable nodule detected? true=yes, false=no")

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
