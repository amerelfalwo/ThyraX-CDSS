from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime

class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    gender: Optional[str] = None

class VisitCreate(BaseModel):
    tsh: Optional[float] = None
    t3: Optional[float] = None
    t4: Optional[float] = None
    test_date: Optional[str] = None
    notes: Optional[str] = None
    functional_status: Optional[str] = None
    clinical_recommendation: Optional[str] = None
    next_step: Optional[str] = None
    imaging_result_id: Optional[int] = None

class VisitResponse(BaseModel):
    id: int
    visit_date: Optional[datetime.datetime]
    tsh: Optional[float]
    t3: Optional[float]
    t4: Optional[float]
    functional_status: Optional[str]
    clinical_recommendation: Optional[str]
    next_step: Optional[str]
    notes: Optional[str]
    imaging_result_id: Optional[int]

    class Config:
        from_attributes = True

class ImagingResponse(BaseModel):
    id: int
    image_path: Optional[str]
    classification_label: Optional[str]
    confidence: Optional[float]
    tirads_stage: Optional[str]
    processed_at: Optional[datetime.datetime]

    class Config:
        from_attributes = True

class PatientDemographics(BaseModel):
    id: int
    name: str
    age: int
    gender: Optional[str]
    created_at: Optional[datetime.datetime]

    class Config:
        from_attributes = True

class DashboardResponse(BaseModel):
    patient: PatientDemographics
    visits: List[VisitResponse]
    imaging: List[ImagingResponse]
    latest_clinical_recommendation: Optional[str] = None
    status: str = "success"
