from pydantic import BaseModel
from typing import Optional, List

class ImageValidationResponse(BaseModel):
    is_ultrasound: bool
    status: str = "success"

class ClassificationResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    tirads_stage: str

class ImageUrlsResponse(BaseModel):
    mask_url: str
    overlay_url: str
    roi_url: str

class ImagePredictionResponse(BaseModel):
    status: str
    bbox: Optional[List[int]] = None
    classification: Optional[ClassificationResponse] = None
    images: Optional[ImageUrlsResponse] = None
    message: Optional[str] = None

