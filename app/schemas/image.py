from pydantic import BaseModel
from typing import Optional

class ImageValidationResponse(BaseModel):
    is_ultrasound: bool
    status: str = "success"
