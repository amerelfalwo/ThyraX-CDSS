from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ExtractedLabsResponse(BaseModel):
    TSH: Optional[float] = None
    T3: Optional[float] = None
    TT4: Optional[float] = None
    FTI: Optional[float] = None
    T4U: Optional[float] = None
    test_date: Optional[str] = None
