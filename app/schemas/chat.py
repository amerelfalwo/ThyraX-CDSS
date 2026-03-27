from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Medical question for the AI assistant")
    patient_id: Optional[int] = Field(None, description="Optional patient ID for history-aware queries")

class ChatResponse(BaseModel):
    status: str
    query: str
    response: str
    tools_used: List[str]
    disclaimer: str = (
        "⚕️ CDSS Disclaimer: This AI-generated analysis is provided as a "
        "clinical decision support tool only. It does not constitute medical "
        "advice or a definitive diagnosis. All clinical decisions must be made "
        "by a qualified healthcare professional."
    )
