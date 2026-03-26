"""
Phase 4 — AI Agent Chat Endpoint.

POST /agent/chat
  - Accepts a medical query and optional patient_id
  - Invokes the LangChain agent with RAG, SQL, and similar case tools
  - Returns AI response with CDSS disclaimer
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.agent.agent import run_agent

router = APIRouter(prefix="/agent", tags=["AI Agent"])


# ═══════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """
    Chat with the ThyraX AI medical assistant.
    
    The agent can:
    - Search medical guidelines (ATA, TI-RADS, PubMed)
    - Query patient history from the database
    - Find similar anonymized cases for comparison
    """
    try:
        result = await run_agent(
            query=req.query,
            patient_id=req.patient_id,
        )

        return ChatResponse(
            status="success",
            query=req.query,
            response=result["output"],
            tools_used=result["tools_used"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}",
        )
