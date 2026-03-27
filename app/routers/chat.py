"""
Phase 4 — AI Agent Chat Endpoint.

POST /agent/chat
  - Accepts a medical query and optional patient_id
  - Invokes the LangChain agent with RAG, SQL, and similar case tools
  - Returns AI response with CDSS disclaimer
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List

from app.agent.agent import run_agent
from app.core.security import verify_internal_api_key
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/agent", tags=["AI Agent"], dependencies=[Depends(verify_internal_api_key)])


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
