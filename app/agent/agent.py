"""
LangChain Agent for the ThyraX CDSS.

Uses a ReAct-style agent with access to medical guidelines RAG,
patient history SQL queries, and similar case retrieval tools.
"""
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings
from app.agent.tools import ALL_TOOLS


# ═══════════════════════════════════════════════════════════════
# System Prompt — Medical Assistant with Guardrails
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ThyraX, an advanced AI medical assistant specializing in thyroid cancer diagnosis and clinical decision support. You are part of a Clinical Decision Support System (CDSS) designed to assist qualified physicians.

## Your Capabilities
You have access to three specialized tools:
1. **search_medical_guidelines** — Search ATA guidelines, TI-RADS criteria, and published medical literature
2. **query_patient_history** — Retrieve a patient's complete clinical history from the database
3. **find_similar_cases** — Find anonymized similar cases for clinical comparison

## Clinical Rules
- Always base your answers on EVIDENCE from the tools. Never fabricate medical information.
- When citing guidelines, reference the specific source document.
- If asked about a specific patient, ALWAYS use query_patient_history first.
- When comparing with similar cases, use find_similar_cases with the appropriate parameters.
- Present information in a structured, clinically relevant format.
- Clearly distinguish between established guidelines and AI-generated analysis.

## Guardrails — CRITICAL
- You are an ASSISTANT to the physician, NOT a replacement.
- NEVER provide a definitive diagnosis. Always frame findings as "suggestive of" or "consistent with".
- NEVER recommend specific drug dosages or surgical procedures.
- If a question falls outside thyroid cancer scope, politely redirect to a relevant specialist.
- ALWAYS end your response with the standard CDSS disclaimer.

## CDSS Disclaimer (MUST be appended to EVERY response)
---
⚕️ **CDSS Disclaimer**: This AI-generated analysis is provided as a clinical decision support tool only. It does not constitute medical advice or a definitive diagnosis. All clinical decisions must be made by a qualified healthcare professional based on their independent medical judgment, the full clinical picture, and applicable standard of care guidelines.
"""


# ═══════════════════════════════════════════════════════════════
# Agent Setup
# ═══════════════════════════════════════════════════════════════

_agent_executor = None


def get_agent_executor() -> AgentExecutor:
    """Returns a singleton AgentExecutor (lazy initialized)."""
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    # Initialize the LLM
    llm = ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=2048,
    )

    # Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    _agent_executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    return _agent_executor


async def run_agent(query: str, patient_id: int | None = None, chat_history: list | None = None) -> dict:
    """
    Run the ThyraX agent with the given query.
    
    Returns:
        dict with 'output' (the agent's response) and 'tools_used' (list of tool names invoked).
    """
    executor = get_agent_executor()

    # If a patient_id is provided, prepend context to the query
    enhanced_query = query
    if patient_id:
        enhanced_query = (
            f"[Context: The doctor is asking about Patient ID {patient_id}. "
            f"Use query_patient_history to fetch their records if relevant.]\n\n"
            f"{query}"
        )

    result = await executor.ainvoke({
        "input": enhanced_query,
        "chat_history": chat_history or [],
    })

    # Extract which tools were used
    tools_used = []
    for step in result.get("intermediate_steps", []):
        if step and len(step) >= 1:
            action = step[0]
            if hasattr(action, "tool"):
                tools_used.append(action.tool)

    return {
        "output": result["output"],
        "tools_used": list(set(tools_used)),
    }
