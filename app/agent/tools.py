"""
LangChain Tools for the ThyraX CDSS Agent.

Tool 1: search_medical_guidelines  –  RAG over ChromaDB (ATA guidelines, TI-RADS, etc.)
Tool 2: query_patient_history      –  SQL query against the patient history database
Tool 3: find_similar_cases         –  Vector search for anonymized similar patient cases
"""
import os
import numpy as np
from typing import Optional
from langchain_core.tools import tool
from sqlalchemy import select, create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.models.patient import Patient, Visit, ImagingResult

# Sync engine for use inside LangChain tools (avoids greenlet/asyncio conflicts)
_sync_db_url = settings.DATABASE_URL.replace("sqlite+aiosqlite", "sqlite")
_sync_engine = create_engine(_sync_db_url, connect_args={"check_same_thread": False})
SyncSession = sessionmaker(bind=_sync_engine, expire_on_commit=False)


# ═══════════════════════════════════════════════════════════════
# Shared: Embedding Manager & ChromaDB Client (lazy loaded)
# ═══════════════════════════════════════════════════════════════

_embedding_model = None
_chroma_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def _embed_query(query: str) -> list:
    model = _get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()


# ═══════════════════════════════════════════════════════════════
# Tool 1: Medical Guidelines RAG
# ═══════════════════════════════════════════════════════════════

@tool
def search_medical_guidelines(query: str) -> str:
    """
    Search the medical knowledge base containing ATA guidelines,
    TI-RADS criteria, PubMed papers, and thyroid cancer references.
    Use this tool when the doctor asks questions about medical protocols,
    treatment guidelines, staging criteria, or clinical recommendations.
    
    Args:
        query: The medical question to search for.
    
    Returns:
        Relevant excerpts from medical guidelines with source references.
    """
    try:
        client = _get_chroma_client()
        collection = client.get_collection(settings.CHROMA_GUIDELINES_COLLECTION)

        query_embedding = _embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant medical guidelines found for this query."

        # Format results with sources
        formatted = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            source = metadata.get("source", "Unknown source")
            # Extract just the filename for cleaner display
            source_name = os.path.basename(source) if source else "Unknown"
            formatted.append(
                f"[Source {i}: {source_name}]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Error searching medical guidelines: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 2: Patient History Query
# ═══════════════════════════════════════════════════════════════

@tool
def query_patient_history(patient_id: int) -> str:
    """
    Retrieve the complete clinical history for a specific patient,
    including all previous visits, lab results, functional status,
    and imaging results. Use this tool to compare current results
    with the patient's historical data and track disease progression.
    
    Args:
        patient_id: The unique identifier of the patient.
    
    Returns:
        A formatted summary of the patient's visit history.
    """
    try:
        with SyncSession() as db:
            patient = db.get(Patient, patient_id)
            if not patient:
                return f"No patient found with ID {patient_id}."

            stmt = (
                select(Visit)
                .where(Visit.patient_id == patient_id)
                .order_by(Visit.visit_date.desc())
            )
            visits = db.execute(stmt).scalars().all()

            if not visits:
                return f"Patient '{patient.name}' (ID: {patient_id}) has no recorded visits."

            lines = [
                f"═══ Patient History: {patient.name} ═══",
                f"Age: {patient.age} | Gender: {patient.gender or 'N/A'}",
                f"Total Visits: {len(visits)}",
                "",
            ]

            for v in visits:
                lines.append(f"── Visit #{v.id} ({v.visit_date.strftime('%Y-%m-%d') if v.visit_date else 'N/A'}) ──")
                lines.append(f"  Labs: TSH={v.tsh}, T3={v.t3}, T4={v.tt4}, FTI={v.fti}")
                lines.append(f"  Status: {v.functional_status or 'N/A'}")
                lines.append(f"  Nodule: {'Yes' if v.nodule_present else 'No'}")
                lines.append(f"  Recommendation: {v.clinical_recommendation or 'N/A'}")
                lines.append(f"  Next Step: {v.next_step or 'N/A'}")
                if v.imaging_result:
                    ir = v.imaging_result
                    lines.append(f"  Imaging: {ir.classification_label} | Confidence: {ir.confidence}% | TI-RADS: {ir.tirads_stage}")
                if v.notes:
                    lines.append(f"  Doctor Notes: {v.notes}")
                lines.append("")

            return "\n".join(lines)
    except Exception as e:
        return f"Error querying patient history: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 3: Similar Cases Retrieval
# ═══════════════════════════════════════════════════════════════

@tool
def find_similar_cases(
    functional_status: str,
    classification_label: Optional[str] = None,
    tirads_stage: Optional[str] = None,
) -> str:
    """
    Search the anonymized case database to find similar past patients.
    Useful for comparing the current patient's profile with historical
    outcomes. Can match on functional status, classification, and TI-RADS stage.
    
    Args:
        functional_status: The thyroid functional status (normal, hypothyroid, hyperthyroid).
        classification_label: Optional. The imaging classification (benign, malignant).
        tirads_stage: Optional. The TI-RADS stage (TR1-TR5).
    
    Returns:
        A summary of similar anonymized cases from the database.
    """
    try:
        # Build a descriptive query for vector search
        query_parts = [f"Patient with {functional_status} thyroid function"]
        if classification_label:
            query_parts.append(f"and {classification_label} nodule classification")
        if tirads_stage:
            query_parts.append(f"at TI-RADS stage {tirads_stage}")
        query_text = " ".join(query_parts)

        client = _get_chroma_client()

        # Try querying the similar_cases collection
        try:
            collection = client.get_collection(settings.CHROMA_SIMILAR_CASES_COLLECTION)
        except Exception:
            # If collection doesn't exist, fall back to guidelines with clinical query
            return (
                f"The similar cases database is not yet populated. "
                f"To build it, anonymized past patient records need to be ingested. "
                f"Query attempted: '{query_text}'"
            )

        query_embedding = _embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )

        if not results["documents"] or not results["documents"][0]:
            return f"No similar cases found matching: {query_text}"

        formatted = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            formatted.append(f"[Case {i}]\n{doc}")

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Error searching similar cases: {str(e)}"


# ── Export all tools for the agent ──
ALL_TOOLS = [search_medical_guidelines, query_patient_history, find_similar_cases]
