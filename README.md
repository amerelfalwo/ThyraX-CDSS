# 🦋 ThyraX CDSS — Clinical Decision Support System for Thyroid Cancer

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.135+-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-Agent-1C3C3C?logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/ONNX-Runtime-blue?logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/ChromaDB-RAG-orange" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker" alt="Docker">
</div>

<p align="center">
  <em>An agentic, multi-phase AI system that assists physicians in diagnosing thyroid cancer — from lab assessment to ultrasound analysis and AI-guided clinical reasoning.</em>
</p>

---

## ⚕️ Project Overview

**ThyraX** is a comprehensive **Clinical Decision Support System (CDSS)** for Thyroid Cancer diagnosis, built as a graduation capstone project. It strictly follows the **"Human-in-the-Loop"** principle — every AI output is clearly a recommendation to assist the physician, never a replacement for clinical judgment.

The system combines:
- 🔬 An **XGBoost disease model** that interprets lab results (TSH, T3, T4)
- 🤖 A **medically-driven agentic router** that decides the next clinical step
- 🖼️ An **ONNX image pipeline** for ultrasound tumor segmentation & TI-RADS classification
- 🧠 A **LangChain ReAct AI Agent** with RAG over 18 medical PDFs and SQL patient history

---

## 🏗️ System Architecture — 4 Phases

```
Phase 1: Lab Input (TSH, T3, T4) ──→ XGBoost Disease Model
                                              │
Phase 2:                         ┌───────────▼────────────┐
         Agentic Router ─────────┤ hyperthyroid → I-123 scan│
                                 │ hypo/normal + nodule     │──→ Phase 3
                                 │   → Upload Ultrasound    │
                                 │ normal, no nodule        │
                                 │   → Routine Follow-up    │
                                 └──────────────────────────┘

Phase 3: Ultrasound Image ──→ ONNX Segmentation ──→ Nodule ROI
                                                        │
                                              ONNX Classification
                                             (Benign / Malignant)
                                              + TI-RADS Stage (TR1-TR5)

Phase 4: Doctor's Question ──→ LangChain ReAct Agent
                               ├── Tool 1: RAG (ATA guidelines, TI-RADS, PubMed)
                               ├── Tool 2: SQL Patient History
                               └── Tool 3: Similar Cases Search
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/predict/disease` | Raw XGBoost disease model prediction |
| `POST` | `/predict/image` | Ultrasound image segmentation + classification |
| `POST` | `/assess/clinical` | Full CDSS: Phase 1 & 2 (labs → routing) |
| `POST` | `/agent/chat` | AI medical assistant with RAG + patient history |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Python 3.13 |
| **Disease Model** | XGBoost (Joblib compressed) |
| **Image Pipeline** | ONNX Runtime, OpenCV, NumPy |
| **AI Agent** | LangChain, Groq (`llama-3.3-70b-versatile`) |
| **RAG Knowledge Base** | ChromaDB, `all-MiniLM-L6-v2` embeddings |
| **Patient Database** | SQLAlchemy Async, SQLite / PostgreSQL |
| **Package Manager** | `uv` (Rust-based, ultra-fast) |
| **Containerization** | Docker, Docker Compose |

---

## 💻 Local Setup

```bash
git clone https://github.com/your-username/thyrax-cdss.git
cd thyrax-cdss
```

### 1. Configure Environment
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=sqlite+aiosqlite:///./thyrax.db
```

### 2. Run with `uv` (Development)
```bash
uv sync
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run with Docker (Production)
```bash
docker compose up --build
```

> **Note:** The ONNX models (`models/compressed/`) and ChromaDB vector store (`data/`) are excluded from git due to size. Place them in the correct directories before running.

Swagger UI: **http://localhost:8000/docs**

---

## 📁 Project Structure

```
thyrax-cdss/
├── main.py                    # FastAPI entry point
├── app/
│   ├── core/
│   │   ├── config.py          # Centralized settings (Pydantic)
│   │   └── database.py        # Async SQLAlchemy engine
│   ├── disease/
│   │   ├── model.py           # XGBoost predict_thyroid()
│   │   └── schema.py          # ThyroidInput schema
│   ├── segmentation/
│   │   └── model.py           # ONNX seg + classification pipeline
│   ├── models/
│   │   └── patient.py         # Patient, Visit, ImagingResult ORM
│   ├── agent/
│   │   ├── agent.py           # LangChain ReAct agent + guardrails
│   │   └── tools.py           # 3 tools: RAG, SQL, Similar Cases
│   └── routers/
│       ├── clinical.py        # /assess/clinical (Phase 1 & 2)
│       └── chat.py            # /agent/chat (Phase 4)
├── models/compressed/         # ONNX + Joblib models (gitignored)
├── data/
│   ├── documents/             # 18 medical PDFs for RAG
│   │   ├── Primary Guidelines/    # ATA 2025 & 2015 guidelines
│   │   ├── Radiology & TI-RADS/   # ACR TI-RADS papers
│   │   ├── PubMed Central/        # Research papers
│   │   └── AI & Intelligent Diagnosis/
│   └── (vector_store/)        # ChromaDB index (gitignored)
├── test_agent.py              # Agent integration test script
├── pyproject.toml
└── dockerfile / docker-compose.yml
```

---

## 🧪 Testing the Agent

```bash
# Make sure the server is running, then:
uv run python test_agent.py
```

Expected output:
```
✅ Server is running: ThyraX CDSS API v2.0

🔧 Tools Invoked: search_medical_guidelines, query_patient_history

🩺 Agent Response:
  According to the ATA guidelines, the recommended management plan
  for a hypothyroid patient with a palpable nodule includes
  fine-needle aspiration (FNA)...

Status: SUCCESS | HTTP: 200
```

---

## 🔒 Clinical Guardrails

Every AI response includes a mandatory CDSS disclaimer:

> ⚕️ *This AI-generated analysis is provided as a clinical decision support tool only. It does not constitute medical advice or a definitive diagnosis. All clinical decisions must be made by a qualified healthcare professional.*

---

## 👥 Contributors

- **[Your Name]** — AI / Backend Engineer

*Designed and developed as a Graduation Capstone Project.*
