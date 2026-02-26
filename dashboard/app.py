"""
app.py — FastAPI Backend
------------------------
Endpoints:
  POST /api/ingest          → upload + ingest a document
  POST /api/query           → ask a question (full RAG pipeline)
  GET  /api/documents       → list all ingested documents
  GET  /api/sessions        → list chat sessions
  GET  /api/health          → health check
  GET  /                    → dashboard HTML
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel

from config import get_settings
from ingestion.pipeline import run_ingestion_pipeline
from retrieval.pipeline import run_retrieval_pipeline
from generation.answer_generator import generate_answer
from evaluation.evaluator import evaluate_answer

app = FastAPI(title="Multi-Modal RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    evaluate: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    rewritten_queries: list[str]
    retrieval_stats: dict
    eval_scores: dict | None = None
    session_id: str
    query_id: str


# ── ROUTES ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the RAG pipeline."""
    allowed = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff"}
    suffix  = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, f"File type {suffix} not supported. Allowed: {allowed}")

    # Save upload
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())

    logger.info(f"Saved upload: {save_path}")

    try:
        result = run_ingestion_pipeline(save_path)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run full RAG pipeline: retrieve → generate → evaluate."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    query_id   = str(uuid.uuid4())

    # Step 1: Retrieval
    retrieval = run_retrieval_pipeline(req.question)

    # Step 2: Generation
    generated = generate_answer(
        query=req.question,
        context_chunks=retrieval.context,
        rewritten_queries=retrieval.rewritten_queries,
        retrieval_stats={
            "total_candidates": retrieval.total_candidates,
            "after_rerank":     retrieval.total_after_rerank,
            "context_chunks":   len(retrieval.context),
        },
    )

    # Step 3: Evaluation (optional)
    eval_scores = None
    if req.evaluate and retrieval.context:
        try:
            scores = evaluate_answer(
                query=req.question,
                answer=generated.answer,
                context_chunks=retrieval.context,
            )
            eval_scores = {
                "groundedness":          scores.groundedness,
                "relevance":             scores.relevance,
                "coherence":             scores.coherence,
                "overall":               scores.overall,
                "hallucination_detected": scores.hallucination_detected,
            }
        except Exception as e:
            logger.warning(f"Evaluation failed (non-critical): {e}")

    # Save to Cosmos
    try:
        from dashboard.cosmos_client import get_cosmos_client
        cosmos = get_cosmos_client()
        cosmos.save_session({
            "id":          query_id,
            "session_id":  session_id,
            "question":    req.question,
            "answer":      generated.answer,
            "citations":   generated.citations,
            "eval_scores": eval_scores,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.warning(f"Cosmos save failed (non-critical): {e}")

    return QueryResponse(
        answer=generated.answer,
        citations=generated.citations,
        rewritten_queries=generated.rewritten_queries,
        retrieval_stats=generated.retrieval_stats,
        eval_scores=eval_scores,
        session_id=session_id,
        query_id=query_id,
    )


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    try:
        from dashboard.cosmos_client import get_cosmos_client
        cosmos = get_cosmos_client()
        docs = cosmos.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception:
        return {"documents": [], "count": 0, "source": "demo"}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    template_path = Path("dashboard/templates/index.html")
    if template_path.exists():
        return HTMLResponse(template_path.read_text(encoding="utf-8")) 
    return HTMLResponse("<h1>RAG Dashboard — template not found</h1>")