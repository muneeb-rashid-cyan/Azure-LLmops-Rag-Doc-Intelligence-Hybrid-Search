"""
pipeline.py (retrieval)
-----------------------
Full advanced retrieval pipeline:

  1. Query Rewriting   → 3 alternative phrasings
  2. Hybrid Search     → vector + BM25 + RRF on all phrasings
  3. Cross-Encoder     → rerank top 20 → keep top 5
  4. Context Compress  → extract only relevant sentences
"""

from dataclasses import dataclass
from loguru import logger

from config import get_settings
from retrieval.query_rewriter import rewrite_query
from retrieval.hybrid_search import multi_query_hybrid_search
from retrieval.reranker import rerank
from retrieval.context_compressor import compress_context


@dataclass
class RetrievalResult:
    original_query: str
    rewritten_queries: list[str]
    context: list[dict]        # compressed, citation-ready chunks
    total_candidates: int      # chunks before reranking
    total_after_rerank: int    # chunks after reranking


def run_retrieval_pipeline(query: str) -> RetrievalResult:
    """
    Run the full advanced retrieval pipeline for a user query.

    Args:
        query: raw user question

    Returns:
        RetrievalResult with context ready for GPT-4.1
    """
    settings = get_settings()

    logger.info(f"Retrieval pipeline starting for: '{query[:80]}'")

    # ── Step 1: Query Rewriting ───────────────────────────────
    rewrites = rewrite_query(query, n=settings.query_rewrite_count)
    all_queries = [query] + rewrites
    logger.info(f"Step 1 complete: {len(all_queries)} query variants")

    # ── Step 2: Multi-Query Hybrid Search ─────────────────────
    candidates = multi_query_hybrid_search(
        queries=all_queries,
        top_k=settings.top_k_retrieval,
    )
    logger.info(f"Step 2 complete: {len(candidates)} candidates retrieved")

    # ── Step 3: Cross-Encoder Reranking ───────────────────────
    reranked = rerank(
        query=query,
        results=candidates,
        top_k=settings.top_k_rerank,
        model_name=settings.reranker_model,
    )
    logger.info(f"Step 3 complete: {len(reranked)} chunks after reranking")

    # ── Step 4: Context Compression ───────────────────────────
    context = compress_context(query, reranked, max_tokens=2000)
    logger.info(f"Step 4 complete: {len(context)} compressed context chunks")

    return RetrievalResult(
        original_query=query,
        rewritten_queries=rewrites,
        context=context,
        total_candidates=len(candidates),
        total_after_rerank=len(reranked),
    )