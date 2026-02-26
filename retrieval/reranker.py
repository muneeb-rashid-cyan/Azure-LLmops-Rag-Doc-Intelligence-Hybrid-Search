"""
reranker.py
-----------
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Why rerank?
  Bi-encoder (embeddings) encodes query and chunk SEPARATELY.
  Fast but imprecise — it can't model query-chunk interaction.

  Cross-encoder reads BOTH query + chunk TOGETHER.
  Slower but much more precise — gives a true relevance score.

Pipeline:
  Hybrid search returns top 20 candidates (high recall)
  Cross-encoder reranks them → keep top 5 (high precision)
"""

from dataclasses import dataclass
from loguru import logger

from retrieval.hybrid_search import SearchResult


@dataclass
class RankedResult:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_type: str
    content: str
    rerank_score: float
    rrf_score: float
    final_rank: int


_reranker_model = None


def _get_reranker(model_name: str):
    """Lazy load the cross-encoder model (downloads on first call)."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {model_name}")
            _reranker_model = CrossEncoder(model_name)
            logger.info("Cross-encoder loaded")
        except Exception as e:
            logger.warning(f"Cross-encoder load failed: {e}")
            _reranker_model = None
    return _reranker_model


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RankedResult]:
    """
    Rerank search results using cross-encoder.

    Args:
        query:      original user question
        results:    candidates from hybrid search
        top_k:      number to keep after reranking
        model_name: HuggingFace cross-encoder model

    Returns:
        top_k results sorted by rerank_score descending
    """
    if not results:
        return []

    reranker = _get_reranker(model_name)

    if reranker is None:
        # Fallback: return top_k by RRF score
        logger.warning("Reranker unavailable — using RRF scores as fallback")
        sorted_results = sorted(results, key=lambda x: x.rrf_score, reverse=True)[:top_k]
        return [
            RankedResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                doc_name=r.doc_name,
                page_number=r.page_number,
                chunk_type=r.chunk_type,
                content=r.content,
                rerank_score=r.rrf_score,
                rrf_score=r.rrf_score,
                final_rank=i + 1,
            )
            for i, r in enumerate(sorted_results)
        ]

    # Build query-passage pairs for cross-encoder
    pairs = [(query, r.content) for r in results]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        scores = [r.rrf_score for r in results]

    # Attach scores and sort
    scored = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    ranked = [
        RankedResult(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            doc_name=r.doc_name,
            page_number=r.page_number,
            chunk_type=r.chunk_type,
            content=r.content,
            rerank_score=float(score),
            rrf_score=r.rrf_score,
            final_rank=i + 1,
        )
        for i, (r, score) in enumerate(scored)
    ]

    logger.info(
        f"Reranked {len(results)} → {len(ranked)} results | "
        f"top score: {ranked[0].rerank_score:.4f}"
    )
    return ranked