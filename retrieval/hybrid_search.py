"""
hybrid_search.py
----------------
Hybrid search = vector search + BM25 keyword search combined
via Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Vector search: finds semantically similar chunks
    ("revenue figures" matches "financial performance metrics")
  - BM25: finds exact keyword matches
    ("Q3 2024" matches "Q3 2024" exactly)
  - RRF combines both ranked lists into one better list

RRF formula:
  score(d) = Σ 1 / (rank_i(d) + k)
  where k=60 is a smoothing constant
"""

from dataclasses import dataclass
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from loguru import logger

from config import get_settings
from ingestion.embedder import embed_query


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_type: str
    content: str
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0


def _get_client() -> SearchClient:
    s = get_settings()
    return SearchClient(
        endpoint=s.azure_search_endpoint,
        index_name=s.azure_search_index,
        credential=AzureKeyCredential(s.azure_search_key),
    )


def vector_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """Pure vector search using query embedding."""
    client    = _get_client()
    embedding = embed_query(query)

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "doc_id", "doc_name", "page_number", "chunk_type", "content"],
        top=top_k,
    )

    return [
        SearchResult(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            doc_name=r["doc_name"],
            page_number=r["page_number"],
            chunk_type=r["chunk_type"],
            content=r["content"],
        )
        for r in results
    ]


def keyword_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """Pure BM25 keyword search."""
    client = _get_client()

    results = client.search(
        search_text=query,
        query_type="semantic" if False else "simple",
        select=["id", "doc_id", "doc_name", "page_number", "chunk_type", "content"],
        top=top_k,
    )

    return [
        SearchResult(
            chunk_id=r["id"],
            doc_id=r["doc_id"],
            doc_name=r["doc_name"],
            page_number=r["page_number"],
            chunk_type=r["chunk_type"],
            content=r["content"],
        )
        for r in results
    ]


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """
    Combine vector and keyword ranked lists using RRF.
    RRF score = 1/(rank_v + k) + 1/(rank_k + k)
    Higher score = more relevant.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (rank + k)
        result.vector_rank = rank
        chunk_map[cid] = result

    for rank, result in enumerate(keyword_results, start=1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (rank + k)
        result.keyword_rank = rank
        if cid not in chunk_map:
            chunk_map[cid] = result
        else:
            chunk_map[cid].keyword_rank = rank

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        result = chunk_map[cid]
        result.rrf_score = scores[cid]
        fused.append(result)

    return fused


def hybrid_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """
    Full hybrid search: vector + keyword → RRF fusion.
    """
    logger.debug(f"Hybrid search: '{query[:60]}...' top_k={top_k}")

    vec_results = vector_search(query, top_k)
    kw_results  = keyword_search(query, top_k)
    fused       = reciprocal_rank_fusion(vec_results, kw_results)

    logger.debug(
        f"Hybrid search returned {len(fused)} results "
        f"(vec={len(vec_results)}, kw={len(kw_results)})"
    )
    return fused[:top_k]


def multi_query_hybrid_search(
    queries: list[str],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Run hybrid search for multiple query phrasings (from query rewriter),
    then fuse all results with RRF.

    This is the core retrieval step after query rewriting.
    """
    all_results: list[SearchResult] = []

    for query in queries:
        results = hybrid_search(query, top_k)
        all_results.extend(results)

    # De-duplicate by chunk_id keeping highest RRF score
    best: dict[str, SearchResult] = {}
    for r in all_results:
        if r.chunk_id not in best or r.rrf_score > best[r.chunk_id].rrf_score:
            best[r.chunk_id] = r

    # Re-sort by rrf_score
    merged = sorted(best.values(), key=lambda x: x.rrf_score, reverse=True)

    logger.info(
        f"Multi-query hybrid search: {len(queries)} queries → "
        f"{len(merged)} unique results"
    )
    return merged[:top_k]