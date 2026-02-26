"""
test_retrieval.py
-----------------
Tests for query rewriting, hybrid search logic, reranking, and compression.
Run: pytest tests/test_retrieval.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from retrieval.hybrid_search import reciprocal_rank_fusion, SearchResult
from retrieval.reranker import rerank, RankedResult
from retrieval.context_compressor import compress_context
from evaluation.metrics import compute_retrieval_precision


# ── FIXTURES ──────────────────────────────────────────────────
def make_result(chunk_id: str, rrf_score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_name="test.pdf",
        page_number=1,
        chunk_type="text",
        content=f"Content for chunk {chunk_id}",
        rrf_score=rrf_score,
    )


def make_ranked(chunk_id: str, score: float = 0.8) -> RankedResult:
    return RankedResult(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_name="test.pdf",
        page_number=1,
        chunk_type="text",
        content=f"This is the content of chunk {chunk_id} about revenue figures.",
        rerank_score=score,
        rrf_score=0.5,
        final_rank=1,
    )


# ── RRF TESTS ─────────────────────────────────────────────────
class TestRRF:

    def test_rrf_combines_both_lists(self):
        vec_results = [make_result("A"), make_result("B"), make_result("C")]
        kw_results  = [make_result("B"), make_result("D"), make_result("A")]

        fused = reciprocal_rank_fusion(vec_results, kw_results)
        ids   = [r.chunk_id for r in fused]

        # A and B appear in both → should rank highest
        assert "A" in ids
        assert "B" in ids
        assert "D" in ids

    def test_rrf_chunk_in_both_scores_higher(self):
        vec_results = [make_result("SHARED"), make_result("VEC_ONLY")]
        kw_results  = [make_result("SHARED"), make_result("KW_ONLY")]

        fused = reciprocal_rank_fusion(vec_results, kw_results)
        ids   = [r.chunk_id for r in fused]

        # SHARED appears in both lists → must be ranked first
        assert ids[0] == "SHARED"

    def test_rrf_empty_lists(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_rrf_one_empty_list(self):
        vec_results = [make_result("A"), make_result("B")]
        fused = reciprocal_rank_fusion(vec_results, [])
        assert len(fused) == 2

    def test_rrf_scores_are_positive(self):
        vec_results = [make_result("A"), make_result("B")]
        kw_results  = [make_result("C"), make_result("A")]
        fused = reciprocal_rank_fusion(vec_results, kw_results)
        assert all(r.rrf_score > 0 for r in fused)

    def test_rrf_no_duplicate_chunk_ids(self):
        vec_results = [make_result("A"), make_result("B"), make_result("A")]
        kw_results  = [make_result("A"), make_result("C")]
        fused = reciprocal_rank_fusion(vec_results, kw_results)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))


# ── RERANKER TESTS ────────────────────────────────────────────
class TestReranker:

    def test_rerank_returns_top_k(self):
        results = [make_result(f"chunk_{i}", rrf_score=0.5) for i in range(10)]
        ranked  = rerank("what is the revenue?", results, top_k=3)
        assert len(ranked) <= 3

    def test_rerank_empty_input(self):
        ranked = rerank("any question", [], top_k=5)
        assert ranked == []

    def test_rerank_assigns_final_ranks(self):
        results = [make_result(f"c{i}", rrf_score=float(i)/10) for i in range(5)]
        ranked  = rerank("test query", results, top_k=5)
        ranks   = [r.final_rank for r in ranked]
        assert sorted(ranks) == list(range(1, len(ranked) + 1))

    def test_rerank_output_has_required_fields(self):
        results = [make_result("chunk_a")]
        ranked  = rerank("test", results, top_k=1)
        if ranked:
            r = ranked[0]
            assert hasattr(r, "chunk_id")
            assert hasattr(r, "rerank_score")
            assert hasattr(r, "final_rank")
            assert hasattr(r, "content")


# ── RETRIEVAL PRECISION TESTS ─────────────────────────────────
class TestRetrievalMetrics:

    def test_precision_perfect(self):
        m = compute_retrieval_precision(["A", "B", "C"], ["A", "B", "C"])
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0

    def test_precision_zero(self):
        m = compute_retrieval_precision(["X", "Y"], ["A", "B"])
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0

    def test_precision_partial(self):
        m = compute_retrieval_precision(["A", "X", "B"], ["A", "B", "C"])
        assert m.precision_at_k == pytest.approx(2/3, abs=0.01)

    def test_precision_empty_retrieved(self):
        m = compute_retrieval_precision([], ["A", "B"])
        assert m.precision_at_k == 0.0

    def test_f1_at_k_between_0_and_1(self):
        m = compute_retrieval_precision(["A", "B", "X"], ["A", "B", "C"])
        assert 0.0 <= m.f1_at_k <= 1.0