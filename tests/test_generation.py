"""
test_generation.py
------------------
Tests for answer generation, citation extraction, and formatting.
Run: pytest tests/test_generation.py -v
"""

import pytest
from generation.citation_formatter import (
    extract_citations_from_answer,
    format_citations_markdown,
    format_answer_with_citations,
    Citation,
)
from evaluation.metrics import (
    compute_answer_f1,
    compute_token_overlap,
    compute_context_utilization,
    compute_hallucination_rate,
    aggregate_eval_scores,
)


# ── CITATION EXTRACTOR TESTS ──────────────────────────────────
class TestCitationExtractor:

    def test_extract_single_citation(self):
        answer  = "Revenue was $4.2B [report.pdf, Page 3]."
        chunks  = [{"doc_name": "report.pdf", "page_number": 3, "content": "Revenue $4.2B"}]
        clean, citations = extract_citations_from_answer(answer, chunks)

        assert len(citations) == 1
        assert citations[0].doc_name == "report.pdf"
        assert citations[0].page_number == 3
        assert "[1]" in clean
        assert "Page 3" not in clean  # original marker replaced

    def test_extract_multiple_citations(self):
        answer = (
            "Q3 revenue was $4.2B [report.pdf, Page 3]. "
            "Expenses were $3.1B [report.pdf, Page 5]."
        )
        chunks = [
            {"doc_name": "report.pdf", "page_number": 3, "content": "Q3 revenue"},
            {"doc_name": "report.pdf", "page_number": 5, "content": "expenses"},
        ]
        clean, citations = extract_citations_from_answer(answer, chunks)
        assert len(citations) == 2
        assert "[1]" in clean
        assert "[2]" in clean

    def test_duplicate_citations_merged(self):
        answer = (
            "Revenue [report.pdf, Page 3] and profit [report.pdf, Page 3] both grew."
        )
        chunks = [{"doc_name": "report.pdf", "page_number": 3, "content": "revenue profit"}]
        clean, citations = extract_citations_from_answer(answer, chunks)
        # Same source cited twice → only one citation object
        assert len(citations) == 1

    def test_no_citations_in_answer(self):
        answer = "There are no citations in this answer."
        clean, citations = extract_citations_from_answer(answer, [])
        assert citations == []
        assert clean == answer

    def test_citation_case_insensitive_page(self):
        answer = "See [doc.pdf, page 7] for details."
        chunks = [{"doc_name": "doc.pdf", "page_number": 7, "content": "details here"}]
        clean, citations = extract_citations_from_answer(answer, chunks)
        assert len(citations) == 1
        assert citations[0].page_number == 7

    def test_format_citations_markdown(self):
        citations = [
            Citation(index=1, doc_name="report.pdf", page_number=3, snippet="Revenue was $4B"),
            Citation(index=2, doc_name="report.pdf", page_number=5, snippet=None),
        ]
        md = format_citations_markdown(citations)
        assert "[1]" in md
        assert "[2]" in md
        assert "report.pdf" in md
        assert "Page 3" in md

    def test_format_answer_with_citations_dict(self):
        answer  = "The answer is here [doc.pdf, Page 2]."
        chunks  = [{"doc_name": "doc.pdf", "page_number": 2, "content": "the answer"}]
        result  = format_answer_with_citations(answer, chunks, output_format="dict")

        assert "clean_answer" in result
        assert "citations" in result
        assert result["citation_count"] == 1


# ── AGGREGATION METRICS TESTS ─────────────────────────────────
class TestAggregationMetrics:

    def test_context_utilization_full(self):
        answer = "revenue was four billion and expenses were three billion"
        chunks = [{"content": "revenue was four billion dollars reported"}]
        score = compute_context_utilization(answer, chunks)
        assert score > 0.0

    def test_context_utilization_no_chunks(self):
        score = compute_context_utilization("any answer", [])
        assert score == 0.0

    def test_hallucination_rate_all_true(self):
        runs = [{"hallucination_detected": True}] * 5
        assert compute_hallucination_rate(runs) == 1.0

    def test_hallucination_rate_none(self):
        runs = [{"hallucination_detected": False}] * 4
        assert compute_hallucination_rate(runs) == 0.0

    def test_hallucination_rate_empty(self):
        assert compute_hallucination_rate([]) == 0.0

    def test_aggregate_eval_scores(self):
        runs = [
            {"groundedness": 4.0, "relevance": 4.5, "coherence": 3.5,
             "overall": 0.8, "hallucination_detected": False},
            {"groundedness": 3.0, "relevance": 3.5, "coherence": 4.0,
             "overall": 0.7, "hallucination_detected": True},
        ]
        agg = aggregate_eval_scores(runs)
        assert "groundedness_mean" in agg
        assert agg["groundedness_mean"] == pytest.approx(3.5)
        assert agg["hallucination_rate"] == 0.5
        assert agg["total_runs"] == 2

    def test_token_overlap_complete(self):
        score = compute_token_overlap("the cat sat on the mat", "cat sat mat")
        assert score == 1.0

    def test_token_overlap_zero(self):
        score = compute_token_overlap("hello world", "completely different")
        assert score == 0.0