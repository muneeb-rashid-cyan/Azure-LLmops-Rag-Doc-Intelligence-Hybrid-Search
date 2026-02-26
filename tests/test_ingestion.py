"""
test_ingestion.py
-----------------
Tests for document loading, chunking, embedding, and indexing.
Run: pytest tests/test_ingestion.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ingestion.chunker import (
    chunk_document,
    fixed_size_chunk_page,
    chunk_tables,
    ChunkType,
    Chunk,
)
from ingestion.document_loader import ExtractedDocument, ExtractedPage, ExtractedTable
from evaluation.metrics import compute_answer_f1, compute_exact_match


# ── FIXTURES ──────────────────────────────────────────────────
def make_fake_page(text: str, page_num: int = 1) -> ExtractedPage:
    return ExtractedPage(
        page_number=page_num,
        text=text,
        tables=[],
        word_count=len(text.split()),
    )


def make_fake_doc(pages: list[ExtractedPage]) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="test_doc",
        file_name="test_doc.pdf",
        file_type=".pdf",
        total_pages=len(pages),
        pages=pages,
        raw_text="\n\n".join(p.text for p in pages),
        tables=[],
        metadata={},
    )


# ── CHUNKER TESTS ─────────────────────────────────────────────
class TestChunker:

    def test_fixed_size_chunk_basic(self):
        page = make_fake_page(
            "The quick brown fox jumps over the lazy dog. "
            "This is a second sentence about something else. "
            "And a third sentence to make it longer."
        )
        chunks = fixed_size_chunk_page(page, "doc1", "test.pdf", chunk_size=20)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunk_type == ChunkType.TEXT for c in chunks)

    def test_chunk_ids_are_unique(self):
        page = make_fake_page(
            "Sentence one is here. Sentence two follows. "
            "Sentence three comes after. Sentence four is last. "
            "And a fifth one to fill the page."
        )
        chunks = fixed_size_chunk_page(page, "doc1", "test.pdf", chunk_size=15)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_metadata_populated(self):
        page = make_fake_page("Hello world this is a test sentence.", page_num=3)
        chunks = fixed_size_chunk_page(page, "mydoc", "myfile.pdf")
        for chunk in chunks:
            assert chunk.doc_id == "mydoc"
            assert chunk.doc_name == "myfile.pdf"
            assert chunk.page_number == 3
            assert chunk.token_count > 0

    def test_table_chunking(self):
        table = ExtractedTable(
            page_number=2,
            row_count=3,
            column_count=2,
            cells=[
                {"row": 0, "col": 0, "content": "Name",    "is_header": True},
                {"row": 0, "col": 1, "content": "Value",   "is_header": True},
                {"row": 1, "col": 0, "content": "Revenue", "is_header": False},
                {"row": 1, "col": 1, "content": "$4.2B",   "is_header": False},
            ],
            markdown="| Name | Value |\n|---|---|\n| Revenue | $4.2B |",
        )
        doc = make_fake_doc([make_fake_page("Some text")])
        doc.tables = [table]

        table_chunks = chunk_tables(doc)
        assert len(table_chunks) == 1
        assert table_chunks[0].chunk_type == ChunkType.TABLE
        assert "$4.2B" in table_chunks[0].content
        assert "Page 2" in table_chunks[0].content

    def test_empty_page_returns_no_chunks(self):
        page = make_fake_page("")
        chunks = fixed_size_chunk_page(page, "doc1", "test.pdf")
        assert len(chunks) == 0

    def test_chunk_content_not_empty(self):
        page = make_fake_page(
            "This document contains important financial information. "
            "The quarterly results show significant growth. "
            "Revenue increased by 15 percent year over year."
        )
        chunks = fixed_size_chunk_page(page, "doc1", "test.pdf")
        for chunk in chunks:
            assert chunk.content.strip() != ""


# ── METRICS TESTS ─────────────────────────────────────────────
class TestMetrics:

    def test_f1_perfect_match(self):
        score = compute_answer_f1("the revenue was four billion", "the revenue was four billion")
        assert score == 1.0

    def test_f1_no_overlap(self):
        score = compute_answer_f1("completely different text here", "nothing in common at all")
        assert score == 0.0

    def test_f1_partial_overlap(self):
        score = compute_answer_f1("revenue was four billion dollars", "revenue four billion")
        assert 0 < score < 1.0

    def test_exact_match_true(self):
        assert compute_exact_match("the total revenue was $4.2 billion", "$4.2 billion") is True

    def test_exact_match_false(self):
        assert compute_exact_match("revenue was strong", "$4.2 billion") is False

    def test_exact_match_case_insensitive(self):
        assert compute_exact_match("The Revenue Was HIGH", "the revenue was high") is True

    def test_f1_empty_strings(self):
        assert compute_answer_f1("", "some text") == 0.0
        assert compute_answer_f1("some text", "") == 0.0