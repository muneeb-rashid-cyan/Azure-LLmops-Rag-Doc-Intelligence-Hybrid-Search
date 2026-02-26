"""
chunker.py
----------
Advanced chunking strategies:

1. Semantic chunking   — splits on meaning drop (cosine similarity)
2. Table-aware         — keeps table rows together, never splits mid-table
3. Overlap chunks      — sliding window with configurable overlap
4. Metadata injection  — every chunk carries doc_name, page_num, chunk_type
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import tiktoken
from loguru import logger
from openai import AzureOpenAI

from config import get_settings
from ingestion.document_loader import ExtractedDocument, ExtractedPage


class ChunkType(str, Enum):
    TEXT  = "text"
    TABLE = "table"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_index: int
    chunk_type: ChunkType
    content: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter that handles common edge cases."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _embed_sentences_local(sentences: list[str]) -> np.ndarray:
    """
    Embed sentences for semantic similarity using OpenAI embeddings.
    Used only during chunking to detect topic boundaries.
    """
    settings = get_settings()
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    # Batch embed (max 16 at a time to avoid rate limits)
    embeddings = []
    batch_size = 16
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=settings.azure_openai_emb_deployment,
        )
        embeddings.extend([e.embedding for e in response.data])

    return np.array(embeddings)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def semantic_chunk_page(
    page: ExtractedPage,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """
    Semantic chunking on a single page:
    1. Split page text into sentences
    2. Embed each sentence
    3. Find topic boundaries (where cosine similarity drops)
    4. Group sentences into chunks at boundaries
    5. Apply overlap between adjacent chunks
    """
    chunks: list[Chunk] = []
    text = page.text.strip()

    if not text:
        return chunks

    sentences = _split_into_sentences(text)
    if len(sentences) < 2:
        # Too short to semantically chunk — return as single chunk
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{page.page_number}_c0",
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=page.page_number,
            chunk_index=0,
            chunk_type=ChunkType.TEXT,
            content=text,
            token_count=_count_tokens(text),
            metadata={"sentence_count": len(sentences)},
        ))
        return chunks

    # Get embeddings for boundary detection
    try:
        embeddings = _embed_sentences_local(sentences)
    except Exception as e:
        logger.warning(f"Embedding failed for semantic chunking, falling back to fixed: {e}")
        return fixed_size_chunk_page(page, doc_id, doc_name, chunk_size, chunk_overlap)

    # Find boundaries where similarity drops below threshold
    boundaries = [0]
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            boundaries.append(i)
    boundaries.append(len(sentences))

    # Build chunks from boundary segments
    chunk_index = 0
    for b in range(len(boundaries) - 1):
        start = boundaries[b]
        end   = boundaries[b + 1]

        segment_sentences = sentences[start:end]
        segment_text = " ".join(segment_sentences)

        # If segment exceeds chunk_size, further split by token count
        if _count_tokens(segment_text) > chunk_size:
            sub_chunks = _split_by_tokens(
                segment_sentences, doc_id, doc_name,
                page.page_number, chunk_index, chunk_size, chunk_overlap
            )
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)
        else:
            # Add overlap from previous chunk
            overlap_text = ""
            if chunks and chunk_overlap > 0:
                prev_words = chunks[-1].content.split()[-chunk_overlap:]
                overlap_text = " ".join(prev_words) + " "

            final_text = (overlap_text + segment_text).strip()
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page.page_number}_c{chunk_index}",
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                chunk_type=ChunkType.TEXT,
                content=final_text,
                token_count=_count_tokens(final_text),
                metadata={"sentence_count": len(segment_sentences), "has_overlap": bool(overlap_text)},
            ))
            chunk_index += 1

    return chunks


def _split_by_tokens(
    sentences: list[str],
    doc_id: str,
    doc_name: str,
    page_number: int,
    start_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Fixed-size token splitting for oversized semantic segments."""
    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = start_index

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_type=ChunkType.TEXT,
                content=text,
                token_count=_count_tokens(text),
                metadata={"split_method": "token_fallback"},
            ))
            chunk_index += 1
            # Keep last N sentences as overlap
            overlap_count = max(1, chunk_overlap // 20)
            current_sentences = current_sentences[-overlap_count:]
            current_tokens = sum(_count_tokens(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        text = " ".join(current_sentences)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_type=ChunkType.TEXT,
            content=text,
            token_count=_count_tokens(text),
            metadata={"split_method": "token_fallback"},
        ))

    return chunks


def fixed_size_chunk_page(
    page: ExtractedPage,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Simple fallback token-based chunker."""
    return _split_by_tokens(
        _split_into_sentences(page.text),
        doc_id, doc_name, page.page_number, 0, chunk_size, chunk_overlap
    )


def chunk_tables(doc: ExtractedDocument) -> list[Chunk]:
    """
    Table-aware chunking — each table becomes its own chunk.
    Tables are NEVER split mid-row.
    """
    chunks = []
    for i, table in enumerate(doc.tables):
        content = (
            f"[TABLE from {doc.file_name}, Page {table.page_number}]\n"
            f"{table.markdown}"
        )
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_table_{i}",
            doc_id=doc.doc_id,
            doc_name=doc.file_name,
            page_number=table.page_number,
            chunk_index=i,
            chunk_type=ChunkType.TABLE,
            content=content,
            token_count=_count_tokens(content),
            metadata={
                "row_count": table.row_count,
                "column_count": table.column_count,
                "is_table": True,
            },
        ))
    return chunks


def chunk_document(
    doc: ExtractedDocument,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    use_semantic: bool = True,
) -> list[Chunk]:
    """
    Full document chunking pipeline:
    1. Semantic chunk each page (or fixed if semantic fails)
    2. Table-aware chunk all tables separately
    3. Return combined list sorted by page + chunk_index
    """
    all_chunks: list[Chunk] = []

    logger.info(f"Chunking {doc.file_name} — {doc.total_pages} pages, semantic={use_semantic}")

    for page in doc.pages:
        if use_semantic:
            page_chunks = semantic_chunk_page(
                page, doc.doc_id, doc.file_name, chunk_size, chunk_overlap
            )
        else:
            page_chunks = fixed_size_chunk_page(
                page, doc.doc_id, doc.file_name, chunk_size, chunk_overlap
            )
        all_chunks.extend(page_chunks)

    table_chunks = chunk_tables(doc)
    all_chunks.extend(table_chunks)

    logger.info(
        f"Created {len(all_chunks)} chunks "
        f"({len(all_chunks) - len(table_chunks)} text, {len(table_chunks)} table)"
    )
    return all_chunks