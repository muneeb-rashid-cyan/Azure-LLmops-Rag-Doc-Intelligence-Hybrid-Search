"""
pipeline.py (ingestion)
-----------------------
Orchestrates the full ingestion flow:
  document_loader → chunker → embedder → indexer → cosmos
"""

from pathlib import Path
from loguru import logger

from config import get_settings
from ingestion.document_loader import load_document
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks


def run_ingestion_pipeline(file_path: str | Path) -> dict:
    """
    Full ingestion pipeline for a single document.

    Returns:
        dict with doc_id, chunk_count, page_count, table_count
    """
    settings  = get_settings()
    file_path = Path(file_path)

    logger.info(f"Starting ingestion pipeline for: {file_path.name}")

    # Step 1: Extract with Document Intelligence
    doc = load_document(file_path)

    # Step 2: Chunk (semantic + table-aware)
    chunks = chunk_document(
        doc,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        use_semantic=True,
    )

    # Step 3: Embed
    chunks = embed_chunks(chunks)

    # Step 4: Index into Azure AI Search
    from ingestion.indexer import upload_chunks
    uploaded = upload_chunks(chunks)

    # Step 5: Store document record in Cosmos DB
    try:
        from dashboard.cosmos_client import get_cosmos_client
        cosmos = get_cosmos_client()
        cosmos.upsert_document_record({
            "id":          doc.doc_id,
            "file_name":   doc.file_name,
            "total_pages": doc.total_pages,
            "total_chunks": uploaded,
            "total_tables": len(doc.tables),
            "metadata":    doc.metadata,
        })
    except Exception as e:
        logger.warning(f"Cosmos record save failed (non-critical): {e}")

    result = {
        "doc_id":      doc.doc_id,
        "file_name":   doc.file_name,
        "page_count":  doc.total_pages,
        "chunk_count": uploaded,
        "table_count": len(doc.tables),
    }
    logger.info(f"Ingestion complete: {result}")
    return result