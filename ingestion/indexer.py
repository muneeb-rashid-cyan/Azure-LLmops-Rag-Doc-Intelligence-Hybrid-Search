"""
indexer.py
----------
Creates the Azure AI Search index (if not exists) and uploads chunks.

Index schema supports:
  - Vector field (1536-dim for ada-002)
  - BM25 keyword field (same content)
  - Metadata fields: doc_id, page_number, chunk_type, doc_name
  → Enables hybrid search (vector + keyword) in one query
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from loguru import logger

from config import get_settings
from ingestion.chunker import Chunk

VECTOR_DIM = 1536   # text-embedding-ada-002 output dimension


def get_index_client() -> SearchIndexClient:
    s = get_settings()
    return SearchIndexClient(
        endpoint=s.azure_search_endpoint,
        credential=AzureKeyCredential(s.azure_search_key),
    )


def get_search_client() -> SearchClient:
    s = get_settings()
    return SearchClient(
        endpoint=s.azure_search_endpoint,
        index_name=s.azure_search_index,
        credential=AzureKeyCredential(s.azure_search_key),
    )


def create_index_if_not_exists() -> None:
    """
    Create the search index with hybrid search support.
    Safe to call multiple times — skips if index already exists.
    """
    settings = get_settings()
    client   = get_index_client()

    existing = [idx.name for idx in client.list_indexes()]
    if settings.azure_search_index in existing:
        logger.info(f"Index '{settings.azure_search_index}' already exists — skipping creation")
        return

    logger.info(f"Creating index '{settings.azure_search_index}'...")

    index = SearchIndex(
        name=settings.azure_search_index,
        fields=[
            SimpleField(name="id",           type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc_id",        type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="doc_name",      type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_number",   type=SearchFieldDataType.Int32,  filterable=True, sortable=True),
            SimpleField(name="chunk_index",   type=SearchFieldDataType.Int32,  filterable=True),
            SimpleField(name="chunk_type",    type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content",   type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=VECTOR_DIM,
                vector_search_profile_name="rag-vector-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="rag-hnsw")],
            profiles=[VectorSearchProfile(name="rag-vector-profile", algorithm_configuration_name="rag-hnsw")],
        ),
    )

    client.create_index(index)
    logger.info(f"Index '{settings.azure_search_index}' created successfully")


def upload_chunks(chunks: list[Chunk], batch_size: int = 100) -> int:
    """
    Upload embedded chunks to Azure AI Search.
    Each chunk must have metadata['embedding'] set by embedder.py.

    Returns number of successfully uploaded chunks.
    """
    create_index_if_not_exists()
    search_client = get_search_client()

    documents = []
    for chunk in chunks:
        if "embedding" not in chunk.metadata:
            logger.warning(f"Chunk {chunk.chunk_id} missing embedding — skipping")
            continue

        documents.append({
            "id":             chunk.chunk_id,
            "doc_id":         chunk.doc_id,
            "doc_name":       chunk.doc_name,
            "page_number":    chunk.page_number,
            "chunk_index":    chunk.chunk_index,
            "chunk_type":     chunk.chunk_type.value,
            "content":        chunk.content,
            "content_vector": chunk.metadata["embedding"],
        })

    if not documents:
        logger.warning("No documents to upload")
        return 0

    uploaded = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = search_client.upload_documents(batch)
        succeeded = sum(1 for r in result if r.succeeded)
        uploaded += succeeded
        logger.info(f"Uploaded batch {i//batch_size + 1}: {succeeded}/{len(batch)} succeeded")

    logger.info(f"Total uploaded: {uploaded}/{len(documents)} chunks")
    return uploaded


def delete_doc_chunks(doc_id: str) -> int:
    """Delete all chunks belonging to a specific document."""
    search_client = get_search_client()
    results = list(search_client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["id"],
    ))
    if not results:
        return 0
    docs_to_delete = [{"id": r["id"]} for r in results]
    search_client.delete_documents(docs_to_delete)
    logger.info(f"Deleted {len(docs_to_delete)} chunks for doc_id={doc_id}")
    return len(docs_to_delete)