"""
embedder.py
-----------
Embeds chunks using Azure OpenAI text-embedding-ada-002.
Handles batching, retries, and rate limit backoff.
"""

import time
from loguru import logger
from openai import AzureOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from ingestion.chunker import Chunk


def get_client() -> AzureOpenAI:
    s = get_settings()
    return AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_key,
        api_version=s.azure_openai_api_version,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _embed_batch(texts: list[str], client: AzureOpenAI, model: str) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [e.embedding for e in response.data]


def embed_chunks(chunks: list[Chunk], batch_size: int = 16) -> list[Chunk]:
    """
    Embed all chunks in batches.
    Adds embedding vector directly to each chunk's metadata.
    Returns chunks with embeddings attached.
    """
    settings = get_settings()
    client   = get_client()
    model    = settings.azure_openai_emb_deployment

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")

    embedded: list[Chunk] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.content for c in batch]

        try:
            vectors = _embed_batch(texts, client, model)
            for chunk, vector in zip(batch, vectors):
                chunk.metadata["embedding"] = vector
                embedded.append(chunk)
        except Exception as e:
            logger.error(f"Embedding batch {i//batch_size} failed: {e}")
            raise

        # Gentle rate limit
        if i + batch_size < len(chunks):
            time.sleep(0.2)

    logger.info(f"Embedded {len(embedded)} chunks successfully")
    return embedded


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    settings = get_settings()
    client   = get_client()
    response = client.embeddings.create(
        input=[query],
        model=settings.azure_openai_emb_deployment,
    )
    return response.data[0].embedding