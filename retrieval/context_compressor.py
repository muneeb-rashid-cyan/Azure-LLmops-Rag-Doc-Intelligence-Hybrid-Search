"""
context_compressor.py
---------------------
Compresses retrieved chunks by extracting ONLY the sentences
that are relevant to the query.

Why?
  A retrieved chunk might be 400 tokens but only 2 sentences
  actually answer the question. Sending 400 tokens wastes
  context window and adds noise. Compression extracts the 2
  relevant sentences → cleaner, more focused answer.
"""

from loguru import logger
from openai import AzureOpenAI

from config import get_settings
from retrieval.reranker import RankedResult


COMPRESS_PROMPT = """You are a precise text extractor.

Given a QUESTION and a TEXT PASSAGE, extract ONLY the sentences from the passage
that are directly relevant to answering the question.

Rules:
- Copy sentences VERBATIM from the passage (word for word)
- Only include sentences that contain information needed to answer the question
- If no sentences are relevant, return exactly: [NOT RELEVANT]
- Do NOT paraphrase, summarize, or add any text of your own
- Do NOT include your own commentary

QUESTION: {query}

TEXT PASSAGE:
{passage}

Relevant sentences (verbatim from passage only):"""


def compress_chunk(query: str, chunk: RankedResult) -> str:
    """
    Extract relevant sentences from a single chunk.
    Returns compressed text or original if compression fails.
    """
    settings = get_settings()
    client   = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    # Tables should not be compressed — structure matters
    if chunk.chunk_type == "table":
        return chunk.content

    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[{
                "role": "user",
                "content": COMPRESS_PROMPT.format(
                    query=query,
                    passage=chunk.content,
                ),
            }],
            temperature=0.0,
            max_tokens=300,
        )

        compressed = response.choices[0].message.content.strip()

        if compressed == "[NOT RELEVANT]" or not compressed:
            return ""

        return compressed

    except Exception as e:
        logger.warning(f"Compression failed for chunk {chunk.chunk_id}: {e}")
        return chunk.content


def compress_context(
    query: str,
    chunks: list[RankedResult],
    max_tokens: int = 2000,
) -> list[dict]:
    """
    Compress all retrieved chunks and return context list.

    Returns list of dicts with compressed content + citation metadata.
    Stops when total tokens would exceed max_tokens.
    """
    compressed_context = []
    total_tokens = 0

    for chunk in chunks:
        compressed = compress_chunk(query, chunk)

        if not compressed:
            continue

        # Rough token estimate (4 chars ≈ 1 token)
        estimated_tokens = len(compressed) // 4

        if total_tokens + estimated_tokens > max_tokens:
            logger.debug(f"Context limit reached at {total_tokens} tokens")
            break

        compressed_context.append({
            "chunk_id":    chunk.chunk_id,
            "doc_name":    chunk.doc_name,
            "page_number": chunk.page_number,
            "chunk_type":  chunk.chunk_type,
            "content":     compressed,
            "rerank_score": chunk.rerank_score,
        })
        total_tokens += estimated_tokens

    logger.info(
        f"Compressed {len(chunks)} chunks → {len(compressed_context)} relevant "
        f"(~{total_tokens} tokens)"
    )
    return compressed_context