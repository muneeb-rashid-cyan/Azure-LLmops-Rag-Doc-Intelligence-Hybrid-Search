"""
answer_generator.py
-------------------
GPT-4.1 answer generation grounded in retrieved context.
Returns structured response with answer + citations + confidence.
"""

from dataclasses import dataclass, field
from loguru import logger
from openai import AzureOpenAI

from config import get_settings
from generation.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[dict]
    query: str
    context_chunks_used: int
    rewritten_queries: list[str] = field(default_factory=list)
    retrieval_stats: dict = field(default_factory=dict)


def _format_context(context_chunks: list[dict]) -> str:
    """Format context chunks into a numbered list for the prompt."""
    lines = []
    for i, chunk in enumerate(context_chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk['doc_name']}, Page {chunk['page_number']} "
            f"(type: {chunk['chunk_type']})\n{chunk['content']}"
        )
    return "\n\n".join(lines)


def generate_answer(
    query: str,
    context_chunks: list[dict],
    rewritten_queries: list[str] | None = None,
    retrieval_stats: dict | None = None,
) -> GeneratedAnswer:
    """
    Generate a grounded answer using GPT-4.1.

    Args:
        query:             original user question
        context_chunks:    compressed context from retrieval pipeline
        rewritten_queries: alternative phrasings used for retrieval
        retrieval_stats:   metadata about retrieval process

    Returns:
        GeneratedAnswer with answer text and extracted citations
    """
    settings = get_settings()
    client   = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    if not context_chunks:
        return GeneratedAnswer(
            answer="I could not find relevant information in the provided documents.",
            citations=[],
            query=query,
            context_chunks_used=0,
        )

    context_text = _format_context(context_chunks)

    response = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": RAG_USER_PROMPT.format(
                context=context_text,
                query=query,
            )},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer_text = response.choices[0].message.content.strip()

    # Extract citations from the answer
    citations = _extract_citations(answer_text, context_chunks)

    logger.info(
        f"Generated answer: {len(answer_text)} chars, "
        f"{len(citations)} citations, "
        f"{len(context_chunks)} context chunks used"
    )

    return GeneratedAnswer(
        answer=answer_text,
        citations=citations,
        query=query,
        context_chunks_used=len(context_chunks),
        rewritten_queries=rewritten_queries or [],
        retrieval_stats=retrieval_stats or {},
    )


def _extract_citations(answer: str, context_chunks: list[dict]) -> list[dict]:
    """Extract citation references from the generated answer."""
    import re
    citations = []
    seen = set()

    # Match patterns like [filename, Page 3] or [doc.pdf, Page 12]
    pattern = r'\[([^\]]+),\s*Page\s*(\d+)\]'
    matches = re.findall(pattern, answer)

    for doc_name, page_num in matches:
        key = f"{doc_name.strip()}_{page_num}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "doc_name":    doc_name.strip(),
                "page_number": int(page_num),
            })

    # Also include all source chunks as citation candidates
    for chunk in context_chunks:
        key = f"{chunk['doc_name']}_{chunk['page_number']}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "doc_name":    chunk["doc_name"],
                "page_number": chunk["page_number"],
            })

    return citations