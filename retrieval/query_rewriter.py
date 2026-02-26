"""
query_rewriter.py
-----------------
Rewrites the user query into N alternative phrasings.
This dramatically improves recall — different phrasings
match different chunk styles in the index.

Example:
  Input:  "what were the revenue numbers?"
  Output: [
    "total revenue figures reported in the document",
    "financial performance revenue metrics and figures",
    "annual revenue quarterly earnings breakdown"
  ]
"""

import json
from loguru import logger
from openai import AzureOpenAI

from config import get_settings


REWRITE_PROMPT = """You are an expert at query reformulation for document retrieval systems.

Given a user question, generate {n} alternative phrasings that:
1. Preserve the original intent exactly
2. Use different vocabulary and sentence structure
3. Cover different ways the answer might be phrased in a document
4. Include both specific and general phrasings

Return ONLY a JSON array of strings. No explanation, no markdown.

User question: {query}

JSON array of {n} alternative phrasings:"""


def rewrite_query(query: str, n: int | None = None) -> list[str]:
    """
    Rewrite query into N alternative phrasings.

    Returns list of rewritten queries (original NOT included).
    The caller should combine original + rewrites for retrieval.
    """
    settings = get_settings()
    n = n or settings.query_rewrite_count

    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )

    try:
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[{
                "role": "user",
                "content": REWRITE_PROMPT.format(query=query, n=n),
            }],
            temperature=0.7,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        rewrites = json.loads(raw)

        if not isinstance(rewrites, list):
            raise ValueError("Expected JSON array")

        rewrites = [r.strip() for r in rewrites if r.strip()][:n]
        logger.info(f"Query rewritten into {len(rewrites)} alternatives")
        return rewrites

    except Exception as e:
        logger.warning(f"Query rewriting failed: {e} — using original only")
        return []