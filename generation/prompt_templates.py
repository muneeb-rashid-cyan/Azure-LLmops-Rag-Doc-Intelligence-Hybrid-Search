"""
prompt_templates.py
-------------------
All GPT-4.1 prompt templates in one place.
"""

RAG_SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer questions ONLY using the provided context chunks.
Every factual claim MUST include a citation in format [doc_name, Page X].

Rules:
- If the answer is not in the context, say "I could not find this information in the provided documents."
- Never use prior knowledge — only use the provided context
- Always cite your source for every claim
- For tables, reference the table directly
- Be concise and accurate"""


RAG_USER_PROMPT = """Context chunks retrieved from documents:

{context}

---
Question: {query}

Provide a precise answer with citations for every claim.
Format citations as [filename, Page X] immediately after each claim."""


CITATION_EXTRACTION_PROMPT = """Extract all citations from this answer text.
Return a JSON array of objects with: doc_name, page_number, quote_snippet.

Answer text:
{answer}

JSON array only, no other text:"""