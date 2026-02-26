"""
citation_formatter.py
---------------------
Formats citations from generated answers into clean,
display-ready structures for the dashboard.

Input:  raw answer text with inline [doc, Page X] markers
Output: structured citation objects + highlighted answer HTML
"""

import re
from dataclasses import dataclass


@dataclass
class Citation:
    index: int          # citation number [1], [2], etc.
    doc_name: str
    page_number: int
    chunk_id: str | None = None
    snippet: str | None = None   # the text this citation supports


def extract_citations_from_answer(
    answer: str,
    context_chunks: list[dict],
) -> tuple[str, list[Citation]]:
    """
    Extract citations from answer text and return:
    1. Clean answer with numbered citation markers [1], [2]
    2. List of Citation objects

    Example:
      Input:  "Revenue was $4.2B [report.pdf, Page 3]"
      Output: "Revenue was $4.2B [1]", [Citation(1, "report.pdf", 3)]
    """
    # Pattern: [filename, Page N] or [filename, page N]
    pattern = r'\[([^\]]+),\s*[Pp]age\s*(\d+)\]'
    matches = list(re.finditer(pattern, answer))

    citations: list[Citation] = []
    seen: dict[str, int] = {}   # "doc_page" → citation index
    citation_map: dict[str, str] = {}   # original marker → [N]

    for match in matches:
        doc_name   = match.group(1).strip()
        page_num   = int(match.group(2))
        key        = f"{doc_name}_{page_num}"
        original   = match.group(0)

        if key not in seen:
            idx = len(citations) + 1
            seen[key] = idx

            # Find matching chunk for snippet
            snippet = None
            chunk_id = None
            for chunk in context_chunks:
                if (chunk.get("doc_name") == doc_name and
                        chunk.get("page_number") == page_num):
                    snippet  = chunk.get("content", "")[:150] + "..."
                    chunk_id = chunk.get("chunk_id")
                    break

            citations.append(Citation(
                index=idx,
                doc_name=doc_name,
                page_number=page_num,
                chunk_id=chunk_id,
                snippet=snippet,
            ))

        citation_map[original] = f"[{seen[key]}]"

    # Replace inline markers with numbered references
    clean_answer = answer
    for original, numbered in citation_map.items():
        clean_answer = clean_answer.replace(original, numbered)

    return clean_answer, citations


def format_citations_markdown(citations: list[Citation]) -> str:
    """Format citations as a markdown reference list."""
    if not citations:
        return ""

    lines = ["\n**Sources:**"]
    for c in citations:
        lines.append(f"[{c.index}] {c.doc_name}, Page {c.page_number}")
        if c.snippet:
            lines.append(f"   > {c.snippet}")

    return "\n".join(lines)


def format_citations_html(citations: list[Citation]) -> str:
    """Format citations as HTML for dashboard display."""
    if not citations:
        return ""

    items = []
    for c in citations:
        snippet_html = (
            f'<div class="citation-snippet">{c.snippet}</div>'
            if c.snippet else ""
        )
        items.append(f"""
        <div class="citation-item">
          <span class="citation-index">[{c.index}]</span>
          <span class="citation-source">{c.doc_name}</span>
          <span class="citation-page">Page {c.page_number}</span>
          {snippet_html}
        </div>""")

    return '<div class="citations">' + "".join(items) + "</div>"


def format_answer_with_citations(
    answer: str,
    context_chunks: list[dict],
    output_format: str = "dict",
) -> dict:
    """
    Full citation formatting pipeline.

    Args:
        answer:          raw answer from GPT with [doc, Page X] markers
        context_chunks:  context used for retrieval
        output_format:   "dict" | "markdown" | "html"

    Returns:
        dict with clean_answer, citations, formatted output
    """
    clean_answer, citations = extract_citations_from_answer(answer, context_chunks)

    result = {
        "clean_answer": clean_answer,
        "citations": [
            {
                "index":       c.index,
                "doc_name":    c.doc_name,
                "page_number": c.page_number,
                "chunk_id":    c.chunk_id,
                "snippet":     c.snippet,
            }
            for c in citations
        ],
        "citation_count": len(citations),
    }

    if output_format == "markdown":
        result["formatted"] = clean_answer + format_citations_markdown(citations)
    elif output_format == "html":
        result["formatted"] = clean_answer + format_citations_html(citations)

    return result