"""
document_loader.py
------------------
Uses Azure Document Intelligence (layout model) to extract:
  - Text with reading order preserved
  - Tables as structured JSON
  - Page numbers + bounding boxes
  - Handwritten/scanned content via OCR

Supports: PDF, DOCX, PNG, JPG, TIFF, BMP
"""

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from loguru import logger

from config import get_settings


@dataclass
class ExtractedTable:
    page_number: int
    row_count: int
    column_count: int
    cells: list[dict]
    markdown: str          # table rendered as markdown string


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    tables: list[ExtractedTable] = field(default_factory=list)
    word_count: int = 0


@dataclass
class ExtractedDocument:
    doc_id: str
    file_name: str
    file_type: str
    total_pages: int
    pages: list[ExtractedPage]
    raw_text: str           # all pages concatenated
    tables: list[ExtractedTable]
    metadata: dict = field(default_factory=dict)


def _table_to_markdown(table) -> str:
    """Convert Document Intelligence table object to markdown."""
    if not table.cells:
        return ""

    grid: dict[tuple, str] = {}
    for cell in table.cells:
        grid[(cell.row_index, cell.column_index)] = cell.content.strip()

    rows = table.row_count
    cols = table.column_count
    lines = []

    for r in range(rows):
        row_cells = [grid.get((r, c), "") for c in range(cols)]
        lines.append("| " + " | ".join(row_cells) + " |")
        if r == 0:
            lines.append("|" + "|".join(["---"] * cols) + "|")

    return "\n".join(lines)


def load_document(file_path: str | Path) -> ExtractedDocument:
    """
    Extract full content from a document using Azure Document Intelligence.

    Args:
        file_path: Path to PDF, DOCX, or image file

    Returns:
        ExtractedDocument with text, tables, page numbers
    """
    settings = get_settings()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    client = DocumentAnalysisClient(
        endpoint=settings.document_intelligence_endpoint,
        credential=AzureKeyCredential(settings.document_intelligence_key),
    )

    logger.info(f"Extracting document: {file_path.name}")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Use layout model — best for mixed text + tables + figures
    poller = client.begin_analyze_document("prebuilt-layout", file_bytes)
    result = poller.result()

    # ── Build per-page structure ──────────────────────────────
    pages: list[ExtractedPage] = []
    all_tables: list[ExtractedTable] = []

    # Map tables to page numbers
    table_page_map: dict[int, list[ExtractedTable]] = {}
    for table in result.tables:
        page_num = table.bounding_regions[0].page_number if table.bounding_regions else 1
        ext_table = ExtractedTable(
            page_number=page_num,
            row_count=table.row_count,
            column_count=table.column_count,
            cells=[
                {
                    "row": cell.row_index,
                    "col": cell.column_index,
                    "content": cell.content,
                    "is_header": cell.kind == "columnHeader",
                }
                for cell in table.cells
            ],
            markdown=_table_to_markdown(table),
        )
        all_tables.append(ext_table)
        table_page_map.setdefault(page_num, []).append(ext_table)

    # Extract text per page
    for page in result.pages:
        page_num = page.page_number
        page_text = ""

        if page.lines:
            page_text = "\n".join(line.content for line in page.lines)

        # Append table markdown to page text so it gets chunked with context
        page_tables = table_page_map.get(page_num, [])
        for tbl in page_tables:
            page_text += f"\n\n[TABLE PAGE {page_num}]\n{tbl.markdown}\n"

        pages.append(ExtractedPage(
            page_number=page_num,
            text=page_text,
            tables=page_tables,
            word_count=len(page_text.split()),
        ))

    raw_text = "\n\n".join(p.text for p in pages)

    doc = ExtractedDocument(
        doc_id=file_path.stem,
        file_name=file_path.name,
        file_type=file_path.suffix.lower(),
        total_pages=len(pages),
        pages=pages,
        raw_text=raw_text,
        tables=all_tables,
        metadata={
            "file_size_bytes": file_path.stat().st_size,
            "total_words": len(raw_text.split()),
            "total_tables": len(all_tables),
        },
    )

    logger.info(
        f"Extracted {doc.total_pages} pages, "
        f"{len(all_tables)} tables, "
        f"{doc.metadata['total_words']} words from {file_path.name}"
    )
    return doc