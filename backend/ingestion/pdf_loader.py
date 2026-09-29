"""Extract page text and metadata from PDFs in the raw document directory."""

from __future__ import annotations

import re
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_docs"
HEADER_FOOTER_MARGIN = 0.12
PAGE_NUMBER_PATTERN = re.compile(
    r"(?:page\s*)?[-\u2013\u2014|]?\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*[-\u2013\u2014|]?",
    re.IGNORECASE,
)


def _page_text_lines(page: pymupdf.Page) -> list[tuple[str, float, float]]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((text, line["bbox"][1], line["bbox"][3]))
    return lines


def _normalized_line(text: str) -> str:
    return " ".join(text.casefold().split())


def _extract_document_pages(
    pdf: pymupdf.Document,
    document_metadata: dict[str, Any],
    source: str,
    file_name: str,
) -> list[dict[str, Any]]:
    page_lines = [_page_text_lines(page) for page in pdf]
    edge_occurrences: Counter[str] = Counter()

    for page, lines in zip(pdf, page_lines, strict=True):
        page_height = page.rect.height
        edge_occurrences.update(
            {
                _normalized_line(text)
                for text, y0, y1 in lines
                if y1 <= page_height * HEADER_FOOTER_MARGIN
                or y0 >= page_height * (1 - HEADER_FOOTER_MARGIN)
            }
        )

    repeat_threshold = max(2, ceil(pdf.page_count * 0.05))
    repeated_edge_lines = {
        text
        for text, occurrences in edge_occurrences.items()
        if occurrences >= repeat_threshold
    }

    pages = []
    for page_index, (page, lines) in enumerate(zip(pdf, page_lines, strict=True)):
        page_height = page.rect.height
        retained_lines = []
        for text, y0, y1 in lines:
            is_top_or_bottom = (
                y1 <= page_height * HEADER_FOOTER_MARGIN
                or y0 >= page_height * (1 - HEADER_FOOTER_MARGIN)
            )
            if is_top_or_bottom and _normalized_line(text) in repeated_edge_lines:
                continue
            if (
                y0 >= page_height * (1 - HEADER_FOOTER_MARGIN)
                and PAGE_NUMBER_PATTERN.fullmatch(text)
            ):
                continue
            retained_lines.append(text)

        pages.append(
            {
                "text": "\n".join(retained_lines).strip(),
                "metadata": {
                    **document_metadata,
                    "source": source,
                    "file_name": file_name,
                    "page_number": page_index + 1,
                    "total_pages": pdf.page_count,
                },
            }
        )

    return pages


def load_pdfs(input_dir: str | Path = DEFAULT_INPUT_DIR) -> list[dict[str, Any]]:
    """Load every PDF below ``input_dir`` as one text-and-metadata record per page.

    Each result has ``text`` and ``metadata`` keys. Page numbers are 1-based,
    and the source path in metadata is relative to ``input_dir``.
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"PDF input directory does not exist: {input_path}")

    pages: list[dict[str, Any]] = []
    for pdf_path in sorted(input_path.rglob("*.pdf")):
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"PDF is password-protected: {pdf_path}")

            document_metadata = {
                key: value for key, value in (pdf.metadata or {}).items() if value
            }
            source = pdf_path.relative_to(input_path).as_posix()

            pages.extend(
                _extract_document_pages(
                    pdf,
                    document_metadata,
                    source,
                    pdf_path.name,
                )
            )

    return pages