"""Extract page text and metadata from PDFs in the raw document directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_docs"


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

            for page_index, page in enumerate(pdf):
                pages.append(
                    {
                        "text": page.get_text("text").strip(),
                        "metadata": {
                            **document_metadata,
                            "source": source,
                            "file_name": pdf_path.name,
                            "page_number": page_index + 1,
                            "total_pages": pdf.page_count,
                        },
                    }
                )

    return pages