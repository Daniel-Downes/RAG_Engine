"""Token-aware text splitting for extracted PDF pages."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100


def split_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split extracted page records into token-sized chunks, retaining metadata.

    Chunk sizes and overlap are measured with the ``cl100k_base`` tokenizer.
    Fenced code blocks and paragraph boundaries are preferred split points.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n```", "\n\n", "\n", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_index, text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        **page["metadata"],
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks