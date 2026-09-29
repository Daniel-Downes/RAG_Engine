"""Embed document chunks and upsert them into Qdrant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from backend.ingestion import load_pdfs, split_pages
from backend.ingestion.pdf_loader import DEFAULT_INPUT_DIR

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_BATCH_SIZE = 64


def index_chunks(
    chunks: list[dict[str, Any]],
    client: QdrantClient,
    model: SentenceTransformer,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed chunks and upsert them as Qdrant points with text and metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not chunks:
        return 0

    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        raise ValueError("The embedding model did not report its vector dimension")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        embeddings = model.encode(
            [chunk["text"] for chunk in batch],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if len(embeddings) != len(batch):
            raise ValueError("The embedding model returned an unexpected vector count")

        points = []
        for chunk, embedding in zip(batch, embeddings, strict=True):
            metadata = chunk["metadata"]
            point_key = ":".join(
                str(metadata[field])
                for field in ("source", "page_number", "chunk_index")
            )
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, point_key)),
                    vector=[float(value) for value in embedding],
                    payload={"text": chunk["text"], "metadata": metadata},
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)

    stored_count = client.count(collection_name=collection_name, exact=True).count
    if stored_count != len(chunks):
        raise RuntimeError(
            f"Qdrant point count mismatch for collection '{collection_name}': "
            f"generated {len(chunks)} chunks, but Qdrant contains {stored_count} "
            "points. Check for stale points or failed upserts."
        )

    return len(chunks)


def index_raw_documents(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    collection_name: str | None = None,
    model_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Load, chunk, embed, and index all PDFs under ``input_dir``."""
    chunks = split_pages(load_pdfs(input_dir))
    if not chunks:
        return 0

    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = os.getenv("QDRANT_URL", f"http://{host}:{port}")
    target_collection = collection_name or os.getenv(
        "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME
    )
    target_model = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)

    client = QdrantClient(url=qdrant_url)
    try:
        model = SentenceTransformer(target_model)
        return index_chunks(
            chunks,
            client,
            model,
            collection_name=target_collection,
            batch_size=batch_size,
        )
    finally:
        client.close()