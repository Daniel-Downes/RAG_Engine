from types import SimpleNamespace

import pytest

from backend.vectorstore.indexer import index_chunks


class FakeModel:
    def __init__(self):
        self.encoded_batches = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.encoded_batches.append((texts, kwargs))
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeQdrantClient:
    def __init__(self, collection_exists=False, count_override=None):
        self.exists = collection_exists
        self.created = []
        self.upserts = []
        self.points = {}
        self.count_override = count_override

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        for point in kwargs["points"]:
            self.points[point.id] = point

    def count(self, collection_name, exact):
        assert exact is True
        count = self.count_override
        if count is None:
            count = len(self.points)
        return SimpleNamespace(count=count)


def test_index_chunks_creates_collection_and_upserts_text_metadata_in_batches():
    chunks = [
        {
            "text": f"Chunk {index}",
            "metadata": {
                "source": "manual.pdf",
                "page_number": 1,
                "chunk_index": index,
            },
        }
        for index in range(3)
    ]
    client = FakeQdrantClient()
    model = FakeModel()

    indexed_count = index_chunks(
        chunks,
        client,
        model,
        collection_name="test_documents",
        batch_size=2,
    )

    assert indexed_count == 3
    assert len(client.created) == 1
    vector_config = client.created[0]["vectors_config"]
    assert vector_config.size == 3
    assert vector_config.distance.value == "Cosine"
    assert len(client.upserts) == 2
    assert [len(upsert["points"]) for upsert in client.upserts] == [2, 1]
    assert len(model.encoded_batches) == 2
    assert model.encoded_batches[0][1]["normalize_embeddings"] is True

    point = client.upserts[0]["points"][0]
    assert point.vector == [1.0, 0.0, 0.0]
    assert point.payload == {"text": "Chunk 0", "metadata": chunks[0]["metadata"]}
    assert client.upserts[0]["collection_name"] == "test_documents"


def test_index_chunks_does_not_recreate_existing_collection():
    client = FakeQdrantClient(collection_exists=True)
    model = FakeModel()
    chunks = [
        {
            "text": "A chunk",
            "metadata": {"source": "manual.pdf", "page_number": 1, "chunk_index": 0},
        }
    ]

    assert index_chunks(chunks, client, model) == 1
    assert client.created == []


def test_index_chunks_returns_zero_without_creating_collection():
    client = FakeQdrantClient()
    model = FakeModel()

    assert index_chunks([], client, model) == 0
    assert client.created == []
    assert client.upserts == []


def test_index_chunks_raises_when_qdrant_point_count_does_not_match():
    client = FakeQdrantClient(count_override=2)
    model = FakeModel()
    chunks = [
        {
            "text": "A chunk",
            "metadata": {"source": "manual.pdf", "page_number": 1, "chunk_index": 0},
        }
    ]

    with pytest.raises(RuntimeError, match="generated 1 chunks, but Qdrant contains 2"):
        index_chunks(chunks, client, model, collection_name="test_documents")