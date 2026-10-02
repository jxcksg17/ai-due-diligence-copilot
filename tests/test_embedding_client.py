"""Unit tests for the OpenAI embedding client without real API calls."""

import pytest

from app.config import Settings
from app.embeddings.client import (
    BGE_DIMENSIONS,
    BGE_QUERY_INSTRUCTION,
    BGEEmbeddingClient,
    OpenAIEmbeddingClient,
    get_embedding_client,
)


class _FakeEmbeddingsEndpoint:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return type(
            "Response",
            (),
            {"data": [type("Embedding", (), {"embedding": self.vector})()]},
        )()


def test_openai_client_uses_configured_model_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _FakeEmbeddingsEndpoint([0.5] * 1536)

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.embeddings = endpoint

    monkeypatch.setattr("app.embeddings.client.OpenAI", FakeOpenAI)
    client = OpenAIEmbeddingClient(
        api_key="test-key", model="text-embedding-3-small", dimensions=1536
    )

    assert client.embed("source text") == [0.5] * 1536
    assert endpoint.calls == [
        {
            "input": "source text",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        }
    ]


def test_openai_client_rejects_an_unexpected_vector_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _FakeEmbeddingsEndpoint([0.5])

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            self.embeddings = endpoint

    monkeypatch.setattr("app.embeddings.client.OpenAI", FakeOpenAI)
    client = OpenAIEmbeddingClient(
        api_key="test-key", model="text-embedding-3-small", dimensions=1536
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        client.embed("source text")


class _FakeBGEEncoder:
    def __init__(self, dimensions: int = BGE_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.inputs: list[str] = []

    def encode(self, text: str, *, normalize_embeddings: bool) -> list[float]:
        assert normalize_embeddings is True
        self.inputs.append(text)
        return [0.25] * self.dimensions


def test_bge_uses_distinct_document_and_query_encodings() -> None:
    encoder = _FakeBGEEncoder()
    client = BGEEmbeddingClient(
        model_name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        device="cpu",
        encoder=encoder,
    )

    assert client.embed("filing text") == [0.25] * 1024
    assert client.embed_query("What are the risks?") == [0.25] * 1024
    assert encoder.inputs == [
        "filing text",
        f"{BGE_QUERY_INSTRUCTION}What are the risks?",
    ]


def test_bge_rejects_a_dimension_other_than_its_native_output() -> None:
    with pytest.raises(ValueError, match="1024-dimension"):
        BGEEmbeddingClient(
            model_name="BAAI/bge-large-en-v1.5",
            dimensions=1536,
            encoder=_FakeBGEEncoder(),
        )


def test_factory_selects_bge_without_an_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakeBGEClient:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr("app.embeddings.client.BGEEmbeddingClient", FakeBGEClient)
    settings = Settings(
        database_url="sqlite://",
        embedding_provider="bge",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_dimensions=1024,
    )

    get_embedding_client(settings)

    assert created == {
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
    }
