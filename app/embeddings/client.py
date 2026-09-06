"""OpenAI embedding client behind a small provider abstraction."""

from typing import Any, Protocol

from openai import OpenAI

from app.config import Settings


class EmbeddingClient(Protocol):
    """Generate document and query vectors for the configured provider."""

    def embed(self, text: str) -> list[float]:
        """Return a document embedding vector for *text*."""

    def embed_query(self, text: str) -> list[float]:
        """Return a query embedding vector for *text*."""


class OpenAIEmbeddingClient:
    """EmbeddingClient implementation for the approved OpenAI provider."""

    def __init__(self, *, api_key: str, model: str, dimensions: int) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        response = self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimensions,
        )
        embedding = list(response.data[0].embedding)

        if len(embedding) != self._dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self._dimensions}, received {len(embedding)}"
            )
        return embedding


BGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_DIMENSIONS = 1024
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _select_local_device() -> str:
    """Prefer Apple Metal when PyTorch exposes an available MPS device."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbeddingClient:
    """Local BGE provider with BGE's distinct query instruction handling."""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        if dimensions != BGE_DIMENSIONS:
            raise ValueError(
                f"{BGE_MODEL} produces {BGE_DIMENSIONS}-dimension embeddings"
            )

        self._dimensions = dimensions
        self.device = device or _select_local_device()
        if encoder is not None:
            self._encoder = encoder
            return

        from sentence_transformers import SentenceTransformer

        try:
            self._encoder = SentenceTransformer(model_name, device=self.device)
        except RuntimeError:
            if self.device != "mps":
                raise
            # Some PyTorch operations may not be implemented by MPS. Loading
            # on CPU preserves a functional, entirely local fallback.
            self.device = "cpu"
            self._encoder = SentenceTransformer(model_name, device=self.device)

    def embed(self, text: str) -> list[float]:
        """Embed a document chunk without a query instruction."""
        return self._encode(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query using BGE's recommended retrieval instruction."""
        return self._encode(f"{BGE_QUERY_INSTRUCTION}{text}")

    def _encode(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        encoded = self._encoder.encode(text, normalize_embeddings=True)
        embedding = list(encoded)
        if len(embedding) != self._dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self._dimensions}, received {len(embedding)}"
            )
        return embedding


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    """Construct the configured embedding provider.

    Keeping construction in one factory means callers depend only on the small
    EmbeddingClient interface.
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    if settings.embedding_provider == "bge":
        if settings.embedding_model != BGE_MODEL:
            raise ValueError(
                f"Unsupported BGE model {settings.embedding_model!r}; expected {BGE_MODEL!r}"
            )
        return BGEEmbeddingClient(
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER "
        f"{settings.embedding_provider!r}; expected 'bge' or 'openai'"
    )
