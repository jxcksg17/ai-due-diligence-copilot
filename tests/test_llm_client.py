"""Unit tests for Ollama configuration without a running local model."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generation.llm_client import OllamaLLMClient, get_llm_client
from app.generation.service import GroundedAnswer


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"answer":"Supported [1]","citation_ids":[1],'
                '"insufficient_evidence":false}'
            )
        )


def test_ollama_client_uses_pinned_safe_generation_settings() -> None:
    transport = FakeOllamaClient()
    client = OllamaLLMClient(
        model="qwen3:8b-q4_K_M",
        host="http://localhost:11434",
        num_ctx=8192,
        temperature=0.0,
        max_output_tokens=512,
        client=transport,
    )

    response = client.generate(
        system_prompt="system",
        user_prompt="user",
        response_format=GroundedAnswer,
    )

    assert '"citation_ids":[1]' in response
    assert transport.calls == [
        {
            "model": "qwen3:8b-q4_K_M",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "format": GroundedAnswer.model_json_schema(),
            "options": {
                "num_ctx": 8192,
                "temperature": 0.0,
                "num_predict": 512,
                "seed": 42,
            },
            "think": False,
            "stream": False,
        }
    ]


def test_factory_selects_ollama_from_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr("app.generation.llm_client.OllamaLLMClient", FakeClient)
    settings = Settings(database_url="sqlite://")

    get_llm_client(settings)

    assert created == {
        "model": "qwen3:8b-q4_K_M",
        "host": "http://localhost:11434",
        "num_ctx": 8192,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "timeout_seconds": 180.0,
    }


def test_factory_rejects_unknown_provider() -> None:
    settings = Settings(database_url="sqlite://", llm_provider="unknown")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_client(settings)
