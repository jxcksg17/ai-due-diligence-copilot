"""Provider boundary for structured LLM generation."""

from typing import Any, Protocol

from pydantic import BaseModel

from app.config import Settings


class LLMClient(Protocol):
    """Generate a structured response from system and user prompts."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
    ) -> str:
        """Return the provider's response body as a JSON string."""


class OllamaLLMClient:
    """Local Ollama implementation of the LLM provider boundary."""

    def __init__(
        self,
        *,
        model: str,
        host: str,
        num_ctx: int,
        temperature: float,
        max_output_tokens: int,
        client: Any | None = None,
    ) -> None:
        if num_ctx < 1:
            raise ValueError("num_ctx must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        if client is None:
            from ollama import Client

            client = Client(host=host)

        self._client = client
        self._model = model
        self._options = {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": max_output_tokens,
            "seed": 42,
        }

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
    ) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=response_format.model_json_schema(),
            options=self._options,
            think=False,
            stream=False,
        )

        if isinstance(response, dict):
            return str(response["message"]["content"])
        return str(response.message.content)


def get_llm_client(settings: Settings) -> LLMClient:
    """Construct the explicitly configured generation provider."""
    if settings.llm_provider == "ollama":
        return OllamaLLMClient(
            model=settings.llm_model,
            host=settings.ollama_host,
            num_ctx=settings.llm_num_ctx,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER {settings.llm_provider!r}; expected 'ollama'"
    )
