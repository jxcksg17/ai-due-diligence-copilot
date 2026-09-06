"""
Centralized, environment-based configuration.

Every other module that needs a config value (database URL, app name,
future API keys, etc.) imports `get_settings()` from here instead of
calling `os.getenv()` directly. This keeps configuration:

  - typed and validated (pydantic will error loudly at startup if a
    required value is missing or malformed, instead of failing later
    with a confusing runtime error deep in the code),
  - centralized (one file to look at to see every configurable value
    the app depends on),
  - environment-agnostic (the same code runs in dev/test/prod; only
    the underlying environment variables change).

`lru_cache` makes `get_settings()` a cheap singleton: the .env file
and environment are only read once per process, and every caller gets
the same Settings instance.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # General application metadata.
    app_name: str = "AI Due Diligence Copilot"
    app_env: str = "development"  # development | test | production
    debug: bool = True

    # Database connection string, e.g.:
    #   postgresql+psycopg://user:password@localhost:5432/dd_copilot
    # Required: there is no sensible default for a real database, so if
    # this is missing, startup should fail clearly rather than silently
    # falling back to something unexpected.
    database_url: str

    # The local BGE model is the default M3 embedding provider. OpenAI remains
    # available when explicitly selected through EMBEDDING_PROVIDER.
    embedding_provider: str = "bge"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimensions: int = 1024
    openai_api_key: str | None = None

    # M3 grounded generation. Ollama keeps generation local by default while
    # the provider setting preserves a clean boundary for future clients.
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b-q4_K_M"
    ollama_host: str = "http://localhost:11434"
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 512

    # M6 hybrid retrieval. The small cross-encoder keeps reranking practical
    # alongside BGE-large and Ollama on a 16 GB Apple Silicon machine.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_candidate_k: int = 20
    reranker_batch_size: int = 8
    hybrid_rrf_k: int = 60

    # M7 citation entailment verification. This is a distinct NLI model, not
    # the relevance reranker or the answer-generating LLM.
    citation_verifier_model: str = "cross-encoder/nli-deberta-v3-small"
    citation_verifier_batch_size: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using a function (rather than a module-level `settings = Settings()`)
    means FastAPI's dependency-injection system can override this in
    tests (see tests/conftest.py) without monkeypatching module state.
    """
    return Settings()
