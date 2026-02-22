from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment and defaults."""

    app_name: str = Field(default="job-rag")
    log_level: str = Field(default="INFO")

    data_path: str = Field(default="./data/lf_jobs.csv")
    vector_dir: str = Field(default="./storage")
    embedding_model: str = Field(default="intfloat/e5-large-v2")
    embedding_batch_size: int = Field(default=16)

    top_k: int = Field(default=5)
    use_hybrid: bool = Field(default=False)
    hybrid_alpha: float = Field(default=0.35)
    rerank_model: str | None = Field(default=None)

    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=500)

    pinecone_api_key: str | None = Field(default=None)
    pinecone_index: str = Field(default="job-rag")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")
    pinecone_metric: str = Field(default="cosine")

    redis_url: str | None = Field(default=None)
    cache_ttl_seconds: int = Field(default=300, ge=0)

    model_config = SettingsConfigDict(env_file=(".env", ".env.example"), case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Returns:
        The application Settings instance.
    """
    return Settings()
