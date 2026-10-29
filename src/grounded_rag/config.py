from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every value is overridable by environment variable (prefix GR_)."""

    model_config = SettingsConfigDict(env_prefix="GR_", env_file=".env", extra="ignore")

    cohere_api_key: str | None = Field(default=None, validation_alias=AliasChoices("COHERE_API_KEY", "GR_COHERE_API_KEY"))
    offline: bool = False  # force the deterministic backend even if a key is present
    chat_model: str = "command-a-03-2025"
    embed_model: str = "embed-v4.0"
    rerank_model: str = "rerank-v3.5"
    embed_dim: int = 1024
    db_path: str = "grounded_rag.sqlite"
    chunk_chars: int = 900
    chunk_overlap: int = 150
    dense_k: int = 20
    lexical_k: int = 20
    rerank_top_n: int = 5
    min_relevance: float = 0.0  # drop reranked hits with score <= this (offline overlap score is 0 when no term matches)
    agent_max_steps: int = 6
    agent_max_tool_calls: int = 8
    log_level: str = "INFO"
    rate_limit_per_min: int = 0  # client-side throttle; trial keys allow 10 calls/min

    @property
    def use_live(self) -> bool:
        return bool(self.cohere_api_key) and not self.offline


def get_settings() -> Settings:
    return Settings()
