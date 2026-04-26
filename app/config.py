"""
APEX Configuration — Centralized settings via Pydantic.
All environment variables are loaded here and validated at startup.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Literal
from functools import lru_cache


class TigerGraphSettings(BaseSettings):
    """TigerGraph database connection settings."""
    host: str = Field(default="http://localhost:14240", alias="TIGERGRAPH_HOST")
    graph_name: str = Field(default="APEX_KG", alias="TIGERGRAPH_GRAPH_NAME")
    username: str = Field(default="tigergraph", alias="TIGERGRAPH_USERNAME")
    password: str = Field(default="tigergraph", alias="TIGERGRAPH_PASSWORD")
    api_token: str = Field(default="", alias="TIGERGRAPH_API_TOKEN")

    model_config = {"env_file": ".env", "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    provider: Literal["openai", "google", "groq", "ollama"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )
    model: str = Field(default="llama3.2:latest", alias="LLM_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    temperature: float = 0.1
    max_tokens: int = 2048

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def api_key(self) -> str:
        """Return the active API key based on provider."""
        key_map = {
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "ollama": "",
        }
        return key_map.get(self.provider, "")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model_name: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    model_config = {"env_file": ".env", "extra": "ignore"}


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    cache_ttl: int = Field(default=3600, alias="REDIS_CACHE_TTL")

    model_config = {"env_file": ".env", "extra": "ignore"}


class EvaluationSettings(BaseSettings):
    """Evaluation and benchmark configuration."""
    batch_size: int = Field(default=10, alias="BENCHMARK_BATCH_SIZE")
    crag_grade_threshold: float = Field(default=0.75, alias="CRAG_GRADE_THRESHOLD")
    cache_confidence_threshold: float = Field(
        default=0.75, alias="CACHE_CONFIDENCE_THRESHOLD"
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


class AppSettings(BaseSettings):
    """Main application settings."""
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_name: str = "APEX — Self-Improving GraphRAG"
    version: str = "1.0.0"

    # Sub-settings
    tigergraph: TigerGraphSettings = TigerGraphSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    redis: RedisSettings = RedisSettings()
    evaluation: EvaluationSettings = EvaluationSettings()

    model_config = {"env_file": ".env", "extra": "ignore"}

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """Handle common non-boolean DEBUG values from shell environments."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
                return False
        return value


@lru_cache
def get_settings() -> AppSettings:
    """Cached singleton for app settings."""
    return AppSettings()
