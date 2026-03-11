from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: str = "data"
    urls_file: str = "data/urls.txt"
    vectorstore_dir: str = "vectorstore"
    collection_name: str = "docs"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 8
    min_relevance: float = 0.55

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("min_relevance")
    @classmethod
    def validate_relevance(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("MIN_RELEVANCE must be between 0 and 1.")
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

