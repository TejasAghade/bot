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
    top_k: int = 4
    max_context_docs: int = 3
    max_context_chars: int = 3500
    min_relevance: float = 0.55
    fast_path_min_relevance: float = 0.72
    max_answer_sentences: int = 5
    answer_cache_size: int = 128

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_num_predict: int = 320
    ollama_keep_alive: str = "30m"
    embedding_model: str = "nomic-embed-text"
    embedding_keep_alive: int = 1800
    azure_devops_pat: str | None = None
    azure_devops_org: str | None = None
    azure_devops_project: str | None = None
    azure_devops_wiki: str | None = None
    azure_devops_wiki_path: str = "/"
    azure_devops_api_version: str = "7.1"
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

    @field_validator("fast_path_min_relevance")
    @classmethod
    def validate_fast_path_relevance(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("FAST_PATH_MIN_RELEVANCE must be between 0 and 1.")
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
