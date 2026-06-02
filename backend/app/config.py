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
    enable_fast_path: bool = False
    fast_path_min_relevance: float = 0.6
    fast_path_min_overlap: float = 0.2
    fast_path_max_docs: int = 2
    max_answer_sentences: int = 6
    answer_cache_size: int = 128

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_num_predict: int = 192
    ollama_keep_alive: str = "30m"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_threads: int | None = None
    embedding_cache_dir: str | None = None
    azure_devops_pat: str | None = None
    azure_devops_org: str | None = None
    azure_devops_project: str | None = None
    azure_devops_projects: str | None = None
    azure_devops_wiki: str | None = None
    azure_devops_wiki_path: str = "/"
    azure_devops_api_version: str = "7.1"

    # SharePoint (Microsoft Graph, app-only / client-credentials flow).
    sharepoint_tenant_id: str | None = None
    sharepoint_client_id: str | None = None
    sharepoint_client_secret: str | None = None
    # Optional: limit ingestion to a single site. Accepts a Graph site id or an
    # addressable path like "contoso.sharepoint.com:/sites/Marketing". When blank,
    # every site the app can read is enumerated.
    sharepoint_site: str | None = None
    # Optional comma-separated folder scope. When blank, all folders are ingested
    # (current behaviour). When set, only files under matching folders are ingested.
    # This is also the seam for later per-folder answer scoping.
    sharepoint_folders: str | None = None
    sharepoint_graph_base_url: str = "https://graph.microsoft.com/v1.0"
    sharepoint_login_base_url: str = "https://login.microsoftonline.com"

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

    @field_validator("fast_path_min_overlap")
    @classmethod
    def validate_fast_path_overlap(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("FAST_PATH_MIN_OVERLAP must be between 0 and 1.")
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def azure_devops_projects_list(self) -> list[str]:
        projects: list[str] = []
        seen: set[str] = set()
        for raw in (self.azure_devops_projects or "").split(","):
            name = raw.strip()
            if name and name not in seen:
                seen.add(name)
                projects.append(name)
        if not projects and self.azure_devops_project:
            single = self.azure_devops_project.strip()
            if single:
                projects.append(single)
        return projects

    @property
    def sharepoint_enabled(self) -> bool:
        return bool(
            self.sharepoint_tenant_id
            and self.sharepoint_client_id
            and self.sharepoint_client_secret
        )

    @property
    def sharepoint_folders_list(self) -> list[str]:
        folders: list[str] = []
        seen: set[str] = set()
        for raw in (self.sharepoint_folders or "").split(","):
            name = raw.strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                folders.append(name)
        return folders


@lru_cache
def get_settings() -> Settings:
    return Settings()
