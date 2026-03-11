from __future__ import annotations

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.config import Settings


def get_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vectorstore(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=settings.vectorstore_dir,
    )


def clear_vectorstore(vectorstore: Chroma) -> int:
    existing = vectorstore.get(include=[])
    ids = existing.get("ids", []) or []
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)
