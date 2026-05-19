from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.config import Settings


def get_embeddings(settings: Settings) -> FastEmbedEmbeddings:
    kwargs: dict[str, object] = {"model_name": settings.embedding_model}
    if settings.embedding_threads:
        kwargs["threads"] = settings.embedding_threads
    if settings.embedding_cache_dir:
        kwargs["cache_dir"] = settings.embedding_cache_dir
    return FastEmbedEmbeddings(**kwargs)


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
