from __future__ import annotations

import hashlib

from app.config import Settings, get_settings
from app.document_loader import load_documents, split_documents
from app.vectorstore import clear_vectorstore, get_vectorstore


def run_ingestion(append: bool = False, settings: Settings | None = None) -> dict[str, int]:
    cfg = settings or get_settings()
    vectorstore = get_vectorstore(cfg)

    if not append:
        clear_vectorstore(vectorstore)

    documents = load_documents(
        cfg.data_dir,
        cfg.urls_file,
        azure_devops_pat=cfg.azure_devops_pat,
    )
    if not documents:
        return {
            "documents_loaded": 0,
            "chunks_indexed": 0,
            "total_chunks_in_store": vectorstore._collection.count(),
        }

    chunks = split_documents(documents, cfg.chunk_size, cfg.chunk_overlap)
    ids = [_doc_id(doc) for doc in chunks]
    vectorstore.add_documents(chunks, ids=ids)

    total = vectorstore._collection.count()
    return {
        "documents_loaded": len(documents),
        "chunks_indexed": len(chunks),
        "total_chunks_in_store": total,
    }


def _doc_id(doc) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", ""))
    chunk_id = str(doc.metadata.get("chunk_id", ""))
    payload = f"{source}|{page}|{chunk_id}|{doc.page_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
