from __future__ import annotations

import hashlib
import logging

from app.config import Settings, get_settings
from app.document_loader import IncrementalSink, load_documents, split_documents
from app.vectorstore import clear_vectorstore, get_vectorstore

logger = logging.getLogger(__name__)


def run_ingestion(append: bool = False, settings: Settings | None = None) -> dict[str, int]:
    """Ingest documents into the vector store, one file at a time.

    Each loaded document is chunked and written immediately, so the run is
    interruptible: if it is stopped part-way, every file processed so far is already
    persisted. Re-running with ``append=True`` resumes -- files already in the store
    are skipped (SharePoint files before their download), so no work is repeated.

    ``append=False`` clears the store first (a clean full rebuild); ``append=True``
    keeps existing content and only adds what is missing.
    """
    cfg = settings or get_settings()
    vectorstore = get_vectorstore(cfg)

    if not append:
        clear_vectorstore(vectorstore)

    # Resume support: when appending, skip files already present in the store.
    ingested_sources = _existing_sources(vectorstore) if append else set()

    stats = {"documents_loaded": 0, "chunks_indexed": 0}

    def persist(doc) -> None:
        chunks = split_documents([doc], cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            return
        ids = [_doc_id(chunk) for chunk in chunks]
        # Write all chunks of a single file in one call so a file is either fully
        # persisted or not at all -- an interruption never leaves a half-indexed file.
        vectorstore.add_documents(chunks, ids=ids)
        stats["documents_loaded"] += 1
        stats["chunks_indexed"] += len(chunks)

    sink = IncrementalSink(persist, ingested_sources)

    load_documents(
        cfg.data_dir,
        cfg.urls_file,
        azure_devops_pat=cfg.azure_devops_pat,
        azure_devops_org=cfg.azure_devops_org,
        azure_devops_project=cfg.azure_devops_project,
        azure_devops_projects=cfg.azure_devops_projects_list,
        azure_devops_wiki=cfg.azure_devops_wiki,
        azure_devops_wiki_path=cfg.azure_devops_wiki_path,
        azure_devops_api_version=cfg.azure_devops_api_version,
        sharepoint_tenant_id=cfg.sharepoint_tenant_id,
        sharepoint_client_id=cfg.sharepoint_client_id,
        sharepoint_client_secret=cfg.sharepoint_client_secret,
        sharepoint_site=cfg.sharepoint_site,
        sharepoint_folders=cfg.sharepoint_folders_list,
        sharepoint_graph_base_url=cfg.sharepoint_graph_base_url,
        sharepoint_login_base_url=cfg.sharepoint_login_base_url,
        sink=sink,
    )

    return {
        "documents_loaded": stats["documents_loaded"],
        "chunks_indexed": stats["chunks_indexed"],
        "total_chunks_in_store": vectorstore._collection.count(),
    }


def _existing_sources(vectorstore) -> set[str]:
    """Return the set of ``source`` values already indexed, for resume skipping."""
    try:
        existing = vectorstore.get(include=["metadatas"])
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not read existing sources for resume: %s", exc)
        return set()
    sources: set[str] = set()
    for meta in existing.get("metadatas", []) or []:
        if isinstance(meta, dict):
            source = str(meta.get("source") or "")
            if source:
                sources.add(source)
    return sources


def _doc_id(doc) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", ""))
    chunk_id = str(doc.metadata.get("chunk_id", ""))
    payload = f"{source}|{page}|{chunk_id}|{doc.page_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
