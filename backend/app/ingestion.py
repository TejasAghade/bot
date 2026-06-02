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
