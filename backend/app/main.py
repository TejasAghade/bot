from __future__ import annotations

import hashlib
import logging
import threading
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.document_loader import DocumentLoadError, list_user_accessible_projects
from app.ingestion import run_ingestion
from app.rag import RAGService
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    ProjectsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# How long to trust the (PAT -> accessible projects) lookup before re-querying AzDO.
PAT_PROJECTS_TTL_SECONDS = 300

settings = get_settings()
app = FastAPI(title="Document-Only Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag_service: RAGService | None = None
_pat_projects_cache: dict[str, tuple[float, list[str]]] = {}
_pat_projects_lock = threading.Lock()


def rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
    return _rag_service


def _pat_fingerprint(pat: str) -> str:
    return hashlib.sha256(pat.encode("utf-8")).hexdigest()


def _resolve_accessible_projects(pat: str) -> list[str]:
    """Return projects the PAT can read in Azure DevOps, using a short-lived cache.

    Access is always sourced from AzDO via the caller's PAT. The server's
    AZURE_DEVOPS_PROJECTS env var only controls what gets ingested; it does
    not gate what a user is allowed to ask about. If a project the user can
    access has not been ingested, retrieval simply returns no results for it
    (there's no data in the vectorstore to leak).
    """
    if not settings.azure_devops_org:
        raise HTTPException(
            status_code=500,
            detail="AZURE_DEVOPS_ORG is not configured on the server.",
        )

    fingerprint = _pat_fingerprint(pat)
    now = time.monotonic()
    with _pat_projects_lock:
        cached = _pat_projects_cache.get(fingerprint)
        if cached and now - cached[0] < PAT_PROJECTS_TTL_SECONDS:
            return list(cached[1])

    try:
        user_projects = list_user_accessible_projects(
            organization=settings.azure_devops_org,
            azure_devops_pat=pat,
            api_version=settings.azure_devops_api_version,
        )
    except DocumentLoadError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    with _pat_projects_lock:
        _pat_projects_cache[fingerprint] = (now, list(user_projects))
    return user_projects


def _require_pat(header_value: str | None) -> str:
    pat = (header_value or "").strip()
    if not pat:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Azure-Devops-Pat header.",
        )
    return pat


@app.get("/health")
def health() -> dict[str, object]:
    service = rag_service()
    return {
        "status": "ok",
        "indexed_chunks": service.indexed_document_count(),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        result = run_ingestion(append=payload.append, settings=settings)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rag_service().reload_vectorstore()
    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    x_azure_devops_pat: str | None = Header(default=None, alias="X-Azure-Devops-Pat"),
) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Azure DevOps access is OPTIONAL. Without a (valid) PAT the bot still answers
    # from non-AzDO sources (SharePoint, uploaded files, URLs). A valid PAT
    # additionally unlocks Azure DevOps wiki content for the caller's projects.
    pat = (x_azure_devops_pat or "").strip()
    accessible_projects: list[str] = []
    if pat:
        try:
            accessible_projects = _resolve_accessible_projects(pat)
        except HTTPException as exc:
            # A bad PAT must not block non-AzDO answers: log and continue without AzDO.
            logger.warning(
                "Azure DevOps access unavailable (%s); answering from non-AzDO sources only.",
                exc.detail,
            )

    requested_project = (payload.project or "").strip() or None
    if requested_project:
        accessible_lower = {name.lower(): name for name in accessible_projects}
        if requested_project.lower() not in accessible_lower:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Unauthorized: you do not have access to project "
                    f"'{requested_project}'. Supply a valid X-Azure-Devops-Pat with "
                    "access to this project."
                ),
            )
        requested_project = accessible_lower[requested_project.lower()]

    service = rag_service()
    if service.indexed_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No indexed data found. Run /ingest first.",
        )

    result = service.answer(
        question,
        project=requested_project,
        allowed_projects=None if requested_project else accessible_projects,
    )
    return ChatResponse(
        answer=result.answer,
        used_context=result.used_context,
        project=requested_project,
        sources=result.sources,
    )


@app.get("/projects", response_model=ProjectsResponse)
def projects(
    x_azure_devops_pat: str | None = Header(default=None, alias="X-Azure-Devops-Pat"),
) -> ProjectsResponse:
    pat = _require_pat(x_azure_devops_pat)
    accessible = _resolve_accessible_projects(pat)
    return ProjectsResponse(projects=accessible)
