from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.ingestion import run_ingestion
from app.rag import RAGService
from app.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse

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


def rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
    return _rag_service


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
    result = run_ingestion(append=payload.append, settings=settings)
    rag_service().reload_vectorstore()
    return IngestResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    service = rag_service()
    if service.indexed_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No indexed data found. Run /ingest first.",
        )

    result = service.answer(question)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        used_context=result.used_context,
    )

