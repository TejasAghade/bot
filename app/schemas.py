from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class ChatResponse(BaseModel):
    answer: str
    used_context: bool


class IngestRequest(BaseModel):
    append: bool = False


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    total_chunks_in_store: int
