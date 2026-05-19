from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    project: str | None = Field(
        default=None,
        description="Optional Azure DevOps project name. When set, the bot only "
        "answers from chunks ingested for this project.",
    )


class ChatResponse(BaseModel):
    answer: str
    used_context: bool
    project: str | None = None


class IngestRequest(BaseModel):
    append: bool = False


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    total_chunks_in_store: int


class ProjectsResponse(BaseModel):
    projects: list[str]
