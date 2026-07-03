from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    project: str | None = Field(
        default=None,
        description="Optional Azure DevOps project name. When set, the bot only "
        "answers from chunks ingested for this project.",
    )


class SourceInfo(BaseModel):
    origin: str = Field(
        description="Where the context came from: 'sharepoint', 'wiki', 'web', or 'file'."
    )
    title: str = Field(description="File name or page title of the source.")
    url: str | None = Field(default=None, description="Link to the source, if available.")
    site: str | None = None
    folder: str | None = None
    project: str | None = None


class ChatResponse(BaseModel):
    answer: str
    used_context: bool
    project: str | None = None
    sources: list[SourceInfo] = Field(default_factory=list)


class IngestRequest(BaseModel):
    append: bool = False


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    total_chunks_in_store: int


class ProjectsResponse(BaseModel):
    projects: list[str]
