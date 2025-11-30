"""Pydantic schemas for vector database."""

from typing import Any

from pydantic import BaseModel, Field


class TextChunkMetadata(BaseModel):
    """Metadata for text chunks."""

    domain: int | None = Field(None, description="Knowledge base domain (1-11)")
    source: str = Field(..., description="Source file or publication ID")
    chunk_index: int = Field(..., description="Chunk index within source")
    publication_id: str | None = Field(None, description="Publication identifier")
    section: str | None = Field(None, description="Section within document")


class CodeChunkMetadata(BaseModel):
    """Metadata for code chunks."""

    file_path: str = Field(..., description="Path to source file")
    function_name: str | None = Field(None, description="Function or method name")
    class_name: str | None = Field(None, description="Class name if applicable")
    engine: str | None = Field(None, description="Engine name (cortex, signalcore, etc)")
    line_start: int | None = Field(None, description="Starting line number")
    line_end: int | None = Field(None, description="Ending line number")


class RetrievalResult(BaseModel):
    """Result from vector database query."""

    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., description="Query text", min_length=1)
    query_type: str | None = Field(None, description="Query type: text, code, or hybrid")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    min_score: float | None = Field(None, description="Minimum similarity score", ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """RAG query response."""

    query: str = Field(..., description="Original query")
    query_type: str = Field(..., description="Detected or specified query type")
    results: list[RetrievalResult] = Field(default_factory=list, description="Retrieved results")
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    total_candidates: int = Field(..., description="Total candidates before reranking")
