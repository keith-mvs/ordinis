"""
Pydantic models for RAG API requests and responses.
"""

from typing import Any

from pydantic import BaseModel, Field

from rag.retrieval.query_classifier import QueryType


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str = Field(..., description="Query text")
    query_type: QueryType | None = Field(None, description="Query type (auto-detected if None)")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")


class QueryResult(BaseModel):
    """Individual query result."""

    content: str = Field(..., description="Result content")
    score: float = Field(..., description="Similarity score")
    metadata: dict[str, Any] = Field(..., description="Result metadata")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    query: str
    query_type: QueryType
    results: list[QueryResult]
    total_candidates: int = Field(..., description="Total candidates before filtering")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    text_embedder_available: bool
    code_embedder_available: bool
    chroma_persist_directory: str


class StatsResponse(BaseModel):
    """Collection statistics response."""

    text_chunks_count: int
    code_chunks_count: int
    total_chunks: int
    text_embedder_model: str
    code_embedder_model: str


class ConfigResponse(BaseModel):
    """RAG configuration response."""

    text_embedding_model: str
    code_embedding_model: str
    rerank_model: str
    use_local_embeddings: bool
    top_k_retrieval: int
    similarity_threshold: float
