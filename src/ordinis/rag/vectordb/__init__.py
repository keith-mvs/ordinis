"""Vector database clients."""

from typing import TYPE_CHECKING

from ordinis.rag.vectordb.schema import (
    CodeChunkMetadata,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    TextChunkMetadata,
)

if TYPE_CHECKING:
    from ordinis.rag.vectordb.chroma_client import ChromaClient

__all__ = [
    "ChromaClient",
    "CodeChunkMetadata",
    "QueryRequest",
    "QueryResponse",
    "RetrievalResult",
    "TextChunkMetadata",
]


def __getattr__(name: str):
    """Lazy import for ChromaClient to avoid import errors when chromadb is not installed."""
    if name == "ChromaClient":
        from ordinis.rag.vectordb.chroma_client import ChromaClient
        return ChromaClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
