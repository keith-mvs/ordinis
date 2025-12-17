"""Abstract interface for vector databases (VDB-agnostic RAG)."""

from abc import ABC, abstractmethod
from typing import Any


class VectorDatabaseInterface(ABC):
    """
    Abstract interface for vector database operations.

    Allows seamless swapping between ChromaDB, Milvus, Weaviate, etc.
    """

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector database.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            collection_name: Name of collection to query
            filters: Optional metadata filters

        Returns:
            List of results with score, text, metadata
        """

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
        collection_name: str,
        ids: list[str] | None = None,
    ) -> None:
        """
        Add documents to the vector database.

        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            collection_name: Target collection name
            ids: Optional document IDs
        """

    @abstractmethod
    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
    ) -> None:
        """Delete documents by ID."""

    @abstractmethod
    def get_collection(
        self,
        collection_name: str,
    ) -> dict[str, Any]:
        """Get collection metadata and stats."""

    @abstractmethod
    def check_collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Check if collection exists."""

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new collection."""

    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """Delete a collection."""

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        """
        Check database health.

        Returns:
            Dict with status, latency, error info
        """

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with collection counts, dimensions, etc.
        """

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
