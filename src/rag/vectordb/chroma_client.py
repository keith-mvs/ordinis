"""ChromaDB client wrapper for vector storage and retrieval."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger
import numpy as np

from rag.config import get_config
from rag.vectordb.schema import RetrievalResult


class ChromaClient:
    """ChromaDB client for managing text and code collections."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        text_collection: str | None = None,
        code_collection: str | None = None,
    ):
        """Initialize ChromaDB client.

        Args:
            persist_directory: Directory for ChromaDB persistence
            text_collection: Name of text embedding collection
            code_collection: Name of code embedding collection
        """
        config = get_config()
        self.persist_directory = persist_directory or config.chroma_persist_directory
        self.text_collection_name = text_collection or config.text_collection_name
        self.code_collection_name = code_collection or config.code_collection_name

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._text_collection = None
        self._code_collection = None

        logger.info(f"ChromaDB client initialized at {self.persist_directory}")

    def get_text_collection(self):
        """Get or create text embedding collection."""
        if self._text_collection is None:
            self._text_collection = self.client.get_or_create_collection(
                name=self.text_collection_name,
                metadata={"description": "Knowledge base text embeddings"},
            )
            logger.info(f"Text collection '{self.text_collection_name}' loaded/created")
        return self._text_collection

    def get_code_collection(self):
        """Get or create code embedding collection."""
        if self._code_collection is None:
            self._code_collection = self.client.get_or_create_collection(
                name=self.code_collection_name,
                metadata={"description": "Codebase embeddings"},
            )
            logger.info(f"Code collection '{self.code_collection_name}' loaded/created")
        return self._code_collection

    def add_texts(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """Add text documents to collection.

        Args:
            texts: List of text documents
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
            metadata: List of metadata dicts
            ids: Optional list of document IDs (auto-generated if None)
        """
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            msg = "Texts, embeddings, and metadata must have same length"
            raise ValueError(msg)

        collection = self.get_text_collection()

        # Generate IDs if not provided
        if ids is None:
            ids = [f"text_{i}" for i in range(collection.count(), collection.count() + len(texts))]

        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
            ids=ids,
        )

        logger.info(f"Added {len(texts)} text documents to collection")

    def add_code(
        self,
        code: list[str],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> None:
        """Add code documents to collection.

        Args:
            code: List of code snippets
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
            metadata: List of metadata dicts
            ids: Optional list of document IDs (auto-generated if None)
        """
        if len(code) != len(embeddings) or len(code) != len(metadata):
            msg = "Code, embeddings, and metadata must have same length"
            raise ValueError(msg)

        collection = self.get_code_collection()

        # Generate IDs if not provided
        if ids is None:
            ids = [f"code_{i}" for i in range(collection.count(), collection.count() + len(code))]

        collection.add(
            documents=code,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
            ids=ids,
        )

        logger.info(f"Added {len(code)} code documents to collection")

    def query_texts(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query text collection.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filters

        Returns:
            List of retrieval results
        """
        collection = self.get_text_collection()

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )

        return self._format_results(results)

    def query_code(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query code collection.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional metadata filters

        Returns:
            List of retrieval results
        """
        collection = self.get_code_collection()

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )

        return self._format_results(results)

    def _format_results(self, raw_results: dict) -> list[RetrievalResult]:
        """Format ChromaDB results into RetrievalResult objects.

        Args:
            raw_results: Raw results from ChromaDB query

        Returns:
            List of formatted retrieval results
        """
        results = []

        # ChromaDB returns nested lists
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for id_, doc, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            # Convert distance to similarity score (cosine similarity)
            # ChromaDB uses L2 distance by default, convert to similarity
            score = 1.0 - (distance / 2.0)  # Approximate conversion

            results.append(
                RetrievalResult(
                    id=id_,
                    text=doc,
                    score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                    metadata=metadata or {},
                )
            )

        return results

    def reset(self) -> None:
        """Reset all collections (WARNING: deletes all data)."""
        logger.warning("Resetting ChromaDB - all data will be deleted")
        self.client.reset()
        self._text_collection = None
        self._code_collection = None
        logger.info("ChromaDB reset complete")

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        text_count = self.get_text_collection().count()
        code_count = self.get_code_collection().count()

        return {
            "text_collection": self.text_collection_name,
            "text_documents": text_count,
            "code_collection": self.code_collection_name,
            "code_documents": code_count,
            "persist_directory": str(self.persist_directory),
        }
