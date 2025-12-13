"""Main retrieval engine coordinating embeddings, search, and reranking."""

import time

from loguru import logger

from ordinis.rag.config import get_config
from ordinis.rag.embedders.code_embedder import CodeEmbedder
from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.retrieval.query_classifier import QueryType, classify_query
from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.vectordb.schema import QueryResponse, RetrievalResult


class RetrievalEngine:
    """Main RAG retrieval engine."""

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        text_embedder: TextEmbedder | None = None,
        code_embedder: CodeEmbedder | None = None,
    ):
        """Initialize retrieval engine.

        Args:
            chroma_client: ChromaDB client (created if None)
            text_embedder: Text embedder (created if None)
            code_embedder: Code embedder (created if None)
        """
        self.config = get_config()
        self.chroma_client = chroma_client or ChromaClient()
        self.text_embedder = text_embedder or TextEmbedder()
        self.code_embedder = code_embedder or CodeEmbedder()

        logger.info("Retrieval engine initialized")

    def query(
        self,
        query: str,
        query_type: QueryType | str | None = None,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> QueryResponse:
        """Execute RAG query.

        Args:
            query: Query text
            query_type: Query type (auto-detected if None)
            top_k: Number of results (uses config default if None)
            filters: Metadata filters

        Returns:
            QueryResponse with results and metadata
        """
        start_time = time.time()

        # Classify query if not specified
        if query_type is None:
            detected_type = classify_query(query)
        else:
            detected_type = QueryType(query_type) if isinstance(query_type, str) else query_type

        top_k = top_k or self.config.top_k_rerank

        logger.info(f"Processing {detected_type.value} query: {query[:50]}...")

        # Retrieve candidates based on query type
        if detected_type == QueryType.TEXT:
            results = self._query_text(query, filters)
        elif detected_type == QueryType.CODE:
            results = self._query_code(query, filters)
        else:  # HYBRID
            results = self._query_hybrid(query, filters)

        # Filter by similarity threshold
        threshold = self.config.similarity_threshold
        filtered_results = [r for r in results if r.score >= threshold]

        # Take top-k after filtering
        final_results = filtered_results[:top_k]

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Query complete: {len(filtered_results)} results above threshold, "
            f"returning top-{len(final_results)} in {latency_ms:.0f}ms"
        )

        return QueryResponse(
            query=query,
            query_type=detected_type.value,
            results=final_results,
            execution_time_ms=latency_ms,
            total_candidates=len(results),
        )

    def _query_text(self, query: str, filters: dict | None = None) -> list[RetrievalResult]:
        """Query text collection.

        Args:
            query: Query text
            filters: Metadata filters

        Returns:
            List of retrieval results
        """
        # Embed query
        query_embedding = self.text_embedder.embed(query)

        # Search ChromaDB
        results = self.chroma_client.query_texts(
            query_embedding=query_embedding,
            top_k=self.config.top_k_retrieval,
            where=filters,
        )

        logger.debug(f"Text search: {len(results)} candidates")
        return results

    def _query_code(self, query: str, filters: dict | None = None) -> list[RetrievalResult]:
        """Query code collection.

        Args:
            query: Query text
            filters: Metadata filters

        Returns:
            List of retrieval results
        """
        # Embed query
        query_embedding = self.code_embedder.embed(query)

        # Search ChromaDB
        results = self.chroma_client.query_code(
            query_embedding=query_embedding,
            top_k=self.config.top_k_retrieval,
            where=filters,
        )

        logger.debug(f"Code search: {len(results)} candidates")
        return results

    def _query_hybrid(self, query: str, filters: dict | None = None) -> list[RetrievalResult]:
        """Query both text and code collections.

        Args:
            query: Query text
            filters: Metadata filters

        Returns:
            Merged list of retrieval results
        """
        # Query both collections
        text_results = self._query_text(query, filters)
        code_results = self._query_code(query, filters)

        # Merge and sort by score
        all_results = text_results + code_results
        all_results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            f"Hybrid search: {len(text_results)} text + {len(code_results)} code = "
            f"{len(all_results)} total"
        )

        return all_results

    def get_stats(self) -> dict:
        """Get retrieval engine statistics.

        Returns:
            Dictionary with engine stats
        """
        chroma_stats = self.chroma_client.get_stats()

        return {
            "chroma": chroma_stats,
            "text_embedder_available": self.text_embedder.is_available(),
            "code_embedder_available": self.code_embedder.is_available(),
            "config": {
                "top_k_retrieval": self.config.top_k_retrieval,
                "top_k_rerank": self.config.top_k_rerank,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }
