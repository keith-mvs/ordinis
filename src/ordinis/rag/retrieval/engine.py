"""Main retrieval engine coordinating embeddings, search, and reranking."""

from dataclasses import dataclass
from enum import Enum
import re
import time

from loguru import logger

from ordinis.rag.config import get_config
from ordinis.rag.embedders.code_embedder import CodeEmbedder
from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.retrieval.query_classifier import QueryType, classify_query
from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.vectordb.schema import QueryResponse, RetrievalResult

try:
    from sentence_transformers import CrossEncoder

    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False


# ============================================================================
# Error Handling for Non-Vectored Data
# ============================================================================


class RAGErrorType(Enum):
    """Types of RAG retrieval errors."""

    EMPTY_COLLECTION = "empty_collection"
    EMBEDDING_FAILED = "embedding_failed"
    QUERY_TOO_SHORT = "query_too_short"
    NO_RESULTS = "no_results"
    CHROMA_ERROR = "chroma_error"
    UNKNOWN = "unknown"


@dataclass
class RAGError:
    """Structured RAG error with recovery suggestions."""

    error_type: RAGErrorType
    message: str
    suggestion: str
    recoverable: bool = True

    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


class RAGErrorHandler:
    """Handle errors when collections are empty or queries fail."""

    @staticmethod
    def check_collection_status(client: ChromaClient) -> RAGError | None:
        """Check if collections are properly populated.

        Returns:
            RAGError if collection is empty/missing, None if OK
        """
        try:
            text_count = client.get_text_collection().count()
            code_count = client.get_code_collection().count()

            if text_count == 0 and code_count == 0:
                return RAGError(
                    error_type=RAGErrorType.EMPTY_COLLECTION,
                    message="Knowledge base not indexed - ChromaDB collections are empty",
                    suggestion=(
                        "Run: python scripts/index_knowledge_base.py\n"
                        "Or: python -m ordinis.rag.pipeline.kb_indexer"
                    ),
                    recoverable=True,
                )
            return None
        except Exception as e:
            return RAGError(
                error_type=RAGErrorType.CHROMA_ERROR,
                message=f"ChromaDB connection failed: {e}",
                suggestion="Check ChromaDB is running and persist directory exists",
                recoverable=False,
            )

    @staticmethod
    def validate_query(query: str, min_length: int = 3) -> RAGError | None:
        """Validate query is suitable for vector search.

        Returns:
            RAGError if query is invalid, None if OK
        """
        if not query or not query.strip():
            return RAGError(
                error_type=RAGErrorType.QUERY_TOO_SHORT,
                message="Query is empty",
                suggestion="Provide a query with at least a few words",
                recoverable=True,
            )

        stripped = query.strip()
        if len(stripped) < min_length:
            return RAGError(
                error_type=RAGErrorType.QUERY_TOO_SHORT,
                message=f"Query too short ({len(stripped)} chars)",
                suggestion=f"Query should be at least {min_length} characters",
                recoverable=True,
            )

        return None

    @staticmethod
    def handle_no_results(query: str) -> RAGError:
        """Create error for no results found."""
        return RAGError(
            error_type=RAGErrorType.NO_RESULTS,
            message=f"No results found for query: {query[:50]}...",
            suggestion=(
                "Try:\n"
                "- Using different keywords\n"
                "- Broadening the search terms\n"
                "- Checking if the topic is in the knowledge base"
            ),
            recoverable=True,
        )


class RetrievalEngine:
    """Main RAG retrieval engine with reranking and multi-collection support."""

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        text_embedder: TextEmbedder | None = None,
        code_embedder: CodeEmbedder | None = None,
        enable_reranking: bool = True,
    ):
        """Initialize retrieval engine.

        Args:
            chroma_client: ChromaDB client (created if None)
            text_embedder: Text embedder (created if None)
            code_embedder: Code embedder (created if None)
            enable_reranking: Enable cross-encoder reranking
        """
        self.config = get_config()
        self.chroma_client = chroma_client or ChromaClient()
        self.text_embedder = text_embedder or TextEmbedder()
        self.code_embedder = code_embedder or CodeEmbedder()
        self.enable_reranking = enable_reranking and HAS_RERANKER

        # Initialize reranker if available
        self.reranker = None
        if self.enable_reranking:
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.enable_reranking = False

        logger.info("Retrieval engine initialized")

    def query(
        self,
        query: str,
        query_type: QueryType | str | None = None,
        top_k: int | None = None,
        filters: dict | None = None,
        collections: list[str] | None = None,
    ) -> QueryResponse:
        """Execute RAG query with optional reranking and multi-collection support.

        Args:
            query: Query text
            query_type: Query type (auto-detected if None)
            top_k: Number of results (uses config default if None)
            filters: Metadata filters
            collections: Query specific collections (if None, uses default)

        Returns:
            QueryResponse with results and metadata
        """
        start_time = time.time()

        # Check for complex queries and decompose if needed
        if self._is_complex_query(query):
            logger.info(f"Complex query detected, decomposing: {query[:50]}...")
            subqueries = self._decompose_query(query)
            return self._query_decomposed(query, subqueries, top_k, filters, collections)

        # Classify query if not specified
        if query_type is None:
            detected_type = classify_query(query)
        else:
            detected_type = QueryType(query_type) if isinstance(query_type, str) else query_type

        top_k_retrieval = self.config.top_k_retrieval
        top_k_final = top_k or self.config.top_k_rerank

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

        # Rerank if enabled
        if self.enable_reranking and filtered_results:
            filtered_results = self._rerank_results(query, filtered_results, top_k_retrieval)

        # Take top-k after filtering/reranking
        final_results = filtered_results[:top_k_final]

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
            "reranker_enabled": self.enable_reranking,
            "config": {
                "top_k_retrieval": self.config.top_k_retrieval,
                "top_k_rerank": self.config.top_k_rerank,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }

    # ========================================================================
    # Advanced features: reranking, decomposition, multi-collection
    # ========================================================================

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder model.

        Args:
            query: Original query
            results: Results to rerank
            top_k: Number to return

        Returns:
            Reranked results
        """
        if not self.reranker or not results:
            return results

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, r.text] for r in results]

            # Get reranking scores
            scores = self.reranker.predict(pairs)

            # Combine and sort
            ranked = sorted(
                zip(results, scores, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )

            reranked = [r for r, _ in ranked[:top_k]]
            logger.debug(f"Reranked {len(results)} results -> {len(reranked)}")

            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original results")
            return results

    def _is_complex_query(self, query: str) -> bool:
        """Detect if query requires decomposition.

        Complex queries typically have:
        - Multiple questions (? marks)
        - Conjunctions (and, or)
        - Long length (>100 chars)
        """
        question_count = query.count("?")
        has_conjunctions = bool(re.search(r"\b(and|or|also|both)\b", query.lower()))
        is_long = len(query) > 100

        return (question_count > 1) or (has_conjunctions and is_long)

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose complex query into simpler subqueries.

        Args:
            query: Complex query

        Returns:
            List of subqueries
        """
        subqueries = []

        # Split by question marks
        questions = [q.strip() for q in query.split("?") if q.strip()]
        if len(questions) > 1:
            subqueries.extend(questions)
        else:
            # Split by conjunctions
            parts = re.split(r"\b(?:and|or|also)\b", query, flags=re.IGNORECASE)
            subqueries.extend([p.strip() for p in parts if p.strip()])

        # If no split occurred, return original
        if not subqueries or len(subqueries) == 1:
            subqueries = [query]

        logger.info(f"Decomposed query into {len(subqueries)} subqueries")
        return subqueries[:5]  # Limit to 5 subqueries

    def _query_decomposed(
        self,
        original_query: str,
        subqueries: list[str],
        top_k: int | None,
        filters: dict | None,
        collections: list[str] | None,
    ) -> QueryResponse:
        """Execute decomposed query.

        Args:
            original_query: Original complex query
            subqueries: List of subqueries
            top_k: Number of results
            filters: Metadata filters
            collections: Specific collections to query

        Returns:
            Combined QueryResponse
        """
        all_results = []

        for subquery in subqueries:
            logger.info(f"Processing subquery: {subquery[:50]}...")

            # Recursively query each subquery (without decomposing again)
            response = self.query(
                subquery,
                query_type=classify_query(subquery),
                top_k=top_k or self.config.top_k_rerank,
                filters=filters,
                collections=collections,
            )

            all_results.extend(response.results)

        # Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.id not in seen_ids:
                unique_results.append(r)
                seen_ids.add(r.id)

        final_top_k = top_k or self.config.top_k_rerank

        return QueryResponse(
            query=original_query,
            query_type="hybrid",
            results=unique_results[:final_top_k],
            execution_time_ms=0,  # Approximate
            total_candidates=len(all_results),
        )

    def safe_query(
        self,
        query: str,
        query_type: QueryType | str | None = None,
        top_k: int | None = None,
        filters: dict | None = None,
        collections: list[str] | None = None,
    ) -> tuple[QueryResponse | None, RAGError | None]:
        """Execute query with comprehensive error handling.

        Safe wrapper around query() that handles:
        - Empty collections (not indexed)
        - Invalid queries
        - Embedding failures
        - ChromaDB errors

        Args:
            query: Query text
            query_type: Query type (auto-detected if None)
            top_k: Number of results
            filters: Metadata filters
            collections: Specific collections to query

        Returns:
            Tuple of (QueryResponse, None) on success
            Tuple of (None, RAGError) on failure

        Example:
            >>> response, error = engine.safe_query("options strategies")
            >>> if error:
            ...     print(f"Error: {error.message}")
            ...     print(f"Suggestion: {error.suggestion}")
            ... else:
            ...     print(f"Found {len(response.results)} results")
        """
        # Validate query
        query_error = RAGErrorHandler.validate_query(query)
        if query_error:
            logger.warning(f"Query validation failed: {query_error}")
            return None, query_error

        # Check collection status
        collection_error = RAGErrorHandler.check_collection_status(self.chroma_client)
        if collection_error:
            logger.warning(f"Collection check failed: {collection_error}")
            return None, collection_error

        try:
            # Execute query
            response = self.query(
                query=query,
                query_type=query_type,
                top_k=top_k,
                filters=filters,
                collections=collections,
            )

            # Check for empty results
            if not response.results:
                return None, RAGErrorHandler.handle_no_results(query)

            return response, None

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None, RAGError(
                error_type=RAGErrorType.UNKNOWN,
                message=f"Query failed: {e}",
                suggestion="Check logs for details and ensure ChromaDB is running",
                recoverable=False,
            )

    def get_collection_status(self) -> dict:
        """Get detailed collection status for diagnostics.

        Returns:
            Dictionary with collection info and any errors
        """
        status = {
            "healthy": True,
            "text_collection": {"name": None, "count": 0, "error": None},
            "code_collection": {"name": None, "count": 0, "error": None},
            "suggestion": None,
        }

        try:
            text_coll = self.chroma_client.get_text_collection()
            status["text_collection"]["name"] = text_coll.name
            status["text_collection"]["count"] = text_coll.count()
        except Exception as e:
            status["text_collection"]["error"] = str(e)
            status["healthy"] = False

        try:
            code_coll = self.chroma_client.get_code_collection()
            status["code_collection"]["name"] = code_coll.name
            status["code_collection"]["count"] = code_coll.count()
        except Exception as e:
            status["code_collection"]["error"] = str(e)
            status["healthy"] = False

        total = status["text_collection"]["count"] + status["code_collection"]["count"]

        if total == 0:
            status["healthy"] = False
            status["suggestion"] = (
                "Collections are empty. Run:\n  python scripts/index_knowledge_base.py"
            )

        return status
