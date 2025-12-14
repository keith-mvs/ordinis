"""
Synapse - RAG Retrieval Engine.

Wraps RAG infrastructure with unified interface.
"""

import logging
import time
from typing import TYPE_CHECKING

from ordinis.ai.synapse.config import SynapseConfig
from ordinis.ai.synapse.models import (
    RetrievalContext,
    RetrievalResultSet,
    SearchScope,
    Snippet,
)

if TYPE_CHECKING:
    from ordinis.ai.helix import Helix
    from ordinis.rag.retrieval.engine import RetrievalEngine

_logger = logging.getLogger(__name__)


class Synapse:
    """
    Synapse RAG retrieval engine.

    Provides unified interface for context retrieval:
    - `retrieve(query, context)` for general retrieval
    - `retrieve_for_prompt(query)` for LLM context
    """

    def __init__(
        self,
        config: SynapseConfig | None = None,
        helix: "Helix | None" = None,
    ):
        """
        Initialize Synapse.

        Args:
            config: Configuration options. Uses defaults if None.
            helix: Helix instance for embeddings (optional)
        """
        self.config = config or SynapseConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            msg = f"Invalid Synapse configuration: {'; '.join(errors)}"
            raise ValueError(msg)

        self._helix = helix
        self._rag_engine: RetrievalEngine | None = None
        self._initialized = False

    def _ensure_rag_engine(self) -> "RetrievalEngine":
        """Lazy initialize RAG engine."""
        if self._rag_engine is None:
            from ordinis.rag.retrieval.engine import (
                RetrievalEngine,
            )

            self._rag_engine = RetrievalEngine()
            self._initialized = True
            _logger.info("Synapse RAG engine initialized")

        return self._rag_engine

    def _scope_to_query_type(self, scope: SearchScope) -> str | None:
        """Convert SearchScope to RAG query type."""
        if scope == SearchScope.TEXT:
            return "text"
        if scope == SearchScope.CODE:
            return "code"
        if scope == SearchScope.HYBRID:
            return "hybrid"
        return None  # AUTO

    def _convert_to_snippets(
        self,
        results: list,
        scope: SearchScope,
    ) -> tuple[list[Snippet], int, int]:
        """
        Convert RAG RetrievalResults to Snippets.

        Returns:
            Tuple of (snippets, text_count, code_count)
        """
        snippets: list[Snippet] = []
        text_count = 0
        code_count = 0

        for result in results:
            metadata = result.metadata or {}

            # Determine if text or code based on metadata
            is_code = "file_path" in metadata or "function_name" in metadata

            if is_code:
                code_count += 1
                snippet = Snippet(
                    id=result.id,
                    text=result.text,
                    score=result.score,
                    source=metadata.get("file_path", "code"),
                    metadata=metadata,
                    file_path=metadata.get("file_path"),
                    line_start=metadata.get("line_start"),
                    line_end=metadata.get("line_end"),
                    function_name=metadata.get("function_name"),
                    class_name=metadata.get("class_name"),
                )
            else:
                text_count += 1
                snippet = Snippet(
                    id=result.id,
                    text=result.text,
                    score=result.score,
                    source=metadata.get("source", "kb"),
                    metadata=metadata,
                    domain=str(metadata.get("domain", "")),
                    section=metadata.get("section"),
                )

            snippets.append(snippet)

        return snippets, text_count, code_count

    def retrieve(
        self,
        query: str,
        context: RetrievalContext | dict | None = None,
    ) -> RetrievalResultSet:
        """
        Retrieve relevant context for a query.

        Args:
            query: Query text
            context: Retrieval context (scope, filters, etc.)

        Returns:
            RetrievalResultSet with snippets
        """
        # Normalize context
        if context is None:
            ctx = RetrievalContext()
        elif isinstance(context, dict):
            ctx = RetrievalContext(**context)
        else:
            ctx = context

        start_time = time.perf_counter()

        # Get RAG engine
        engine = self._ensure_rag_engine()

        # Build filters from context
        filters = dict(ctx.filters) if ctx.filters else None
        if ctx.engine:
            filters = filters or {}
            filters["engine"] = ctx.engine

        # Execute query
        query_type = self._scope_to_query_type(ctx.scope)
        response = engine.query(
            query=query,
            query_type=query_type,
            top_k=ctx.top_k,
            filters=filters,
        )

        # Filter by min_score
        filtered_results = [r for r in response.results if r.score >= ctx.min_score]

        # Convert to snippets
        snippets, text_count, code_count = self._convert_to_snippets(filtered_results, ctx.scope)

        # Determine actual scope
        actual_scope = ctx.scope
        if ctx.scope == SearchScope.AUTO:
            # Infer from results
            if text_count > 0 and code_count > 0:
                actual_scope = SearchScope.HYBRID
            elif code_count > 0:
                actual_scope = SearchScope.CODE
            else:
                actual_scope = SearchScope.TEXT

        latency_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResultSet(
            query=query,
            snippets=snippets,
            scope=actual_scope,
            execution_time_ms=latency_ms,
            total_candidates=response.total_candidates,
            text_results=text_count,
            code_results=code_count,
        )

    def retrieve_for_prompt(
        self,
        query: str,
        max_tokens: int | None = None,
        scope: SearchScope = SearchScope.AUTO,
    ) -> str:
        """
        Retrieve and format context for LLM prompts.

        Args:
            query: Query text
            max_tokens: Maximum tokens for context
            scope: Search scope

        Returns:
            Formatted context string
        """
        ctx = RetrievalContext(
            scope=scope,
            top_k=self.config.default_top_k,
            min_score=self.config.similarity_threshold,
        )

        result = self.retrieve(query, ctx)

        tokens = max_tokens or self.config.max_context_tokens
        return result.get_context_string(max_tokens=tokens)

    def search_code(
        self,
        query: str,
        engine: str | None = None,
        top_k: int = 5,
    ) -> RetrievalResultSet:
        """
        Search code specifically.

        Args:
            query: Query text
            engine: Filter by engine name
            top_k: Number of results

        Returns:
            Code retrieval results
        """
        ctx = RetrievalContext(
            scope=SearchScope.CODE,
            engine=engine,
            top_k=top_k,
        )
        return self.retrieve(query, ctx)

    def search_docs(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 5,
    ) -> RetrievalResultSet:
        """
        Search documentation specifically.

        Args:
            query: Query text
            domain: Filter by domain
            top_k: Number of results

        Returns:
            Documentation retrieval results
        """
        filters = {}
        if domain:
            filters["domain"] = domain

        ctx = RetrievalContext(
            scope=SearchScope.TEXT,
            filters=filters,
            top_k=top_k,
        )
        return self.retrieve(query, ctx)

    def get_stats(self) -> dict:
        """Get retrieval engine statistics."""
        if not self._initialized:
            return {"initialized": False}

        engine = self._ensure_rag_engine()
        stats = engine.get_stats()
        stats["synapse"] = {
            "initialized": self._initialized,
            "helix_enabled": self._helix is not None,
            "config": {
                "default_scope": self.config.default_scope.value,
                "default_top_k": self.config.default_top_k,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }
        return stats

    @property
    def is_available(self) -> bool:
        """Check if Synapse is available."""
        try:
            self._ensure_rag_engine()
            return True
        except Exception:
            return False
