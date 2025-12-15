"""
Synapse - RAG Retrieval Engine.

Wraps RAG infrastructure with unified interface and Nemotron-Super synthesis.
"""

import time
from typing import TYPE_CHECKING, Any

from ordinis.ai.helix.models import ChatMessage
from ordinis.ai.synapse.config import SynapseConfig
from ordinis.ai.synapse.models import (
    IntentType,
    OrchestrationResult,
    RetrievalContext,
    RetrievalResultSet,
    SearchScope,
    Snippet,
)
from ordinis.core.logging import TraceContext
from ordinis.engines.base import (
    BaseEngine,
    EngineState,
    HealthLevel,
    HealthStatus,
)

if TYPE_CHECKING:
    from ordinis.ai.helix import Helix
    from ordinis.rag.retrieval.engine import RetrievalEngine


class Synapse(BaseEngine):
    """
    Synapse RAG retrieval engine.

    Provides unified interface for context retrieval and synthesis:
    - `retrieve(query, context)` for general retrieval
    - `synthesize(query, context)` for RAG synthesis using Nemotron-Super
    """

    def __init__(
        self,
        helix: "Helix",
        config: SynapseConfig | None = None,
    ):
        """
        Initialize Synapse.

        Args:
            helix: Helix instance for synthesis and embeddings
            config: Configuration options. Uses defaults if None.
        """
        super().__init__(config=config or SynapseConfig())
        self.helix = helix
        # self.config is already set by super().__init__ but typed as BaseEngineConfig
        # We can cast it or just use self._config

        # Validate configuration
        # We need to access the specific config object
        synapse_config = config or SynapseConfig()
        errors = synapse_config.validate()
        if errors:
            msg = f"Invalid Synapse configuration: {'; '.join(errors)}"
            raise ValueError(msg)

        self._rag_engine: RetrievalEngine | None = None

    async def _do_initialize(self) -> None:
        """Initialize Synapse engine."""
        self._logger.info("Initializing Synapse engine...")

        # Initialize RAG engine (lazy load)
        self._ensure_rag_engine()
        self._health_status = HealthStatus(
            level=HealthLevel.HEALTHY,
            message="Synapse engine initialized",
        )

    async def _do_shutdown(self) -> None:
        """Shutdown Synapse engine."""
        self._logger.info("Shutting down Synapse engine...")
        # Custom shutdown logic if needed

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        if self._state != EngineState.READY:
            return HealthStatus(
                level=HealthLevel.CRITICAL,
                message="Engine not initialized",
            )

        # Check Helix health
        helix_health = await self.helix.health_check()
        if helix_health.level != HealthLevel.HEALTHY:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message=f"Helix provider degraded: {helix_health.message}",
            )

        return self._health_status

    def _ensure_rag_engine(self) -> "RetrievalEngine":
        """Lazy initialize RAG engine."""
        if self._rag_engine is None:
            from ordinis.rag.retrieval.engine import (
                RetrievalEngine,
            )

            self._rag_engine = RetrievalEngine()
            self._logger.info("Synapse RAG engine initialized")

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

    async def synthesize(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        """
        Orchestrate the RAG process: Intent -> Retrieve -> Generate (Nemotron-Super).

        Args:
            query: User query
            context: Optional conversation context

        Returns:
            OrchestrationResult containing answer and sources
        """
        if self._state != EngineState.READY:
            await self.initialize()

        with TraceContext():
            start_time = time.time()
            self._logger.info("Starting synthesis", data={"query": query})

            # 1. Intent Recognition
            intent = await self._detect_intent(query)
            self._logger.info(f"Detected intent: {intent}")

            # 2. Retrieval (if needed)
            retrieval_result = None
            if intent in [IntentType.SEARCH, IntentType.CODE_GEN, IntentType.DEBUG]:
                scope = SearchScope.AUTO
                if intent == IntentType.CODE_GEN:
                    scope = SearchScope.CODE
                elif intent == IntentType.SEARCH:
                    scope = SearchScope.TEXT

                retrieval_result = self.retrieve(query, RetrievalContext(scope=scope))

            # 3. Generation using Nemotron-Super
            answer = await self._generate_response(query, intent, retrieval_result)

            latency = (time.time() - start_time) * 1000

            self._logger.info("Synthesis complete", data={"latency_ms": latency, "intent": intent})

            return OrchestrationResult(
                query=query,
                intent=intent,
                answer=answer,
                context=retrieval_result,
                latency_ms=latency,
            )

    async def _detect_intent(self, query: str) -> IntentType:
        """Detect intent using Helix."""
        prompt = f"""
        Analyze the following user query and classify it into one of these intents:
        - chat: General conversation, greetings, or questions not requiring external data.
        - search: Questions requiring factual knowledge, documentation, or specific information retrieval.
        - code_gen: Requests to write, generate, or modify code.
        - debug: Requests to fix errors, troubleshoot issues, or explain stack traces.
        - analysis: Complex requests requiring reasoning, synthesis, or market analysis.

        Query: "{query}"

        Return ONLY the intent label (chat, search, code_gen, debug, analysis).
        """

        try:
            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=10,
                model="nemotron-super",  # Fast model for classification
            )

            intent_str = response.content.strip().lower()

            # Map response to IntentType
            if "code" in intent_str and "gen" in intent_str:
                return IntentType.CODE_GEN
            if "debug" in intent_str:
                return IntentType.DEBUG
            if "chat" in intent_str:
                return IntentType.CHAT
            if "analysis" in intent_str:
                return IntentType.ANALYSIS

            return IntentType.SEARCH

        except Exception as e:
            self._logger.warning(f"Intent detection failed: {e}. Falling back to heuristic.")
            # Fallback heuristic
            query_lower = query.lower()
            if "code" in query_lower or "function" in query_lower or "class" in query_lower:
                return IntentType.CODE_GEN
            if "error" in query_lower or "fix" in query_lower or "debug" in query_lower:
                return IntentType.DEBUG
            if "hello" in query_lower or "hi" in query_lower:
                return IntentType.CHAT

            return IntentType.SEARCH

    async def _generate_response(
        self, query: str, intent: IntentType, context: RetrievalResultSet | None
    ) -> str:
        """Generate response using Helix (Nemotron-Super)."""

        # Construct prompt based on intent and context
        context_str = context.get_context_string() if context else "No external context provided."

        system_prompt = "You are Ordinis, an advanced AI trading assistant."

        if intent == IntentType.CODE_GEN:
            system_prompt += " You are an expert Python developer. Generate clean, efficient, and type-hinted code based on the user's request and the provided context."
        elif intent == IntentType.DEBUG:
            system_prompt += " You are an expert debugger. Analyze the error or issue and provide a solution, using the provided context if relevant."
        elif intent == IntentType.SEARCH:
            system_prompt += " You are a research assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, state that you don't know."
        elif intent == IntentType.ANALYSIS:
            system_prompt += " You are a financial analyst. Provide a detailed analysis based on the query and context."

        user_prompt = f"""
        Query: {query}

        Context:
        {context_str}
        """

        response = await self.helix.generate(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            model="nemotron-super",  # Use Nemotron-Super for synthesis
            temperature=0.2 if intent in [IntentType.CODE_GEN, IntentType.SEARCH] else 0.7,
        )

        return response.content

    def retrieve_for_codegen(
        self,
        query: str,
        language: str = "python",
        max_tokens: int = 2000,
    ) -> str:
        """
        Retrieve context specifically optimized for code generation.

        Args:
            query: The code generation request
            language: Target language
            max_tokens: Max tokens for context

        Returns:
            Formatted context string with relevant code snippets
        """
        # Search for code
        code_results = self.search_code(query, top_k=5)

        # Search for documentation/specs
        doc_results = self.search_docs(query, top_k=3)

        # Combine and format
        context_parts = []

        if code_results.snippets:
            context_parts.append("Existing Codebase Context:")
            for snippet in code_results.snippets:
                context_parts.append(f"File: {snippet.file_path}\n```\n{snippet.text}\n```")

        if doc_results.snippets:
            context_parts.append("\nRelevant Documentation:")
            for snippet in doc_results.snippets:
                context_parts.append(f"Source: {snippet.source}\n{snippet.text}")

        return "\n\n".join(context_parts)

    def get_code_patterns(self, pattern_type: str) -> list[str]:
        """
        Retrieve architectural patterns or templates.

        Args:
            pattern_type: Type of pattern (e.g., 'engine', 'provider', 'model')

        Returns:
            List of code pattern strings
        """
        # This would ideally query a specific collection or use metadata filtering
        # For now, we'll search for the pattern type in the codebase
        query = f"{pattern_type} pattern template structure"
        results = self.search_code(query, top_k=3)
        return [s.text for s in results.snippets]

    def find_similar_implementations(self, description: str) -> list[str]:
        """
        Find similar code implementations to avoid duplication.

        Args:
            description: Description of the functionality

        Returns:
            List of similar code snippets
        """
        results = self.search_code(description, top_k=3)
        return [f"File: {s.file_path}\n{s.text}" for s in results.snippets]

    def get_stats(self) -> dict:
        """Get retrieval engine statistics."""
        if self._state != EngineState.READY:
            return {"initialized": False}

        engine = self._ensure_rag_engine()
        stats = engine.get_stats()
        stats["synapse"] = {
            "initialized": self._state == EngineState.READY,
            "helix_enabled": True,
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
