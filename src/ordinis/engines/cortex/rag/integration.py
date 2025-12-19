"""
Cortex RAG integration helper.

Provides formatted context from ordinis.rag system for Cortex engine operations.
"""

from typing import Any

from loguru import logger

from ordinis.rag.retrieval.engine import RetrievalEngine


class CortexRAGHelper:
    """
    Helper class for integrating RAG into Cortex operations.

    Provides formatted context retrieval for hypothesis generation,
    code analysis, and other Cortex engine operations.
    """

    def __init__(self) -> None:
        """Initialize RAG helper with retrieval engine."""
        self.engine = RetrievalEngine()
        logger.info("CortexRAGHelper initialized")

    def get_kb_context(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """
        Get formatted knowledge base context for a query.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            filters: Optional metadata filters (e.g., {"domain": 8})

        Returns:
            Formatted context string
        """
        response = self.engine.query(
            query=query,
            query_type="text",
            top_k=top_k,
            filters=filters,
        )

        return self._format_kb_context(response)

    def get_code_examples(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """
        Get code examples from codebase.

        Args:
            query: Search query describing desired code
            top_k: Number of examples to retrieve
            filters: Optional metadata filters

        Returns:
            Formatted code examples
        """
        response = self.engine.query(
            query=query,
            query_type="code",
            top_k=top_k,
            filters=filters,
        )

        return self._format_code_examples(response)

    def get_hybrid_context(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Get both knowledge base and code context.

        Args:
            query: Search query
            top_k: Number of results per collection
            filters: Optional metadata filters

        Returns:
            Dict with 'kb_context' and 'code_context' keys
        """
        response = self.engine.query(
            query=query,
            query_type="hybrid",
            top_k=top_k,
            filters=filters,
        )

        # Split results by source collection
        kb_results = [r for r in response.results if r.metadata.get("collection") == "text_chunks"]
        code_results = [
            r for r in response.results if r.metadata.get("collection") == "code_chunks"
        ]

        return {
            "kb_context": self._format_results_as_kb(kb_results),
            "code_context": self._format_results_as_code(code_results),
        }

    def format_hypothesis_context(
        self,
        market_regime: str,
        strategy_type: str | None = None,
    ) -> dict[str, str]:
        """
        Get context for hypothesis generation.

        Args:
            market_regime: Market regime (trending, mean_reverting, etc.)
            strategy_type: Optional specific strategy type

        Returns:
            Dict with kb_context and code_context
        """
        # Build query
        if strategy_type:
            query = f"{strategy_type} strategy in {market_regime} market"
        else:
            query = f"trading strategies for {market_regime} market conditions"

        # Get hybrid context with domain filter for Strategy Design
        return self.get_hybrid_context(
            query=query,
            top_k=3,
            filters={"domain": 8},  # Domain 8 = Strategy Design
        )

    def format_code_analysis_context(
        self,
        analysis_type: str,
        code_snippet: str,
    ) -> str:
        """
        Get context for code analysis.

        Args:
            analysis_type: Type of analysis (review, optimize, explain)
            code_snippet: First ~100 chars of code being analyzed

        Returns:
            Formatted context with best practices and examples
        """
        # Build query based on analysis type
        if analysis_type == "review":
            query = f"code review best practices patterns {code_snippet}"
        elif analysis_type == "optimize":
            query = f"performance optimization techniques {code_snippet}"
        elif analysis_type == "explain":
            query = f"code explanation documentation {code_snippet}"
        else:
            query = f"code analysis {analysis_type} {code_snippet}"

        # Get hybrid context
        context = self.get_hybrid_context(query=query, top_k=3)

        # Combine contexts
        combined = ""
        if context["kb_context"]:
            combined += f"Best Practices:\n{context['kb_context']}\n\n"
        if context["code_context"]:
            combined += f"Code Examples:\n{context['code_context']}"

        return combined.strip()

    def format_risk_context(
        self,
        risk_type: str,
        current_metrics: dict[str, Any],
    ) -> str:
        """
        Get context for risk analysis.

        Args:
            risk_type: Type of risk (position, portfolio, drawdown, etc.)
            current_metrics: Current risk metrics

        Returns:
            Formatted risk management context
        """
        query = f"{risk_type} risk management limits thresholds"

        context = self.get_kb_context(
            query=query,
            top_k=5,
            filters={"domain": 3},  # Domain 3 = Risk Management
        )

        return context

    def _format_kb_context(self, response: Any) -> str:
        """Format KB query response as context string."""
        if not response.results:
            return ""

        context_parts = []
        for i, result in enumerate(response.results, 1):
            source = result.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] Source: {source}\n{result.text}\n")

        return "\n".join(context_parts)

    def _format_code_examples(self, response: Any) -> str:
        """Format code query response as examples."""
        if not response.results:
            return ""

        examples = []
        for i, result in enumerate(response.results, 1):
            file_path = result.metadata.get("file_path", "Unknown")
            func_name = result.metadata.get("function_name", "")
            class_name = result.metadata.get("class_name", "")

            location = f"{file_path}"
            if class_name:
                location += f" :: {class_name}"
            if func_name:
                location += f".{func_name}" if class_name else f" :: {func_name}"

            examples.append(f"Example {i} ({location}):\n```python\n{result.text}\n```\n")

        return "\n".join(examples)

    def _format_results_as_kb(self, results: list) -> str:
        """Format results as KB context."""
        if not results:
            return ""

        parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "Unknown")
            parts.append(f"[{i}] {source}: {result.text}")

        return "\n".join(parts)

    def _format_results_as_code(self, results: list) -> str:
        """Format results as code examples."""
        if not results:
            return ""

        examples = []
        for i, result in enumerate(results, 1):
            file_path = result.metadata.get("file_path", "Unknown")
            examples.append(f"[{i}] {file_path}:\n```python\n{result.text}\n```")

        return "\n".join(examples)
