"""
Synapse data models.

Defines request/response structures for retrieval operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchScope(Enum):
    """Search scope for retrieval."""

    TEXT = "text"  # Knowledge base documents
    CODE = "code"  # Source code
    HYBRID = "hybrid"  # Both text and code
    AUTO = "auto"  # Auto-detect based on query


class IntentType(Enum):
    """Detected intent of the user query."""

    CHAT = "chat"  # General conversation
    SEARCH = "search"  # Knowledge retrieval
    CODE_GEN = "code_gen"  # Code generation request
    DEBUG = "debug"  # Debugging assistance
    ANALYSIS = "analysis"  # Complex reasoning/analysis


@dataclass
class RetrievalContext:
    """Context for retrieval query."""

    # Scope
    scope: SearchScope = SearchScope.AUTO
    domain: str | None = None  # e.g., "trading", "risk", "portfolio"
    engine: str | None = None  # Specific engine to search

    # Filtering
    filters: dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0

    # Results
    top_k: int = 5
    include_metadata: bool = True

    # Source context
    source_file: str | None = None  # Current file for code context
    source_function: str | None = None  # Current function


@dataclass
class Snippet:
    """Retrieved snippet with metadata."""

    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Code-specific
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    function_name: str | None = None
    class_name: str | None = None

    # Text-specific
    domain: str | None = None
    section: str | None = None

    def __str__(self) -> str:
        location = self.file_path or self.domain or "unknown"
        return f"[{self.score:.2f}] {location}: {self.text[:80]}..."


@dataclass
class RetrievalResultSet:
    """Result set from retrieval query."""

    query: str
    snippets: list[Snippet]
    scope: SearchScope
    execution_time_ms: float
    total_candidates: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Statistics
    text_results: int = 0
    code_results: int = 0

    @property
    def count(self) -> int:
        """Number of snippets returned."""
        return len(self.snippets)

    @property
    def top_score(self) -> float:
        """Highest similarity score."""
        return self.snippets[0].score if self.snippets else 0.0

    def get_context_string(self, max_tokens: int = 2000) -> str:
        """
        Format snippets as context string for LLM prompts.

        Args:
            max_tokens: Approximate maximum tokens (4 chars per token)

        Returns:
            Formatted context string
        """
        max_chars = max_tokens * 4
        parts: list[str] = []
        current_chars = 0

        for i, snippet in enumerate(self.snippets, 1):
            entry = f"[{i}] {snippet.source}: {snippet.text}"
            if current_chars + len(entry) > max_chars:
                break
            parts.append(entry)
            current_chars += len(entry) + 2  # +2 for newlines

        return "\n\n".join(parts)

    def to_citations(self) -> list[dict[str, Any]]:
        """
        Convert to citation format for responses.

        Returns:
            List of citation dictionaries
        """
        return [
            {
                "id": s.id,
                "source": s.source,
                "score": s.score,
                "file": s.file_path,
                "lines": f"{s.line_start}-{s.line_end}" if s.line_start else None,
            }
            for s in self.snippets
        ]


@dataclass
class OrchestrationResult:
    """Result of the orchestration process."""

    query: str
    intent: IntentType
    answer: str
    context: RetrievalResultSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
