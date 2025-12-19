"""Query type classification for RAG system."""

from enum import Enum


class QueryType(str, Enum):
    """Query type classification."""

    TEXT = "text"
    CODE = "code"
    HYBRID = "hybrid"


# Keywords that indicate code-related queries
CODE_KEYWORDS = {
    "implement",
    "code",
    "function",
    "class",
    "method",
    "example",
    "show me",
    "def ",
    "class ",
    "async def",
    "return",
    "import",
    "module",
    "package",
    "api",
    "endpoint",
}

# Keywords that indicate hybrid queries (both concepts and code)
HYBRID_KEYWORDS = {
    "design",
    "architecture",
    "pattern",
    "implementation",
    "approach",
    "structure",
    "how to",
    "best practice",
}


def classify_query(query: str) -> QueryType:
    """Classify query as text, code, or hybrid.

    Uses keyword-based heuristic. In production, could be enhanced
    with a small classifier model.

    Args:
        query: User query string

    Returns:
        QueryType enum value
    """
    query_lower = query.lower()

    # Check for hybrid keywords first (more specific)
    if any(keyword in query_lower for keyword in HYBRID_KEYWORDS):
        return QueryType.HYBRID

    # Check for code keywords
    if any(keyword in query_lower for keyword in CODE_KEYWORDS):
        return QueryType.CODE

    # Default to text query
    return QueryType.TEXT
