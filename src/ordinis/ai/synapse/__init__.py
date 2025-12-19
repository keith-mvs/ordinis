"""
Synapse - RAG Retrieval Engine for Ordinis.

Provides context retrieval for AI-powered analysis:
- Knowledge base retrieval (documentation, strategies, market analysis)
- Code retrieval (engine implementations, patterns)
- Hybrid search across both

Synapse wraps the existing RAG infrastructure and provides
a unified interface for retrieval operations.

Example:
    from ordinis.ai.synapse import Synapse, SynapseConfig

    config = SynapseConfig()
    synapse = Synapse(config)

    # Retrieve context for a query
    result = synapse.retrieve(
        query="How does the RiskGuard engine evaluate positions?",
        context={"domain": "code"},
    )
    for snippet in result.snippets:
        print(f"[{snippet.score:.2f}] {snippet.text[:100]}...")
"""

from ordinis.ai.synapse.config import SynapseConfig
from ordinis.ai.synapse.engine import Synapse
from ordinis.ai.synapse.models import (
    RetrievalContext,
    RetrievalResultSet,
    SearchScope,
    Snippet,
)

# Re-export from RAG for convenience
from ordinis.rag.vectordb.schema import (
    CodeChunkMetadata,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    TextChunkMetadata,
)

__all__ = [
    "CodeChunkMetadata",
    "QueryRequest",
    "QueryResponse",
    "RetrievalContext",
    "RetrievalResult",
    "RetrievalResultSet",
    "SearchScope",
    "Snippet",
    "Synapse",
    "SynapseConfig",
    "TextChunkMetadata",
]
