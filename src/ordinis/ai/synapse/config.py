"""
Synapse configuration.

Defines configuration for RAG retrieval settings.
"""

from dataclasses import dataclass

from ordinis.ai.synapse.models import SearchScope
from ordinis.engines.base import BaseEngineConfig


@dataclass
class SynapseConfig(BaseEngineConfig):
    """Configuration for Synapse retrieval engine."""

    # Search defaults
    default_scope: SearchScope = SearchScope.AUTO
    default_top_k: int = 5
    similarity_threshold: float = 0.5

    # Retrieval settings
    top_k_retrieval: int = 20  # Candidates to retrieve before reranking
    top_k_rerank: int = 10  # Results to return after reranking

    # Embedding settings
    use_helix_embeddings: bool = False  # Use Helix instead of direct embeddings
    embedding_model: str = "nv-embedqa"  # Helix model alias for embeddings

    # ChromaDB settings
    chroma_persist_dir: str = "./data/chroma"
    text_collection: str = "ordinis_kb"
    code_collection: str = "ordinis_code"

    # Context formatting
    max_context_tokens: int = 2000
    include_citations: bool = True

    # Performance
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Orchestrator settings
    enable_intent_recognition: bool = True
    max_search_retries: int = 3
    conversation_history_limit: int = 10

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors = super().validate()

        if self.default_top_k < 1:
            errors.append("default_top_k must be >= 1")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")

        if self.top_k_retrieval < self.top_k_rerank:
            errors.append("top_k_retrieval must be >= top_k_rerank")

        return errors
