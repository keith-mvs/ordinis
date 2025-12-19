"""RAG configuration module."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Get project root (4 levels up from this file: src/ordinis/rag/config.py -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class RAGConfig(BaseModel):
    """Configuration for RAG system."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Vector database
    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chromadb",
        description="Directory for ChromaDB persistence",
    )
    text_collection_name: str = Field(
        default="kb_text",
        description="Collection name for text embeddings",
    )
    code_collection_name: str = Field(
        default="codebase",
        description="Collection name for code embeddings",
    )

    # Embedding models
    text_embedding_model: str = Field(
        default="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        description="Text embedding model (lightweight, works on CPU/GPU)",
    )
    code_embedding_model: str = Field(
        default="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        description="Code embedding model (uses same model as text for simplicity)",
    )
    rerank_model: str = Field(
        default="nvidia/llama-3.2-nemoretriever-500m-rerank-v2",
        description="NVIDIA reranking model",
    )

    # Embedding strategy
    use_local_embeddings: bool = Field(
        default=False,
        description="Use local GPU for embeddings (vs API)",
    )
    text_embedding_dimension: int = Field(
        default=1024,
        description="Matryoshka embedding dimension for text (768 → 384)",
    )

    # Retrieval parameters
    top_k_retrieval: int = Field(
        default=20,
        description="Number of candidates to retrieve from vector DB",
    )
    top_k_rerank: int = Field(
        default=5,
        description="Number of results after reranking",
    )
    similarity_threshold: float = Field(
        default=0.15,  # Lower threshold for minimal test dataset
        description="Minimum cosine similarity for retrieval",
    )

    # Chunking parameters
    text_chunk_size: int = Field(
        default=512,
        description="Tokens per text chunk",
    )
    text_chunk_overlap: int = Field(
        default=50,
        description="Overlap between text chunks",
    )
    code_chunk_size: int = Field(
        default=1024,
        description="Tokens per code chunk (larger for full functions)",
    )

    # Knowledge base paths
    kb_base_path: Path = Field(
        default=PROJECT_ROOT / "docs" / "knowledge-base",
        description="Base directory for knowledge base",
    )
    code_base_path: Path = Field(
        default=PROJECT_ROOT / "src",
        description="Base directory for code indexing",
    )
    project_root: Path = Field(
        default=PROJECT_ROOT,
        description="Project root directory (used for relative path calculations)",
    )

    # API fallback
    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key for hosted endpoints",
    )
    use_api_fallback: bool = Field(
        default=True,
        description="Fall back to API if local embedding fails",
    )

    # VRAM management
    max_vram_usage_gb: float = Field(
        default=9.0,
        description="Max VRAM usage before API fallback",
    )
    check_vram_before_load: bool = Field(
        default=True,
        description="Check GPU memory before loading models",
    )

    # Caching
    enable_query_cache: bool = Field(
        default=True,
        description="Enable caching for common queries",
    )
    cache_directory: Path = Field(
        default=Path("data/cache/rag"),
        description="Directory for query cache",
    )
    cache_max_size_mb: int = Field(
        default=100,
        description="Maximum cache size in MB",
    )


# Global config instance
_config: RAGConfig | None = None


def get_config() -> RAGConfig:
    """Get global RAG configuration."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set global RAG configuration."""
    global _config
    _config = config
