"""
Deterministic ID generation for ChromaDB vectors.

Implements R5 from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Collision-resistant IDs using content hashing
- Stable IDs linked to source records
- Enables idempotent upserts

ID Format: {entity_type}:{source_id}:{content_hash}:{chunk_index}
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def generate_vector_id(
    entity_type: str,
    source_id: str,
    content: str,
    chunk_index: int = 0,
) -> str:
    """Generate deterministic, collision-resistant vector ID.
    
    Format: {entity_type}:{source_id}:{content_hash}:{chunk_index}
    
    Args:
        entity_type: Type of entity (trade, session, kb, code, message, summary)
        source_id: Source identifier (trade_id, session_id, file_path)
        content: Content being indexed (for change detection)
        chunk_index: Chunk index within document
    
    Returns:
        Deterministic ID string
        
    Examples:
        >>> generate_vector_id("trade", "t_123", "AAPL long entry at $150")
        'trade:t_123:a1b2c3d4e5f6:0'
        
        >>> generate_vector_id("session", "sess_abc", "chunk content", chunk_index=5)
        'session:sess_abc:f6e5d4c3b2a1:5'
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{entity_type}:{source_id}:{content_hash}:{chunk_index}"


def generate_content_hash(content: str) -> str:
    """Generate a content hash for change detection.
    
    Args:
        content: Content to hash
        
    Returns:
        12-character hex hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def generate_trade_vector_id(trade_id: str, content: str) -> str:
    """Generate vector ID for a trade document.
    
    Args:
        trade_id: Trade identifier from SQLite
        content: Trade document content
        
    Returns:
        Deterministic vector ID
    """
    return generate_vector_id("trade", trade_id, content, chunk_index=0)


def generate_session_chunk_id(session_id: str, content: str, chunk_index: int) -> str:
    """Generate vector ID for a session log chunk.
    
    Args:
        session_id: Session identifier
        content: Chunk content
        chunk_index: Position within session
        
    Returns:
        Deterministic vector ID
    """
    return generate_vector_id("session", session_id, content, chunk_index)


def generate_message_vector_id(session_id: str, sequence: int, content: str) -> str:
    """Generate vector ID for a message.
    
    Args:
        session_id: Session identifier
        sequence: Message sequence number
        content: Message content
        
    Returns:
        Deterministic vector ID
    """
    source_id = f"{session_id}:{sequence}"
    return generate_vector_id("message", source_id, content, chunk_index=0)


def generate_summary_vector_id(
    session_id: str,
    summary_type: str,
    start_seq: int,
    end_seq: int,
    content: str,
) -> str:
    """Generate vector ID for a session summary.
    
    Args:
        session_id: Session identifier
        summary_type: Type of summary (rolling, final, key_facts)
        start_seq: Start sequence of summarized messages
        end_seq: End sequence of summarized messages
        content: Summary content
        
    Returns:
        Deterministic vector ID
    """
    source_id = f"{session_id}:{summary_type}:{start_seq}-{end_seq}"
    return generate_vector_id("summary", source_id, content, chunk_index=0)


def generate_kb_chunk_id(file_path: str, content: str, chunk_index: int) -> str:
    """Generate vector ID for a knowledge base chunk.
    
    Args:
        file_path: Source file path
        content: Chunk content
        chunk_index: Position within file
        
    Returns:
        Deterministic vector ID
    """
    # Normalize path for cross-platform consistency
    normalized_path = file_path.replace("\\", "/").strip("/")
    return generate_vector_id("kb", normalized_path, content, chunk_index)


def generate_code_chunk_id(file_path: str, content: str, chunk_index: int) -> str:
    """Generate vector ID for a code chunk.
    
    Args:
        file_path: Source file path
        content: Code content
        chunk_index: Position within file
        
    Returns:
        Deterministic vector ID
    """
    # Normalize path for cross-platform consistency
    normalized_path = file_path.replace("\\", "/").strip("/")
    return generate_vector_id("code", normalized_path, content, chunk_index)


def parse_vector_id(vector_id: str) -> dict[str, Any]:
    """Parse a vector ID into its components.
    
    Args:
        vector_id: ID string to parse
        
    Returns:
        Dictionary with entity_type, source_id, content_hash, chunk_index
        
    Raises:
        ValueError: If ID format is invalid
    """
    parts = vector_id.split(":")
    if len(parts) < 4:
        raise ValueError(f"Invalid vector ID format: {vector_id}")
    
    # Handle source_id that may contain colons
    entity_type = parts[0]
    chunk_index = int(parts[-1])
    content_hash = parts[-2]
    source_id = ":".join(parts[1:-2])
    
    return {
        "entity_type": entity_type,
        "source_id": source_id,
        "content_hash": content_hash,
        "chunk_index": chunk_index,
    }


def is_valid_vector_id(vector_id: str) -> bool:
    """Check if a vector ID has valid format.
    
    Args:
        vector_id: ID to validate
        
    Returns:
        True if valid format
    """
    try:
        parsed = parse_vector_id(vector_id)
        return (
            parsed["entity_type"] in ("trade", "session", "message", "summary", "kb", "code")
            and len(parsed["content_hash"]) == 12
            and parsed["chunk_index"] >= 0
        )
    except (ValueError, KeyError):
        return False


class VectorIdGenerator:
    """Factory for generating consistent vector IDs with metadata tracking.
    
    Usage:
        generator = VectorIdGenerator(embedding_model="nvidia/llama-3.2-nemoretriever-300m-embed-v2")
        id, metadata = generator.create_trade_id(trade_id, content)
    """
    
    def __init__(
        self,
        embedding_model: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        embedding_dim: int = 1024,
    ):
        """Initialize generator with embedding model info.
        
        Args:
            embedding_model: Model identifier for versioning
            embedding_dim: Embedding dimension for validation
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
    
    def create_trade_id(
        self,
        trade_id: str,
        content: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create vector ID and metadata for a trade.
        
        Args:
            trade_id: SQLite trade_id
            content: Trade document content
            extra_metadata: Additional metadata to include
            
        Returns:
            Tuple of (vector_id, full_metadata)
        """
        vector_id = generate_trade_vector_id(trade_id, content)
        content_hash = generate_content_hash(content)
        
        metadata = {
            "entity_type": "trade",
            "source_id": trade_id,
            "source_table": "trades",
            "content_hash": content_hash,
            "indexed_at": _utcnow().isoformat(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            **(extra_metadata or {}),
        }
        
        return vector_id, metadata
    
    def create_session_chunk_id(
        self,
        session_id: str,
        content: str,
        chunk_index: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create vector ID and metadata for a session chunk.
        
        Args:
            session_id: Session identifier
            content: Chunk content
            chunk_index: Position within session
            extra_metadata: Additional metadata
            
        Returns:
            Tuple of (vector_id, full_metadata)
        """
        vector_id = generate_session_chunk_id(session_id, content, chunk_index)
        content_hash = generate_content_hash(content)
        
        metadata = {
            "entity_type": "session",
            "session_id": session_id,
            "source_table": "messages",
            "chunk_index": chunk_index,
            "content_hash": content_hash,
            "indexed_at": _utcnow().isoformat(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            **(extra_metadata or {}),
        }
        
        return vector_id, metadata
    
    def create_summary_id(
        self,
        session_id: str,
        summary_type: str,
        start_seq: int,
        end_seq: int,
        content: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create vector ID and metadata for a summary.
        
        Args:
            session_id: Session identifier
            summary_type: Type of summary
            start_seq: Start message sequence
            end_seq: End message sequence
            content: Summary content
            extra_metadata: Additional metadata
            
        Returns:
            Tuple of (vector_id, full_metadata)
        """
        vector_id = generate_summary_vector_id(
            session_id, summary_type, start_seq, end_seq, content
        )
        content_hash = generate_content_hash(content)
        
        metadata = {
            "entity_type": "summary",
            "session_id": session_id,
            "summary_type": summary_type,
            "source_table": "session_summaries",
            "start_sequence": start_seq,
            "end_sequence": end_seq,
            "content_hash": content_hash,
            "indexed_at": _utcnow().isoformat(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            **(extra_metadata or {}),
        }
        
        return vector_id, metadata


# Singleton instance with default config
_default_generator: VectorIdGenerator | None = None


def get_id_generator(
    embedding_model: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    embedding_dim: int = 1024,
) -> VectorIdGenerator:
    """Get or create the default ID generator.
    
    Args:
        embedding_model: Model identifier
        embedding_dim: Embedding dimension
        
    Returns:
        VectorIdGenerator instance
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = VectorIdGenerator(embedding_model, embedding_dim)
    return _default_generator
