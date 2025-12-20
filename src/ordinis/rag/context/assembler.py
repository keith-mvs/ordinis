"""Context assembly service for LLM prompt construction.

This module provides the ContextAssembler class that:
1. Assembles relevant context from multiple sources for LLM prompts
2. Combines session history, summaries, retrieved documents, and trade data
3. Manages token budgets and context window limits
4. Supports dynamic context prioritization

Memory integration from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Multi-source context assembly
- Token budget management
- Priority-based context selection
- Caching for repeated queries

Example:
    assembler = ContextAssembler(chroma_client, db_path)
    context = await assembler.assemble_context(
        query="What was my strategy for AAPL trades?",
        session_id="current_session",
        max_tokens=4000,
    )
    prompt = f"{context}\\n\\nUser: {user_message}"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.rag.vectordb.chroma_client import ChromaDBClient
    from ordinis.rag.embedders.text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class ContextSource(Enum):
    """Sources of context information."""
    
    RECENT_MESSAGES = "recent_messages"  # Recent conversation history
    SESSION_SUMMARY = "session_summary"  # Summarized session history
    RETRIEVED_DOCS = "retrieved_docs"  # RAG-retrieved documents
    TRADE_HISTORY = "trade_history"  # Related trades
    KNOWLEDGE_BASE = "knowledge_base"  # KB articles
    CODE_CONTEXT = "code_context"  # Relevant code snippets
    SYSTEM_STATE = "system_state"  # Current system status


class ContextPriority(Enum):
    """Priority levels for context sources."""
    
    CRITICAL = 1  # Must include (e.g., system warnings)
    HIGH = 2  # Important (e.g., recent messages)
    MEDIUM = 3  # Relevant (e.g., retrieved docs)
    LOW = 4  # Nice to have (e.g., older history)


@dataclass
class ContextChunk:
    """A chunk of context from a specific source."""
    
    source: ContextSource
    priority: ContextPriority
    content: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0  # Relevance score (0-1)
    
    @property
    def effective_priority(self) -> float:
        """Calculate effective priority considering score."""
        return self.priority.value + (1 - self.score)


@dataclass
class AssembledContext:
    """The assembled context ready for LLM consumption."""
    
    content: str
    total_tokens: int
    sources_used: list[ContextSource]
    chunks_included: int
    chunks_dropped: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "total_tokens": self.total_tokens,
            "sources_used": [s.value for s in self.sources_used],
            "chunks_included": self.chunks_included,
            "chunks_dropped": self.chunks_dropped,
            "metadata": self.metadata,
        }


class ContextAssembler:
    """Assembles context from multiple sources for LLM prompts.
    
    This class implements a priority-based context assembly strategy:
    1. Gather context chunks from all configured sources
    2. Score and prioritize chunks based on relevance
    3. Pack chunks into token budget, prioritizing high-relevance
    4. Format the final context string
    
    Attributes:
        chroma_client: ChromaDB client for semantic search
        db_path: Path to SQLite database
        text_embedder: Embedder for query encoding
        default_max_tokens: Default token budget
    """
    
    # Default source priorities
    DEFAULT_PRIORITIES = {
        ContextSource.SYSTEM_STATE: ContextPriority.CRITICAL,
        ContextSource.RECENT_MESSAGES: ContextPriority.HIGH,
        ContextSource.TRADE_HISTORY: ContextPriority.HIGH,
        ContextSource.SESSION_SUMMARY: ContextPriority.MEDIUM,
        ContextSource.RETRIEVED_DOCS: ContextPriority.MEDIUM,
        ContextSource.KNOWLEDGE_BASE: ContextPriority.MEDIUM,
        ContextSource.CODE_CONTEXT: ContextPriority.LOW,
    }
    
    # Default token allocations (percentage of budget)
    DEFAULT_ALLOCATIONS = {
        ContextSource.SYSTEM_STATE: 0.05,
        ContextSource.RECENT_MESSAGES: 0.30,
        ContextSource.SESSION_SUMMARY: 0.15,
        ContextSource.RETRIEVED_DOCS: 0.25,
        ContextSource.TRADE_HISTORY: 0.15,
        ContextSource.KNOWLEDGE_BASE: 0.10,
        ContextSource.CODE_CONTEXT: 0.10,
    }
    
    def __init__(
        self,
        chroma_client: "ChromaDBClient | None" = None,
        db_path: Path | str | None = None,
        text_embedder: "TextEmbedder | None" = None,
        default_max_tokens: int = 8000,
        chars_per_token: int = 4,
    ):
        """Initialize the context assembler.
        
        Args:
            chroma_client: ChromaDB client
            db_path: Path to SQLite database
            text_embedder: Text embedder for queries
            default_max_tokens: Default token budget
            chars_per_token: Approximate characters per token
        """
        self.db_path = Path(db_path) if db_path else self._get_default_db_path()
        self.default_max_tokens = default_max_tokens
        self.chars_per_token = chars_per_token
        
        # Lazy initialization
        self._chroma_client = chroma_client
        self._text_embedder = text_embedder
        
        # Simple LRU cache for recent queries
        self._cache: dict[str, tuple[AssembledContext, datetime]] = {}
        self._cache_ttl_seconds = 60
        
        logger.info(f"ContextAssembler initialized with max_tokens={default_max_tokens}")
    
    @staticmethod
    def _get_default_db_path() -> Path:
        """Get default database path."""
        import os
        return Path(os.environ.get("ORDINIS_DB_PATH", "data/ordinis.db"))
    
    @property
    def chroma_client(self) -> "ChromaDBClient":
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            from ordinis.rag.vectordb.chroma_client import ChromaDBClient
            self._chroma_client = ChromaDBClient()
        return self._chroma_client
    
    @property
    def text_embedder(self) -> "TextEmbedder":
        """Get or create text embedder."""
        if self._text_embedder is None:
            from ordinis.rag.embedders.text_embedder import TextEmbedder
            self._text_embedder = TextEmbedder()
        return self._text_embedder
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // self.chars_per_token
    
    async def assemble_context(
        self,
        query: str,
        session_id: str | None = None,
        max_tokens: int | None = None,
        sources: list[ContextSource] | None = None,
        include_system_state: bool = True,
        include_recent_messages: int = 10,
        include_trade_history: int = 5,
        retrieve_n_docs: int = 3,
    ) -> AssembledContext:
        """Assemble context for an LLM prompt.
        
        Args:
            query: User query to contextualize
            session_id: Current session ID
            max_tokens: Token budget (uses default if None)
            sources: Which sources to include (all if None)
            include_system_state: Include current system state
            include_recent_messages: Number of recent messages
            include_trade_history: Number of related trades
            retrieve_n_docs: Number of docs to retrieve per source
            
        Returns:
            AssembledContext ready for LLM consumption
        """
        max_tokens = max_tokens or self.default_max_tokens
        sources = sources or list(ContextSource)
        
        # Check cache
        cache_key = f"{query}:{session_id}:{max_tokens}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if (_utcnow() - timestamp).total_seconds() < self._cache_ttl_seconds:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached
        
        # Gather chunks from all sources
        chunks: list[ContextChunk] = []
        
        # Parallelize gathering where possible
        gather_tasks = []
        
        if ContextSource.SYSTEM_STATE in sources and include_system_state:
            gather_tasks.append(self._gather_system_state())
        
        if ContextSource.RECENT_MESSAGES in sources and session_id:
            gather_tasks.append(
                self._gather_recent_messages(session_id, include_recent_messages)
            )
        
        if ContextSource.SESSION_SUMMARY in sources and session_id:
            gather_tasks.append(self._gather_session_summary(session_id))
        
        if ContextSource.RETRIEVED_DOCS in sources:
            gather_tasks.append(
                self._gather_retrieved_docs(query, retrieve_n_docs)
            )
        
        if ContextSource.TRADE_HISTORY in sources:
            gather_tasks.append(
                self._gather_trade_history(query, include_trade_history)
            )
        
        if ContextSource.KNOWLEDGE_BASE in sources:
            gather_tasks.append(
                self._gather_knowledge_base(query, retrieve_n_docs)
            )
        
        # Execute all gather tasks
        if gather_tasks:
            results = await asyncio.gather(*gather_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Context gather failed: {result}")
                elif result:
                    chunks.extend(result)
        
        # Pack chunks into token budget
        assembled = self._pack_chunks(chunks, max_tokens)
        
        # Cache result
        self._cache[cache_key] = (assembled, _utcnow())
        
        logger.info(
            f"Assembled context: {assembled.total_tokens} tokens, "
            f"{assembled.chunks_included} chunks from {len(assembled.sources_used)} sources"
        )
        
        return assembled
    
    def _pack_chunks(
        self,
        chunks: list[ContextChunk],
        max_tokens: int,
    ) -> AssembledContext:
        """Pack chunks into token budget using priority ordering."""
        # Sort by effective priority (lower is better)
        sorted_chunks = sorted(chunks, key=lambda c: c.effective_priority)
        
        included_chunks: list[ContextChunk] = []
        total_tokens = 0
        sources_used: set[ContextSource] = set()
        
        for chunk in sorted_chunks:
            # Check if we have room
            if total_tokens + chunk.token_count <= max_tokens:
                included_chunks.append(chunk)
                total_tokens += chunk.token_count
                sources_used.add(chunk.source)
            elif chunk.priority == ContextPriority.CRITICAL:
                # Critical chunks must be included - truncate if needed
                available_tokens = max_tokens - total_tokens
                if available_tokens > 100:  # Minimum useful content
                    truncated_content = chunk.content[:available_tokens * self.chars_per_token]
                    chunk.content = truncated_content + "..."
                    chunk.token_count = available_tokens
                    included_chunks.append(chunk)
                    total_tokens += chunk.token_count
                    sources_used.add(chunk.source)
        
        # Format final context
        content_parts = []
        
        for source in ContextSource:
            source_chunks = [c for c in included_chunks if c.source == source]
            if source_chunks:
                content_parts.append(f"\n### {source.value.replace('_', ' ').title()}\n")
                for chunk in source_chunks:
                    content_parts.append(chunk.content)
                    content_parts.append("")
        
        content = "\n".join(content_parts).strip()
        
        return AssembledContext(
            content=content,
            total_tokens=total_tokens,
            sources_used=list(sources_used),
            chunks_included=len(included_chunks),
            chunks_dropped=len(chunks) - len(included_chunks),
        )
    
    async def _gather_system_state(self) -> list[ContextChunk]:
        """Gather current system state."""
        import aiosqlite
        
        chunks = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(
                    "SELECT key, value, updated_at FROM system_state"
                )
                rows = await cursor.fetchall()
                
                if rows:
                    state_lines = [f"- {row['key']}: {row['value']}" for row in rows]
                    content = "Current System State:\n" + "\n".join(state_lines)
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.SYSTEM_STATE,
                        priority=ContextPriority.CRITICAL,
                        content=content,
                        token_count=self._estimate_tokens(content),
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to gather system state: {e}")
        
        return chunks
    
    async def _gather_recent_messages(
        self,
        session_id: str,
        limit: int,
    ) -> list[ContextChunk]:
        """Gather recent conversation messages."""
        import aiosqlite
        
        chunks = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(
                    """
                    SELECT role, content, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
                rows = await cursor.fetchall()
                
                if rows:
                    # Reverse to chronological order
                    messages = list(reversed(rows))
                    formatted = []
                    for msg in messages:
                        role = msg["role"].upper()
                        content = msg["content"]
                        formatted.append(f"[{role}]: {content}")
                    
                    content = "Recent Conversation:\n" + "\n\n".join(formatted)
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.RECENT_MESSAGES,
                        priority=ContextPriority.HIGH,
                        content=content,
                        token_count=self._estimate_tokens(content),
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to gather recent messages: {e}")
        
        return chunks
    
    async def _gather_session_summary(self, session_id: str) -> list[ContextChunk]:
        """Gather session summary if available."""
        import aiosqlite
        
        chunks = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Get session summary (prefer session-level over chunk-level)
                cursor = await db.execute(
                    """
                    SELECT content, summary_type, message_end_idx
                    FROM session_summaries
                    WHERE session_id = ?
                    ORDER BY 
                        CASE summary_type WHEN 'session' THEN 0 ELSE 1 END,
                        message_end_idx DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = await cursor.fetchone()
                
                if row:
                    content = f"Session Summary:\n{row['content']}"
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.SESSION_SUMMARY,
                        priority=ContextPriority.MEDIUM,
                        content=content,
                        token_count=self._estimate_tokens(content),
                        metadata={"summary_type": row["summary_type"]},
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to gather session summary: {e}")
        
        return chunks
    
    async def _gather_retrieved_docs(
        self,
        query: str,
        n_results: int,
    ) -> list[ContextChunk]:
        """Retrieve relevant documents via semantic search."""
        chunks = []
        
        try:
            # Generate query embedding
            query_embedding = self.text_embedder.embed_texts([query])[0]
            
            # Search text collection
            collection = self.chroma_client.get_text_collection()
            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = 1 - min(distance, 1)  # Convert distance to similarity
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    content = f"[Retrieved] {doc}"
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.RETRIEVED_DOCS,
                        priority=ContextPriority.MEDIUM,
                        content=content,
                        token_count=self._estimate_tokens(content),
                        metadata=metadata,
                        score=score,
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to retrieve docs: {e}")
        
        return chunks
    
    async def _gather_trade_history(
        self,
        query: str,
        n_results: int,
    ) -> list[ContextChunk]:
        """Gather relevant trade history."""
        chunks = []
        
        try:
            # First try semantic search for related trades
            query_embedding = self.text_embedder.embed_texts([query])[0]
            
            collection = self.chroma_client.get_text_collection()
            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=n_results,
                where={"entity_type": "trade"},
                include=["documents", "metadatas", "distances"],
            )
            
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = 1 - min(distance, 1)
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    content = f"[Trade] {doc}"
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.TRADE_HISTORY,
                        priority=ContextPriority.HIGH,
                        content=content,
                        token_count=self._estimate_tokens(content),
                        metadata=metadata,
                        score=score,
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to gather trade history: {e}")
        
        return chunks
    
    async def _gather_knowledge_base(
        self,
        query: str,
        n_results: int,
    ) -> list[ContextChunk]:
        """Gather relevant knowledge base articles."""
        chunks = []
        
        try:
            query_embedding = self.text_embedder.embed_texts([query])[0]
            
            collection = self.chroma_client.get_text_collection()
            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=n_results,
                where={"entity_type": {"$in": ["kb", "knowledge_base", "documentation"]}},
                include=["documents", "metadatas", "distances"],
            )
            
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = 1 - min(distance, 1)
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    title = metadata.get("title", metadata.get("file_path", "Document"))
                    content = f"[KB: {title}]\n{doc}"
                    
                    chunks.append(ContextChunk(
                        source=ContextSource.KNOWLEDGE_BASE,
                        priority=ContextPriority.MEDIUM,
                        content=content,
                        token_count=self._estimate_tokens(content),
                        metadata=metadata,
                        score=score,
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to gather KB: {e}")
        
        return chunks
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._cache.clear()
        logger.debug("Context cache cleared")
    
    async def preload_session_context(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> AssembledContext:
        """Preload context for a session without a specific query.
        
        Useful for session initialization.
        
        Args:
            session_id: Session ID to preload
            max_tokens: Token budget
            
        Returns:
            AssembledContext with session context
        """
        return await self.assemble_context(
            query="",  # No specific query
            session_id=session_id,
            max_tokens=max_tokens,
            sources=[
                ContextSource.SYSTEM_STATE,
                ContextSource.RECENT_MESSAGES,
                ContextSource.SESSION_SUMMARY,
            ],
            retrieve_n_docs=0,  # Skip retrieval without query
        )
