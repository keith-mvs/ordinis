"""Session summarization service for conversation memory management.

This module provides the SessionSummarizer class that:
1. Generates rolling summaries of conversation history
2. Stores summaries in SQLite (session_summaries table)
3. Generates embeddings and syncs to ChromaDB
4. Supports hierarchical summarization (message → chunk → session)

Memory integration from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Rolling summarizer for long conversations
- Hierarchical compression (messages → summaries)
- Session-level summary generation
- Vector storage for semantic retrieval

Example:
    summarizer = SessionSummarizer(db_path, chroma_client, llm_client)
    await summarizer.create_chunk_summary(session_id, chunk_index=1)
    await summarizer.create_session_summary(session_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    import numpy as np
    from ordinis.rag.vectordb.chroma_client import ChromaDBClient
    from ordinis.rag.embedders.text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class SummaryType(Enum):
    """Types of summaries."""
    
    CHUNK = "chunk"  # Summary of a message chunk (e.g., 10-20 messages)
    SESSION = "session"  # Summary of entire session
    TOPIC = "topic"  # Summary of messages on specific topic
    DAILY = "daily"  # Daily rollup summary


class SummaryStatus(Enum):
    """Status of summary generation."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Summary:
    """A generated summary."""
    
    id: str
    session_id: str
    summary_type: SummaryType
    content: str
    message_start_idx: int
    message_end_idx: int
    token_count: int
    created_at: datetime
    status: SummaryStatus = SummaryStatus.COMPLETED
    chroma_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "summary_type": self.summary_type.value,
            "content": self.content,
            "message_start_idx": self.message_start_idx,
            "message_end_idx": self.message_end_idx,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "chroma_id": self.chroma_id,
            "metadata": self.metadata,
        }


class SummaryPrompts:
    """Prompt templates for summarization."""
    
    CHUNK_SUMMARY = """Summarize the following conversation segment concisely.
Focus on:
- Key decisions made
- Important information exchanged
- Action items or outcomes
- Any trading-related context

Conversation:
{messages}

Summary:"""

    SESSION_SUMMARY = """Create a comprehensive summary of this trading session.
Include:
- Main topics discussed
- Key decisions and their rationale
- Trading actions taken (if any)
- Outcomes and lessons learned
- Important context for future sessions

Session chunks:
{chunks}

Session summary:"""

    TOPIC_SUMMARY = """Summarize the discussion about: {topic}

Messages:
{messages}

Topic summary:"""


class SessionSummarizer:
    """Generates and manages session summaries.
    
    Implements a hierarchical summarization approach:
    1. Messages are grouped into chunks (e.g., 20 messages each)
    2. Chunks are summarized individually
    3. Chunk summaries are combined into session summary
    
    This provides:
    - Bounded memory for long conversations
    - Fast retrieval of relevant context
    - Semantic search over conversation history
    
    Attributes:
        db_path: Path to SQLite database
        chroma_client: ChromaDB client for vector storage
        llm_client: LLM client for summarization
        chunk_size: Number of messages per chunk
    """
    
    def __init__(
        self,
        db_path: Path | str | None = None,
        chroma_client: "ChromaDBClient | None" = None,
        llm_client: Any | None = None,
        text_embedder: "TextEmbedder | None" = None,
        chunk_size: int = 20,
        max_tokens_per_summary: int = 500,
    ):
        """Initialize the session summarizer.
        
        Args:
            db_path: Path to SQLite database
            chroma_client: ChromaDB client
            llm_client: LLM client with generate() method
            text_embedder: Text embedder for vector generation
            chunk_size: Messages per chunk for summarization
            max_tokens_per_summary: Maximum tokens in generated summary
        """
        self.db_path = Path(db_path) if db_path else self._get_default_db_path()
        self.chunk_size = chunk_size
        self.max_tokens_per_summary = max_tokens_per_summary
        
        # Lazy initialization
        self._chroma_client = chroma_client
        self._llm_client = llm_client
        self._text_embedder = text_embedder
        
        # Token counting (approximate)
        self._avg_chars_per_token = 4
        
        logger.info(f"SessionSummarizer initialized with chunk_size={chunk_size}")
    
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
        """Estimate token count for text."""
        return len(text) // self._avg_chars_per_token
    
    async def create_chunk_summary(
        self,
        session_id: str,
        chunk_index: int | None = None,
        message_start_idx: int | None = None,
        message_end_idx: int | None = None,
    ) -> Summary | None:
        """Create a summary for a message chunk.
        
        Args:
            session_id: Session ID
            chunk_index: Which chunk to summarize (0-indexed)
            message_start_idx: Explicit start index (overrides chunk_index)
            message_end_idx: Explicit end index (overrides chunk_index)
            
        Returns:
            Generated Summary or None if no messages found
        """
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Determine message range
            if message_start_idx is not None and message_end_idx is not None:
                start_idx = message_start_idx
                end_idx = message_end_idx
            elif chunk_index is not None:
                start_idx = chunk_index * self.chunk_size
                end_idx = start_idx + self.chunk_size
            else:
                # Find next unprocessed chunk
                start_idx, end_idx = await self._find_next_chunk(db, session_id)
                if start_idx is None:
                    logger.info(f"No unprocessed chunks for session {session_id}")
                    return None
            
            # Fetch messages
            cursor = await db.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, end_idx - start_idx, start_idx),
            )
            rows = await cursor.fetchall()
            
            if not rows:
                logger.info(f"No messages found for session {session_id} range [{start_idx}, {end_idx})")
                return None
            
            # Format messages for summarization
            messages_text = self._format_messages_for_summary([dict(r) for r in rows])
            
            # Generate summary using LLM
            summary_content = await self._generate_summary(
                prompt_template=SummaryPrompts.CHUNK_SUMMARY,
                messages=messages_text,
            )
            
            # Create summary object
            summary = Summary(
                id=str(uuid4()),
                session_id=session_id,
                summary_type=SummaryType.CHUNK,
                content=summary_content,
                message_start_idx=start_idx,
                message_end_idx=start_idx + len(rows),
                token_count=self._estimate_tokens(summary_content),
                created_at=_utcnow(),
            )
            
            # Store in SQLite
            await self._store_summary(db, summary)
            
            # Generate embedding and store in ChromaDB
            await self._vectorize_summary(summary)
            
            await db.commit()
            
            logger.info(
                f"Created chunk summary for session {session_id} "
                f"[{summary.message_start_idx}, {summary.message_end_idx})"
            )
            
            return summary
    
    async def create_session_summary(self, session_id: str) -> Summary | None:
        """Create a comprehensive session summary.
        
        Combines all chunk summaries into a single session summary.
        
        Args:
            session_id: Session ID
            
        Returns:
            Generated Summary or None if no data
        """
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Fetch all chunk summaries for this session
            cursor = await db.execute(
                """
                SELECT content, message_start_idx, message_end_idx
                FROM session_summaries
                WHERE session_id = ? AND summary_type = ?
                ORDER BY message_start_idx ASC
                """,
                (session_id, SummaryType.CHUNK.value),
            )
            chunks = await cursor.fetchall()
            
            if not chunks:
                # No chunk summaries - create from raw messages
                logger.info(f"No chunk summaries for session {session_id}, creating from messages")
                return await self._create_session_summary_from_messages(db, session_id)
            
            # Combine chunk summaries
            chunks_text = "\n\n---\n\n".join(
                f"[Messages {r['message_start_idx']}-{r['message_end_idx']}]:\n{r['content']}"
                for r in chunks
            )
            
            # Get message count
            cursor = await db.execute(
                "SELECT MIN(message_start_idx), MAX(message_end_idx) FROM session_summaries WHERE session_id = ?",
                (session_id,),
            )
            range_row = await cursor.fetchone()
            start_idx = range_row[0] if range_row else 0
            end_idx = range_row[1] if range_row else 0
            
            # Generate session summary
            summary_content = await self._generate_summary(
                prompt_template=SummaryPrompts.SESSION_SUMMARY,
                chunks=chunks_text,
            )
            
            summary = Summary(
                id=str(uuid4()),
                session_id=session_id,
                summary_type=SummaryType.SESSION,
                content=summary_content,
                message_start_idx=start_idx,
                message_end_idx=end_idx,
                token_count=self._estimate_tokens(summary_content),
                created_at=_utcnow(),
            )
            
            await self._store_summary(db, summary)
            await self._vectorize_summary(summary)
            await db.commit()
            
            logger.info(f"Created session summary for {session_id}")
            
            return summary
    
    async def _create_session_summary_from_messages(
        self,
        db: Any,
        session_id: str,
    ) -> Summary | None:
        """Create session summary directly from messages (for short sessions)."""
        cursor = await db.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT 100
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        
        if not rows:
            return None
        
        messages_text = self._format_messages_for_summary([dict(r) for r in rows])
        
        summary_content = await self._generate_summary(
            prompt_template=SummaryPrompts.CHUNK_SUMMARY,
            messages=messages_text,
        )
        
        summary = Summary(
            id=str(uuid4()),
            session_id=session_id,
            summary_type=SummaryType.SESSION,
            content=summary_content,
            message_start_idx=0,
            message_end_idx=len(rows),
            token_count=self._estimate_tokens(summary_content),
            created_at=_utcnow(),
        )
        
        await self._store_summary(db, summary)
        await self._vectorize_summary(summary)
        
        return summary
    
    async def _find_next_chunk(
        self,
        db: Any,
        session_id: str,
    ) -> tuple[int | None, int | None]:
        """Find the next message range that needs summarization."""
        # Get highest summarized index
        cursor = await db.execute(
            """
            SELECT MAX(message_end_idx) as max_idx
            FROM session_summaries
            WHERE session_id = ? AND summary_type = ?
            """,
            (session_id, SummaryType.CHUNK.value),
        )
        row = await cursor.fetchone()
        last_summarized = row[0] if row and row[0] else 0
        
        # Get total message count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        total_messages = row[0] if row else 0
        
        # Check if there are enough unsummarized messages
        unsummarized = total_messages - last_summarized
        if unsummarized < self.chunk_size:
            return None, None
        
        return last_summarized, last_summarized + self.chunk_size
    
    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for LLM summarization."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            timestamp = msg.get("created_at", "")
            formatted.append(f"[{role}] ({timestamp}): {content}")
        return "\n\n".join(formatted)
    
    async def _generate_summary(
        self,
        prompt_template: str,
        **kwargs: Any,
    ) -> str:
        """Generate summary using LLM or fallback."""
        prompt = prompt_template.format(**kwargs)
        
        if self._llm_client is not None:
            try:
                response = await self._call_llm(prompt)
                return response
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, using fallback")
        
        # Fallback: simple extractive summary
        return self._fallback_summarize(kwargs.get("messages", "") or kwargs.get("chunks", ""))
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for summarization."""
        # Support different LLM client interfaces
        if hasattr(self._llm_client, "generate"):
            response = await self._llm_client.generate(prompt)
            if isinstance(response, str):
                return response
            return response.get("content", response.get("text", str(response)))
        elif hasattr(self._llm_client, "chat"):
            response = await self._llm_client.chat([{"role": "user", "content": prompt}])
            return response.get("content", str(response))
        else:
            raise ValueError("LLM client must have generate() or chat() method")
    
    def _fallback_summarize(self, text: str, max_length: int = 500) -> str:
        """Simple extractive fallback summarization."""
        # Split into sentences
        sentences = text.replace("\n", " ").split(". ")
        
        # Take first few sentences that fit within limit
        summary_parts = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary_parts.append(sentence)
            current_length += len(sentence) + 2
        
        if not summary_parts:
            return text[:max_length] + "..."
        
        return ". ".join(summary_parts) + "."
    
    async def _store_summary(self, db: Any, summary: Summary) -> None:
        """Store summary in SQLite."""
        import json
        
        await db.execute(
            """
            INSERT INTO session_summaries
            (id, session_id, summary_type, content, message_start_idx, message_end_idx,
             token_count, created_at, status, chroma_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.id,
                summary.session_id,
                summary.summary_type.value,
                summary.content,
                summary.message_start_idx,
                summary.message_end_idx,
                summary.token_count,
                summary.created_at.isoformat(),
                summary.status.value,
                summary.chroma_id,
                json.dumps(summary.metadata),
            ),
        )
    
    async def _vectorize_summary(self, summary: Summary) -> None:
        """Generate embedding and store in ChromaDB."""
        from ordinis.rag.vectordb.id_generator import generate_summary_vector_id
        
        try:
            # Generate deterministic ID
            chroma_id = generate_summary_vector_id(
                session_id=summary.session_id,
                summary_type=summary.summary_type.value,
                content=summary.content,
            )
            
            # Generate embedding
            embeddings = self.text_embedder.embed_texts([summary.content])
            
            # Store in ChromaDB
            collection = self.chroma_client.get_text_collection()
            collection.upsert(
                ids=[chroma_id],
                embeddings=embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else [embeddings[0]],
                documents=[summary.content],
                metadatas=[{
                    "entity_type": "summary",
                    "summary_type": summary.summary_type.value,
                    "session_id": summary.session_id,
                    "message_start_idx": summary.message_start_idx,
                    "message_end_idx": summary.message_end_idx,
                    "token_count": summary.token_count,
                    "indexed_at": _utcnow().isoformat(),
                }],
            )
            
            summary.chroma_id = chroma_id
            logger.debug(f"Vectorized summary {summary.id} as {chroma_id}")
            
        except Exception as e:
            logger.error(f"Failed to vectorize summary {summary.id}: {e}")
    
    async def get_session_summaries(self, session_id: str) -> list[Summary]:
        """Get all summaries for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of Summary objects ordered by message index
        """
        import aiosqlite
        import json
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute(
                """
                SELECT * FROM session_summaries
                WHERE session_id = ?
                ORDER BY message_start_idx ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            
            summaries = []
            for row in rows:
                summaries.append(Summary(
                    id=row["id"],
                    session_id=row["session_id"],
                    summary_type=SummaryType(row["summary_type"]),
                    content=row["content"],
                    message_start_idx=row["message_start_idx"],
                    message_end_idx=row["message_end_idx"],
                    token_count=row["token_count"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    status=SummaryStatus(row["status"]),
                    chroma_id=row["chroma_id"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ))
            
            return summaries
    
    async def search_summaries(
        self,
        query: str,
        n_results: int = 5,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search summaries semantically.
        
        Args:
            query: Search query
            n_results: Number of results to return
            session_id: Optional filter by session
            
        Returns:
            List of matching summaries with scores
        """
        # Generate query embedding
        query_embedding = self.text_embedder.embed_texts([query])[0]
        
        # Build filter
        where_filter: dict[str, Any] = {"entity_type": "summary"}
        if session_id:
            where_filter["session_id"] = session_id
        
        # Search ChromaDB
        collection = self.chroma_client.get_text_collection()
        results = collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        summaries = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"][0]):
                summaries.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        
        return summaries
    
    async def process_session_on_close(self, session_id: str) -> dict[str, Any]:
        """Process all pending summarization when a session closes.
        
        Creates any missing chunk summaries and the session summary.
        
        Args:
            session_id: Session ID being closed
            
        Returns:
            Processing statistics
        """
        stats = {"chunk_summaries": 0, "session_summary": False, "errors": []}
        
        try:
            # Create any missing chunk summaries
            while True:
                summary = await self.create_chunk_summary(session_id)
                if summary is None:
                    break
                stats["chunk_summaries"] += 1
            
            # Create session summary
            session_summary = await self.create_session_summary(session_id)
            if session_summary:
                stats["session_summary"] = True
            
            logger.info(f"Processed session {session_id}: {stats}")
            
        except Exception as e:
            stats["errors"].append(str(e))
            logger.error(f"Error processing session {session_id}: {e}")
        
        return stats
