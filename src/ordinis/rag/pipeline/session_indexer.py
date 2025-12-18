"""Session log indexing pipeline for ChromaDB memory bank."""

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
import tiktoken

from ordinis.rag.config import get_config
from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.vectordb.schema import TextChunkMetadata


class SessionLogIndexer:
    """Index Claude Code session logs into ChromaDB for RAG retrieval.

    This enables continuity across session boundaries by storing
    conversation context in a searchable vector database.

    Usage:
        indexer = SessionLogIndexer()
        result = indexer.index_session_log(
            log_path=Path("session_export.txt"),
            session_id="abc123",
        )
    """

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        text_embedder: TextEmbedder | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        collection_name: str = "session_logs",
    ):
        """Initialize session indexer.

        Args:
            chroma_client: ChromaDB client instance
            text_embedder: Text embedder instance
            chunk_size: Token size per chunk
            chunk_overlap: Token overlap between chunks
            collection_name: ChromaDB collection name for sessions
        """
        self.config = get_config()
        self.chroma_client = chroma_client or ChromaClient()
        self.text_embedder = text_embedder or TextEmbedder()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        # Ensure collection exists
        try:
            self.chroma_client.get_or_create_collection(collection_name)
        except Exception as e:
            logger.warning(f"Could not create collection {collection_name}: {e}")

        logger.info(f"Session indexer initialized (collection={collection_name})")

    def index_session_log(
        self,
        log_path: Path | str,
        session_id: str,
        metadata_extra: dict[str, Any] | None = None,
    ) -> dict:
        """Index a session log file.

        Args:
            log_path: Path to the session log file
            session_id: Unique session identifier
            metadata_extra: Additional metadata to store

        Returns:
            Dictionary with indexing statistics
        """
        log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f"Session log not found: {log_path}")

        logger.info(f"Indexing session log: {log_path}")

        # Read content
        content = log_path.read_text(encoding="utf-8")
        file_size = log_path.stat().st_size

        # Extract timestamp from filename if present
        # Format: {timestamp}_{session_id}_*.txt
        export_time = datetime.now().isoformat()
        try:
            filename_parts = log_path.stem.split("_")
            if len(filename_parts) >= 1 and filename_parts[0].isdigit():
                # Parse YYYYMMDDHHMMSS format
                ts_str = filename_parts[0]
                if len(ts_str) == 14:
                    export_time = datetime.strptime(ts_str, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            pass

        # Chunk the content
        chunks = self._chunk_text(content)
        total_tokens = sum(len(self.tokenizer.encode(chunk)) for chunk in chunks)

        logger.info(f"Created {len(chunks)} chunks ({total_tokens} tokens)")

        # Create metadata for each chunk
        base_metadata = {
            "session_id": session_id,
            "export_time": export_time,
            "source_file": str(log_path.name),
            "content_type": "session_log",
            **(metadata_extra or {}),
        }

        # Generate embeddings and store
        all_texts = []
        all_metadata = []
        all_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"session_{session_id}_chunk{i}"
            chunk_meta = TextChunkMetadata(
                source=str(log_path.name),
                chunk_index=i,
                section=f"chunk_{i}",
            )
            # Add session-specific metadata
            meta_dict = chunk_meta.model_dump(exclude_none=True)
            meta_dict.update(base_metadata)
            meta_dict["chunk_index"] = i
            meta_dict["total_chunks"] = len(chunks)

            all_texts.append(chunk)
            all_metadata.append(meta_dict)
            all_ids.append(chunk_id)

        # Embed in batches
        batch_size = 32
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            batch_metadata = all_metadata[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            embeddings = self.text_embedder.embed(batch_texts)

            self.chroma_client.add_texts(
                collection_name=self.collection_name,
                texts=batch_texts,
                embeddings=embeddings,
                metadata=batch_metadata,
                ids=batch_ids,
            )

        stats = {
            "session_id": session_id,
            "chunks": len(chunks),
            "tokens": total_tokens,
            "file_size": file_size,
            "export_time": export_time,
        }

        logger.success(f"Indexed session {session_id}: {len(chunks)} chunks")
        return stats

    def search_sessions(
        self,
        query: str,
        n_results: int = 10,
        session_id: str | None = None,
    ) -> list[dict]:
        """Search session logs for relevant context.

        Args:
            query: Search query
            n_results: Number of results to return
            session_id: Optional filter for specific session

        Returns:
            List of matching chunks with metadata
        """
        # Build filter
        where_filter = None
        if session_id:
            where_filter = {"session_id": session_id}

        # Embed query
        query_embedding = self.text_embedder.embed([query])[0]

        # Search
        results = self.chroma_client.query(
            collection_name=self.collection_name,
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
        )

        # Format results
        formatted = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances, strict=False):
                formatted.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "relevance": 1 - dist if dist <= 1 else 0,
                    }
                )

        return formatted

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text using token-based splitting with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            # Move forward with overlap
            next_start = end - self.chunk_overlap
            start = max(next_start, start + 1)

        return chunks

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Get metadata for recent indexed sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        try:
            # Query for distinct sessions
            results = self.chroma_client.query(
                collection_name=self.collection_name,
                query_texts=["session context"],
                n_results=limit * 5,  # Get extra to dedupe
            )

            # Extract unique sessions
            seen_sessions = set()
            sessions = []

            if results and results.get("metadatas"):
                for meta in results["metadatas"][0]:
                    session_id = meta.get("session_id")
                    if session_id and session_id not in seen_sessions:
                        seen_sessions.add(session_id)
                        sessions.append(
                            {
                                "session_id": session_id,
                                "export_time": meta.get("export_time"),
                                "source_file": meta.get("source_file"),
                            }
                        )
                        if len(sessions) >= limit:
                            break

            return sessions

        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return []
