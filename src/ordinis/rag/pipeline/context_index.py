"""Recent context index for session continuity.

Provides fast lookup of recent session context without full RAG retrieval.
Used by session-context-loader.py to inject context at session start.
"""

from datetime import datetime
import json
from pathlib import Path

from loguru import logger


class RecentContextIndex:
    """Lightweight index for recent session context.

    Stores session summaries and key decisions in a local JSON file
    for fast retrieval at session startup. This complements the full
    ChromaDB index with quick-access recent context.

    Usage:
        index = RecentContextIndex()
        index.add_session_export(session_id, export_path, message_count, context_usage)
        summary = index.get_session_summary(limit=10)
    """

    def __init__(self, index_path: Path | str | None = None):
        """Initialize recent context index.

        Args:
            index_path: Path to index JSON file (default: ~/.claude/logs/recent_context.json)
        """
        if index_path is None:
            index_path = Path.home() / ".claude" / "logs" / "recent_context.json"
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load context index: {e}")
                self._data = self._empty_index()
        else:
            self._data = self._empty_index()

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save context index: {e}")

    def _empty_index(self) -> dict:
        """Create empty index structure."""
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "sessions": [],
            "decisions": [],
            "key_topics": [],
        }

    def add_session_export(
        self,
        session_id: str,
        export_path: str | Path,
        message_count: int,
        context_usage: float,
        key_topics: list[str] | None = None,
    ) -> None:
        """Record a session export.

        Args:
            session_id: Session identifier
            export_path: Path to exported session file
            message_count: Number of messages in session
            context_usage: Context usage percentage (0-100)
            key_topics: Optional list of key topics from session
        """
        export_record = {
            "session_id": session_id,
            "export_path": str(export_path),
            "export_time": datetime.now().isoformat(),
            "message_count": message_count,
            "context_usage": context_usage,
            "key_topics": key_topics or [],
        }

        # Add to sessions list (most recent first)
        self._data["sessions"].insert(0, export_record)

        # Keep only last 50 sessions
        self._data["sessions"] = self._data["sessions"][:50]

        # Update key topics
        if key_topics:
            for topic in key_topics:
                if topic not in self._data["key_topics"]:
                    self._data["key_topics"].append(topic)
            self._data["key_topics"] = self._data["key_topics"][:100]

        self._data["last_updated"] = datetime.now().isoformat()
        self._save_index()

        logger.info(f"Added session export: {session_id}")

    def add_decision(
        self,
        decision: str,
        context: str,
        priority: int = 5,
        session_id: str | None = None,
    ) -> None:
        """Record a key decision for future reference.

        Args:
            decision: The decision made
            context: Context around the decision
            priority: Priority level (1-10, higher = more important)
            session_id: Optional session this decision was made in
        """
        decision_record = {
            "decision": decision,
            "context": context,
            "priority": priority,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "context_type": "decision",
        }

        self._data["decisions"].insert(0, decision_record)
        self._data["decisions"] = self._data["decisions"][:100]
        self._data["last_updated"] = datetime.now().isoformat()
        self._save_index()

    def get_session_summary(self, limit: int = 10) -> str:
        """Get formatted summary of recent sessions.

        Args:
            limit: Maximum number of sessions to include

        Returns:
            Formatted string summary
        """
        sessions = self._data.get("sessions", [])[:limit]

        if not sessions:
            return "No recent sessions indexed."

        lines = []
        for i, session in enumerate(sessions, 1):
            time_str = session.get("export_time", "unknown")[:19]
            msg_count = session.get("message_count", "?")
            topics = session.get("key_topics", [])
            topic_str = ", ".join(topics[:3]) if topics else "general"

            lines.append(f"  {i}. [{time_str}] {msg_count} msgs - {topic_str}")

        return "\n".join(lines)

    def get_critical_decisions(self, min_priority: int = 7, limit: int = 10) -> list[dict]:
        """Get high-priority decisions.

        Args:
            min_priority: Minimum priority level to include
            limit: Maximum decisions to return

        Returns:
            List of decision records
        """
        decisions = self._data.get("decisions", [])
        filtered = [d for d in decisions if d.get("priority", 0) >= min_priority]
        return filtered[:limit]

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get all recent entries (sessions + decisions) sorted by time.

        Args:
            limit: Maximum entries to return

        Returns:
            List of entry records with context_type field
        """
        entries = []

        for session in self._data.get("sessions", []):
            entries.append(
                {
                    **session,
                    "context_type": "session_export",
                    "timestamp": session.get("export_time"),
                }
            )

        for decision in self._data.get("decisions", []):
            entries.append(
                {
                    **decision,
                    "context_type": "decision",
                }
            )

        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return entries[:limit]

    def get_topics(self) -> list[str]:
        """Get all recorded topics."""
        return self._data.get("key_topics", [])

    def clear(self) -> None:
        """Clear all index data."""
        self._data = self._empty_index()
        self._save_index()
        logger.info("Context index cleared")
