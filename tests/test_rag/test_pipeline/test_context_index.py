"""
Tests for RecentContextIndex module.

Tests cover:
- Initialization and index loading
- Session export recording
- Decision recording
- Query methods
- Index persistence
"""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# Store original tiktoken if it exists, then mock it for this module's import
_original_tiktoken = sys.modules.get("tiktoken")
_mock_tiktoken = MagicMock()
sys.modules["tiktoken"] = _mock_tiktoken

from ordinis.rag.pipeline.context_index import RecentContextIndex

# Restore original tiktoken after import to avoid polluting other tests
if _original_tiktoken is not None:
    sys.modules["tiktoken"] = _original_tiktoken
else:
    del sys.modules["tiktoken"]


class TestRecentContextIndexInit:
    """Tests for RecentContextIndex initialization."""

    @pytest.mark.unit
    def test_init_creates_index_file(self, tmp_path):
        """Test initialization creates index file."""
        index_path = tmp_path / "test_index.json"

        index = RecentContextIndex(index_path)

        assert index.index_path == index_path
        assert index.index_path.parent.exists()

    @pytest.mark.unit
    def test_init_loads_existing_index(self, tmp_path):
        """Test initialization loads existing index."""
        index_path = tmp_path / "test_index.json"
        existing_data = {
            "version": "1.0",
            "last_updated": "2024-01-01T00:00:00",
            "sessions": [{"session_id": "test_session"}],
            "decisions": [],
            "key_topics": ["topic1"],
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f)

        index = RecentContextIndex(index_path)

        assert len(index._data["sessions"]) == 1
        assert index._data["sessions"][0]["session_id"] == "test_session"

    @pytest.mark.unit
    def test_init_handles_corrupt_file(self, tmp_path):
        """Test initialization handles corrupt index file."""
        index_path = tmp_path / "test_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("not valid json{")

        index = RecentContextIndex(index_path)

        assert index._data["version"] == "1.0"
        assert index._data["sessions"] == []

    @pytest.mark.unit
    def test_empty_index_structure(self, tmp_path):
        """Test empty index has correct structure."""
        index_path = tmp_path / "test_index.json"
        index = RecentContextIndex(index_path)

        data = index._empty_index()

        assert "version" in data
        assert "last_updated" in data
        assert data["sessions"] == []
        assert data["decisions"] == []
        assert data["key_topics"] == []


class TestRecentContextIndexSessionExport:
    """Tests for session export recording."""

    @pytest.fixture
    def index(self, tmp_path):
        """Create index for testing."""
        return RecentContextIndex(tmp_path / "test_index.json")

    @pytest.mark.unit
    def test_add_session_export(self, index):
        """Test adding session export."""
        index.add_session_export(
            session_id="sess_123",
            export_path="/path/to/export.txt",
            message_count=100,
            context_usage=65.5,
        )

        assert len(index._data["sessions"]) == 1
        session = index._data["sessions"][0]
        assert session["session_id"] == "sess_123"
        assert session["message_count"] == 100
        assert session["context_usage"] == 65.5

    @pytest.mark.unit
    def test_add_session_export_with_topics(self, index):
        """Test adding session export with key topics."""
        index.add_session_export(
            session_id="sess_123",
            export_path="/path/to/export.txt",
            message_count=100,
            context_usage=65.5,
            key_topics=["trading", "risk"],
        )

        session = index._data["sessions"][0]
        assert session["key_topics"] == ["trading", "risk"]
        assert "trading" in index._data["key_topics"]
        assert "risk" in index._data["key_topics"]

    @pytest.mark.unit
    def test_add_session_export_limits_sessions(self, index):
        """Test session export limits to 50 sessions."""
        for i in range(60):
            index.add_session_export(
                session_id=f"sess_{i}",
                export_path=f"/path/to/export_{i}.txt",
                message_count=i,
                context_usage=50.0,
            )

        assert len(index._data["sessions"]) == 50

    @pytest.mark.unit
    def test_add_session_export_most_recent_first(self, index):
        """Test most recent session is first."""
        index.add_session_export(
            session_id="first",
            export_path="/path/first.txt",
            message_count=10,
            context_usage=50.0,
        )
        index.add_session_export(
            session_id="second",
            export_path="/path/second.txt",
            message_count=20,
            context_usage=60.0,
        )

        assert index._data["sessions"][0]["session_id"] == "second"
        assert index._data["sessions"][1]["session_id"] == "first"


class TestRecentContextIndexDecisions:
    """Tests for decision recording."""

    @pytest.fixture
    def index(self, tmp_path):
        """Create index for testing."""
        return RecentContextIndex(tmp_path / "test_index.json")

    @pytest.mark.unit
    def test_add_decision(self, index):
        """Test adding a decision."""
        index.add_decision(
            decision="Use paper trading for testing",
            context="Risk management discussion",
            priority=8,
        )

        assert len(index._data["decisions"]) == 1
        decision = index._data["decisions"][0]
        assert decision["decision"] == "Use paper trading for testing"
        assert decision["priority"] == 8

    @pytest.mark.unit
    def test_add_decision_with_session(self, index):
        """Test adding decision with session ID."""
        index.add_decision(
            decision="Implement new strategy",
            context="Strategy review",
            priority=6,
            session_id="sess_123",
        )

        decision = index._data["decisions"][0]
        assert decision["session_id"] == "sess_123"

    @pytest.mark.unit
    def test_add_decision_limits_to_100(self, index):
        """Test decisions limited to 100."""
        for i in range(110):
            index.add_decision(
                decision=f"Decision {i}",
                context=f"Context {i}",
                priority=5,
            )

        assert len(index._data["decisions"]) == 100

    @pytest.mark.unit
    def test_get_critical_decisions(self, index):
        """Test getting critical decisions."""
        index.add_decision("Low priority", "Context", priority=3)
        index.add_decision("High priority", "Context", priority=8)
        index.add_decision("Critical", "Context", priority=10)

        critical = index.get_critical_decisions(min_priority=7)

        assert len(critical) == 2
        assert all(d["priority"] >= 7 for d in critical)

    @pytest.mark.unit
    def test_get_critical_decisions_with_limit(self, index):
        """Test critical decisions respects limit."""
        for i in range(10):
            index.add_decision(f"Decision {i}", "Context", priority=8)

        critical = index.get_critical_decisions(min_priority=7, limit=5)

        assert len(critical) == 5


class TestRecentContextIndexQueries:
    """Tests for query methods."""

    @pytest.fixture
    def index(self, tmp_path):
        """Create index with test data."""
        index = RecentContextIndex(tmp_path / "test_index.json")
        index.add_session_export(
            session_id="sess_1",
            export_path="/path/1.txt",
            message_count=50,
            context_usage=40.0,
            key_topics=["trading"],
        )
        index.add_session_export(
            session_id="sess_2",
            export_path="/path/2.txt",
            message_count=100,
            context_usage=60.0,
            key_topics=["risk"],
        )
        return index

    @pytest.mark.unit
    def test_get_session_summary(self, index):
        """Test getting session summary."""
        summary = index.get_session_summary(limit=10)

        assert "sess_2" not in summary  # ID not in summary format
        assert "100 msgs" in summary
        assert "50 msgs" in summary

    @pytest.mark.unit
    def test_get_session_summary_empty(self, tmp_path):
        """Test session summary when empty."""
        index = RecentContextIndex(tmp_path / "empty_index.json")

        summary = index.get_session_summary()

        assert "No recent sessions indexed" in summary

    @pytest.mark.unit
    def test_get_session_summary_respects_limit(self, index):
        """Test session summary respects limit."""
        summary = index.get_session_summary(limit=1)

        # Should only show 1 entry
        lines = [l for l in summary.split("\n") if l.strip()]
        assert len(lines) == 1

    @pytest.mark.unit
    def test_get_topics(self, index):
        """Test getting topics."""
        topics = index.get_topics()

        assert "trading" in topics
        assert "risk" in topics

    @pytest.mark.unit
    def test_get_recent(self, index):
        """Test getting recent entries."""
        index.add_decision("Test decision", "Context", priority=5)

        recent = index.get_recent(limit=10)

        assert len(recent) >= 2
        assert any(e.get("context_type") == "session_export" for e in recent)

    @pytest.mark.unit
    def test_get_recent_sorted_by_timestamp(self, index):
        """Test recent entries sorted by timestamp."""
        recent = index.get_recent()

        # Most recent first
        timestamps = [e.get("timestamp", "") for e in recent if e.get("timestamp")]
        assert timestamps == sorted(timestamps, reverse=True)


class TestRecentContextIndexPersistence:
    """Tests for index persistence."""

    @pytest.mark.unit
    def test_save_and_reload(self, tmp_path):
        """Test saving and reloading index."""
        index_path = tmp_path / "test_index.json"

        index = RecentContextIndex(index_path)
        index.add_session_export(
            session_id="persist_test",
            export_path="/path/test.txt",
            message_count=42,
            context_usage=55.0,
        )

        # Create new instance to reload from disk
        index2 = RecentContextIndex(index_path)

        assert len(index2._data["sessions"]) == 1
        assert index2._data["sessions"][0]["session_id"] == "persist_test"

    @pytest.mark.unit
    def test_clear(self, tmp_path):
        """Test clearing index."""
        index = RecentContextIndex(tmp_path / "test_index.json")
        index.add_session_export(
            session_id="test",
            export_path="/path/test.txt",
            message_count=10,
            context_usage=50.0,
        )
        index.add_decision("Test", "Context", priority=5)

        index.clear()

        assert len(index._data["sessions"]) == 0
        assert len(index._data["decisions"]) == 0
