"""Tests for TradeVectorIngester module.

Tests cover:
- SyncResult dataclass operations
- SyncMode enum values
- TradeVectorIngester initialization
- Query building for different sync modes
- Trade to document conversion
- Batch processing
- Sync status retrieval
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ordinis.rag.pipeline.trade_ingester import (
    SyncMode,
    SyncResult,
    TradeVectorIngester,
    _utcnow,
)


class TestUtcNow:
    """Tests for _utcnow helper function."""

    @pytest.mark.unit
    def test_returns_datetime(self):
        """Test _utcnow returns datetime object."""
        result = _utcnow()
        assert isinstance(result, datetime)

    @pytest.mark.unit
    def test_returns_timezone_aware(self):
        """Test _utcnow returns timezone-aware datetime."""
        result = _utcnow()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc


class TestSyncMode:
    """Tests for SyncMode enum."""

    @pytest.mark.unit
    def test_incremental_value(self):
        """Test INCREMENTAL mode value."""
        assert SyncMode.INCREMENTAL.value == "incremental"

    @pytest.mark.unit
    def test_full_value(self):
        """Test FULL mode value."""
        assert SyncMode.FULL.value == "full"

    @pytest.mark.unit
    def test_session_value(self):
        """Test SESSION mode value."""
        assert SyncMode.SESSION.value == "session"

    @pytest.mark.unit
    def test_all_modes_unique(self):
        """Test all modes have unique values."""
        values = [m.value for m in SyncMode]
        assert len(values) == len(set(values))


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are correct."""
        result = SyncResult(mode=SyncMode.INCREMENTAL)
        assert result.total_found == 0
        assert result.synced == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.duration_seconds == 0.0
        assert result.errors == []

    @pytest.mark.unit
    def test_success_rate_empty(self):
        """Test success rate when no trades found."""
        result = SyncResult(mode=SyncMode.INCREMENTAL)
        assert result.success_rate == 100.0

    @pytest.mark.unit
    def test_success_rate_all_synced(self):
        """Test success rate when all trades synced."""
        result = SyncResult(mode=SyncMode.INCREMENTAL, total_found=10, synced=10)
        assert result.success_rate == 100.0

    @pytest.mark.unit
    def test_success_rate_partial(self):
        """Test success rate with partial sync."""
        result = SyncResult(mode=SyncMode.INCREMENTAL, total_found=100, synced=75)
        assert result.success_rate == 75.0

    @pytest.mark.unit
    def test_success_rate_none_synced(self):
        """Test success rate when none synced."""
        result = SyncResult(mode=SyncMode.INCREMENTAL, total_found=10, synced=0)
        assert result.success_rate == 0.0

    @pytest.mark.unit
    def test_to_dict(self):
        """Test to_dict conversion."""
        result = SyncResult(
            mode=SyncMode.FULL,
            total_found=50,
            synced=45,
            failed=5,
            skipped=0,
            duration_seconds=10.5,
            errors=["error1", "error2"],
        )
        d = result.to_dict()

        assert d["mode"] == "full"
        assert d["total_found"] == 50
        assert d["synced"] == 45
        assert d["failed"] == 5
        assert d["skipped"] == 0
        assert d["duration_seconds"] == 10.5
        assert d["success_rate"] == 90.0
        assert d["errors"] == ["error1", "error2"]

    @pytest.mark.unit
    def test_to_dict_limits_errors(self):
        """Test to_dict limits errors to 10."""
        errors = [f"error{i}" for i in range(15)]
        result = SyncResult(mode=SyncMode.INCREMENTAL, errors=errors)
        d = result.to_dict()
        assert len(d["errors"]) == 10


class TestTradeVectorIngesterInit:
    """Tests for TradeVectorIngester initialization."""

    @pytest.mark.unit
    def test_init_with_path(self):
        """Test initialization with explicit path."""
        ingester = TradeVectorIngester(db_path="/test/path.db")
        assert ingester.db_path == Path("/test/path.db")

    @pytest.mark.unit
    def test_init_with_string_path(self):
        """Test initialization with string path."""
        ingester = TradeVectorIngester(db_path="data/test.db")
        assert ingester.db_path == Path("data/test.db")

    @pytest.mark.unit
    def test_init_default_batch_size(self):
        """Test default batch size."""
        ingester = TradeVectorIngester(db_path="test.db")
        assert ingester.batch_size == 100

    @pytest.mark.unit
    def test_init_custom_batch_size(self):
        """Test custom batch size."""
        ingester = TradeVectorIngester(db_path="test.db", batch_size=50)
        assert ingester.batch_size == 50

    @pytest.mark.unit
    def test_init_lazy_clients(self):
        """Test clients are lazily initialized."""
        ingester = TradeVectorIngester(db_path="test.db")
        assert ingester._chroma_client is None
        assert ingester._text_embedder is None

    @pytest.mark.unit
    def test_init_with_clients(self):
        """Test initialization with pre-created clients."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        ingester = TradeVectorIngester(
            db_path="test.db",
            chroma_client=mock_chroma,
            text_embedder=mock_embedder,
        )

        assert ingester._chroma_client is mock_chroma
        assert ingester._text_embedder is mock_embedder

    @pytest.mark.unit
    @patch.dict("os.environ", {"ORDINIS_DB_PATH": "custom/path.db"})
    def test_default_db_path_from_env(self):
        """Test default DB path from environment."""
        path = TradeVectorIngester._get_default_db_path()
        assert path == Path("custom/path.db")

    @pytest.mark.unit
    @patch.dict("os.environ", {}, clear=True)
    def test_default_db_path_fallback(self):
        """Test default DB path fallback."""
        # Need to remove the env var if it exists
        import os
        os.environ.pop("ORDINIS_DB_PATH", None)
        path = TradeVectorIngester._get_default_db_path()
        assert path == Path("data/ordinis.db")


class TestTradeVectorIngesterClients:
    """Tests for TradeVectorIngester client properties."""

    @pytest.mark.unit
    def test_chroma_client_lazy_create(self):
        """Test ChromaDB client is lazily created."""
        ingester = TradeVectorIngester(db_path="test.db")
        # Client should be None until accessed
        assert ingester._chroma_client is None

    @pytest.mark.unit
    def test_text_embedder_lazy_create(self):
        """Test TextEmbedder is lazily created."""
        ingester = TradeVectorIngester(db_path="test.db")
        # Client should be None until accessed
        assert ingester._text_embedder is None


class TestTradeVectorIngesterBuildQuery:
    """Tests for query building."""

    @pytest.fixture
    def ingester(self):
        """Create ingester for tests."""
        return TradeVectorIngester(db_path="test.db")

    @pytest.mark.unit
    def test_build_query_incremental(self, ingester):
        """Test query for incremental mode."""
        query, params = ingester._build_query(SyncMode.INCREMENTAL, None, None)

        assert "WHERE chroma_synced = 0 OR chroma_synced IS NULL" in query
        assert "ORDER BY timestamp ASC" in query
        assert params == ()

    @pytest.mark.unit
    def test_build_query_full(self, ingester):
        """Test query for full mode."""
        query, params = ingester._build_query(SyncMode.FULL, None, None)

        assert "WHERE" not in query or "WHERE chroma_synced" not in query
        assert "ORDER BY timestamp ASC" in query
        assert params == ()

    @pytest.mark.unit
    def test_build_query_session(self, ingester):
        """Test query for session mode."""
        query, params = ingester._build_query(SyncMode.SESSION, None, "session-123")

        assert "WHERE session_id = ?" in query
        assert params == ("session-123",)

    @pytest.mark.unit
    def test_build_query_with_limit(self, ingester):
        """Test query with max_trades limit."""
        query, params = ingester._build_query(SyncMode.INCREMENTAL, 50, None)

        assert "LIMIT 50" in query

    @pytest.mark.unit
    def test_build_query_base_columns(self, ingester):
        """Test query includes required columns."""
        query, _ = ingester._build_query(SyncMode.INCREMENTAL, None, None)

        assert "id" in query
        assert "symbol" in query
        assert "side" in query
        assert "quantity" in query
        assert "price" in query
        assert "timestamp" in query
        assert "strategy_id" in query
        assert "session_id" in query


class TestTradeVectorIngesterTradeToDocument:
    """Tests for trade to document conversion."""

    @pytest.fixture
    def ingester(self):
        """Create ingester for tests."""
        return TradeVectorIngester(db_path="test.db")

    @pytest.mark.unit
    def test_basic_trade(self, ingester):
        """Test basic trade conversion."""
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "price": 150.50,
            "strategy_id": "momentum",
            "timestamp": "2024-01-15T10:30:00",
        }

        doc = ingester._trade_to_document(trade)

        assert "BUY 100 shares of AAPL" in doc
        assert "$150.50" in doc
        assert "momentum" in doc

    @pytest.mark.unit
    def test_trade_with_pnl_positive(self, ingester):
        """Test trade with positive P&L."""
        trade = {
            "side": "sell",
            "quantity": 50,
            "symbol": "TSLA",
            "price": 200.00,
            "pnl": 250.75,
        }

        doc = ingester._trade_to_document(trade)

        assert "P&L: +$250.75" in doc

    @pytest.mark.unit
    def test_trade_with_pnl_negative(self, ingester):
        """Test trade with negative P&L."""
        trade = {
            "side": "sell",
            "quantity": 50,
            "symbol": "TSLA",
            "price": 200.00,
            "pnl": -125.50,
        }

        doc = ingester._trade_to_document(trade)

        assert "P&L: -$125.50" in doc

    @pytest.mark.unit
    def test_trade_with_notes(self, ingester):
        """Test trade with notes."""
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "notes": "Strong earnings report",
        }

        doc = ingester._trade_to_document(trade)

        assert "Notes: Strong earnings report" in doc

    @pytest.mark.unit
    def test_trade_with_metadata_reason(self, ingester):
        """Test trade with metadata containing reason."""
        import json
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "metadata": json.dumps({"reason": "Technical breakout"}),
        }

        doc = ingester._trade_to_document(trade)

        assert "Reason: Technical breakout" in doc

    @pytest.mark.unit
    def test_trade_with_metadata_signal_strength(self, ingester):
        """Test trade with metadata containing signal strength."""
        import json
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "metadata": json.dumps({"signal_strength": 0.85}),
        }

        doc = ingester._trade_to_document(trade)

        assert "Signal strength: 0.85" in doc

    @pytest.mark.unit
    def test_trade_with_dict_metadata(self, ingester):
        """Test trade with metadata as dict (not JSON string)."""
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "metadata": {"reason": "Momentum signal"},
        }

        doc = ingester._trade_to_document(trade)

        assert "Reason: Momentum signal" in doc

    @pytest.mark.unit
    def test_trade_with_invalid_metadata(self, ingester):
        """Test trade with invalid JSON metadata."""
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "metadata": "not valid json {",
        }

        # Should not raise, just skip metadata
        doc = ingester._trade_to_document(trade)
        assert "BUY 100 shares of AAPL" in doc

    @pytest.mark.unit
    def test_trade_without_price(self, ingester):
        """Test trade without price."""
        trade = {
            "side": "buy",
            "quantity": 100,
            "symbol": "AAPL",
            "price": None,
        }

        doc = ingester._trade_to_document(trade)

        # Should not crash, just exclude price
        assert "BUY 100 shares of AAPL" in doc


class TestTradeVectorIngesterSync:
    """Tests for sync operations."""

    @pytest.fixture
    def mock_ingester(self):
        """Create ingester with mocked clients."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 384]

        ingester = TradeVectorIngester(
            db_path="test.db",
            chroma_client=mock_chroma,
            text_embedder=mock_embedder,
        )
        return ingester

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_unsynced_trades_no_trades(self, mock_ingester):
        """Test sync when no unsynced trades exist."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        mock_db = AsyncMock()
        mock_db.row_factory = None
        mock_db.execute.return_value = mock_cursor

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_db

            result = await mock_ingester.sync_unsynced_trades()

            assert result.total_found == 0
            assert result.synced == 0
            assert result.mode == SyncMode.INCREMENTAL

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_session_trades_calls_correct_mode(self, mock_ingester):
        """Test sync_session_trades uses SESSION mode."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_db

            result = await mock_ingester.sync_session_trades("session-123")

            assert result.mode == SyncMode.SESSION

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_reindex_clears_vectors(self, mock_ingester):
        """Test full_reindex clears existing vectors."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall.return_value = []

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["trade:1", "trade:2"]}
        mock_ingester._chroma_client.get_text_collection.return_value = mock_collection

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_db

            result = await mock_ingester.full_reindex()

            assert result.mode == SyncMode.FULL
            mock_collection.delete.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handles_db_error(self, mock_ingester):
        """Test sync handles database errors gracefully."""
        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.side_effect = Exception("DB connection failed")

            result = await mock_ingester.sync_unsynced_trades()

            assert len(result.errors) > 0
            assert "DB connection failed" in result.errors[0]


class TestTradeVectorIngesterStatus:
    """Tests for sync status retrieval."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_sync_status(self):
        """Test get_sync_status returns correct structure."""
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 50
        mock_chroma.get_text_collection.return_value = mock_collection

        ingester = TradeVectorIngester(
            db_path="test.db",
            chroma_client=mock_chroma,
        )

        # Mock database queries
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.side_effect = [(100,), (75,), (25,)]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_db

            status = await ingester.get_sync_status()

            assert "sqlite_total" in status
            assert "sqlite_synced" in status
            assert "sqlite_unsynced" in status
            assert "chroma_count" in status
            assert "sync_percentage" in status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_sync_status_chroma_error(self):
        """Test get_sync_status handles ChromaDB error."""
        mock_chroma = MagicMock()
        mock_chroma.get_text_collection.side_effect = Exception("Chroma unavailable")

        ingester = TradeVectorIngester(
            db_path="test.db",
            chroma_client=mock_chroma,
        )

        mock_cursor = AsyncMock()
        mock_cursor.fetchone.side_effect = [(100,), (100,), (0,)]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_cursor

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_db

            status = await ingester.get_sync_status()

            assert status["chroma_count"] == -1


class TestProcessBatch:
    """Tests for batch processing."""

    @pytest.fixture
    def ingester(self):
        """Create ingester with mocked clients."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 384, [0.2] * 384]

        return TradeVectorIngester(
            db_path="test.db",
            chroma_client=mock_chroma,
            text_embedder=mock_embedder,
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_success(self, ingester):
        """Test successful batch processing."""
        mock_db = AsyncMock()

        # Create mock rows that act like dictionaries
        row1 = {
            "id": 1,
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.0,
            "timestamp": "2024-01-15T10:00:00",
            "strategy_id": "momentum",
            "session_id": "session-1",
            "metadata": None,
            "notes": None,
            "pnl": 50.0,
            "chroma_synced": 0,
            "chroma_id": None,
        }
        row2 = {
            "id": 2,
            "symbol": "TSLA",
            "side": "sell",
            "quantity": 50,
            "price": 200.0,
            "timestamp": "2024-01-15T11:00:00",
            "strategy_id": "momentum",
            "session_id": "session-1",
            "metadata": None,
            "notes": None,
            "pnl": -25.0,
            "chroma_synced": 0,
            "chroma_id": None,
        }

        with patch("ordinis.rag.vectordb.id_generator.generate_trade_vector_id") as mock_id_gen:
            mock_id_gen.side_effect = ["vec-1", "vec-2"]

            result = await ingester._process_batch(mock_db, [row1, row2])

            assert result["synced"] == 2
            assert result["failed"] == 0
            assert result["errors"] == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_batch_embedding_error(self, ingester):
        """Test batch processing handles embedding errors."""
        mock_db = AsyncMock()
        ingester._text_embedder.embed_texts.side_effect = Exception("Embedding failed")

        row = {
            "id": 1,
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.0,
            "timestamp": "2024-01-15T10:00:00",
            "strategy_id": "momentum",
            "session_id": "session-1",
            "metadata": None,
            "notes": None,
            "pnl": None,
            "chroma_synced": 0,
            "chroma_id": None,
        }

        with patch("ordinis.rag.vectordb.id_generator.generate_trade_vector_id") as mock_id_gen:
            mock_id_gen.return_value = "vec-1"

            result = await ingester._process_batch(mock_db, [row])

            assert result["synced"] == 0
            assert result["failed"] == 1
            assert "Embedding failed" in result["errors"][0]
