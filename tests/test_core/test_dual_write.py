"""Tests for DualWriteManager and related classes.

Tests cover:
- WritePhase enum
- DualWriteResult dataclass
- SyncQueueEntry dataclass
- DualWriteManager initialization and methods
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import numpy as np
import pytest

from ordinis.core.dual_write import (
    WritePhase,
    DualWriteResult,
    SyncQueueEntry,
    DualWriteManager,
    dual_write,
    _utcnow,
)


class TestUtcNow:
    """Tests for _utcnow helper."""

    @pytest.mark.unit
    def test_returns_datetime(self):
        """Test _utcnow returns datetime."""
        result = _utcnow()
        assert isinstance(result, datetime)

    @pytest.mark.unit
    def test_returns_timezone_aware(self):
        """Test _utcnow returns timezone-aware datetime."""
        result = _utcnow()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc


class TestWritePhase:
    """Tests for WritePhase enum."""

    @pytest.mark.unit
    def test_all_phases_exist(self):
        """Test all write phases are defined."""
        assert WritePhase.PENDING.value == "pending"
        assert WritePhase.SQLITE_WRITE.value == "sqlite_write"
        assert WritePhase.SQLITE_COMMITTED.value == "sqlite_committed"
        assert WritePhase.CHROMA_WRITE.value == "chroma_write"
        assert WritePhase.COMPLETED.value == "completed"
        assert WritePhase.COMPENSATING.value == "compensating"
        assert WritePhase.FAILED.value == "failed"

    @pytest.mark.unit
    def test_phase_count(self):
        """Test correct number of phases."""
        assert len(WritePhase) == 7


class TestDualWriteResult:
    """Tests for DualWriteResult dataclass."""

    @pytest.mark.unit
    def test_create_success_result(self):
        """Test creating a successful result."""
        result = DualWriteResult(
            success=True,
            transaction_id="test-txn-123",
            sqlite_id="sql-id-456",
            chroma_id="chroma-id-789",
            phase=WritePhase.COMPLETED,
        )

        assert result.success is True
        assert result.transaction_id == "test-txn-123"
        assert result.sqlite_id == "sql-id-456"
        assert result.chroma_id == "chroma-id-789"
        assert result.phase == WritePhase.COMPLETED
        assert result.error is None
        assert result.compensated is False

    @pytest.mark.unit
    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = DualWriteResult(
            success=False,
            transaction_id="test-txn-fail",
            phase=WritePhase.FAILED,
            error="Connection timeout",
            compensated=True,
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.compensated is True

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        result = DualWriteResult(
            success=False,
            transaction_id="txn-1",
        )

        assert result.sqlite_id is None
        assert result.chroma_id is None
        assert result.phase == WritePhase.PENDING
        assert result.error is None
        assert result.compensated is False
        assert result.duration_ms == 0.0

    @pytest.mark.unit
    def test_to_dict(self):
        """Test to_dict serialization."""
        result = DualWriteResult(
            success=True,
            transaction_id="txn-dict",
            sqlite_id="sql-1",
            chroma_id="chr-1",
            phase=WritePhase.COMPLETED,
            duration_ms=15.5,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["transaction_id"] == "txn-dict"
        assert d["sqlite_id"] == "sql-1"
        assert d["chroma_id"] == "chr-1"
        assert d["phase"] == "completed"
        assert d["duration_ms"] == 15.5

    @pytest.mark.unit
    def test_to_dict_with_error(self):
        """Test to_dict includes error field."""
        result = DualWriteResult(
            success=False,
            transaction_id="txn-err",
            error="Failed to connect",
        )

        d = result.to_dict()
        assert d["error"] == "Failed to connect"


class TestSyncQueueEntry:
    """Tests for SyncQueueEntry dataclass."""

    @pytest.mark.unit
    def test_create_entry(self):
        """Test creating a sync queue entry."""
        entry = SyncQueueEntry(
            id="entry-1",
            entity_type="trade",
            entity_id="trade-123",
            operation="insert",
            payload={"data": "test"},
        )

        assert entry.id == "entry-1"
        assert entry.entity_type == "trade"
        assert entry.entity_id == "trade-123"
        assert entry.operation == "insert"
        assert entry.payload == {"data": "test"}

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values."""
        entry = SyncQueueEntry(
            id="entry-2",
            entity_type="order",
            entity_id="order-456",
            operation="update",
            payload={},
        )

        assert entry.retry_count == 0
        assert entry.max_retries == 3
        assert entry.last_attempt_at is None
        assert entry.error_message is None
        assert entry.created_at is not None

    @pytest.mark.unit
    def test_should_retry_true(self):
        """Test should_retry returns True when under limit."""
        entry = SyncQueueEntry(
            id="entry-3",
            entity_type="trade",
            entity_id="t-1",
            operation="insert",
            payload={},
            retry_count=2,
            max_retries=3,
        )

        assert entry.should_retry is True

    @pytest.mark.unit
    def test_should_retry_false(self):
        """Test should_retry returns False at limit."""
        entry = SyncQueueEntry(
            id="entry-4",
            entity_type="trade",
            entity_id="t-2",
            operation="insert",
            payload={},
            retry_count=3,
            max_retries=3,
        )

        assert entry.should_retry is False

    @pytest.mark.unit
    def test_should_retry_false_exceeded(self):
        """Test should_retry returns False when exceeded."""
        entry = SyncQueueEntry(
            id="entry-5",
            entity_type="trade",
            entity_id="t-3",
            operation="insert",
            payload={},
            retry_count=5,
            max_retries=3,
        )

        assert entry.should_retry is False


class TestDualWriteManagerInit:
    """Tests for DualWriteManager initialization."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock async repository."""
        repo = AsyncMock()
        repo.begin_transaction = AsyncMock()
        repo.commit = AsyncMock()
        repo.rollback = AsyncMock()
        repo.insert = AsyncMock(return_value="inserted-id")
        repo.execute = AsyncMock()
        return repo

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client."""
        client = MagicMock()
        collection = MagicMock()
        collection.upsert = MagicMock()
        collection.count = MagicMock(return_value=10)
        client.get_text_collection = MagicMock(return_value=collection)
        client.get_code_collection = MagicMock(return_value=collection)
        return client

    @pytest.mark.unit
    def test_init_default_params(self, mock_repository, mock_chroma_client):
        """Test initialization with default parameters."""
        manager = DualWriteManager(mock_repository, mock_chroma_client)

        assert manager.repository == mock_repository
        assert manager.chroma_client == mock_chroma_client
        assert manager.circuit_breaker_threshold == 5
        assert manager.sync_interval_seconds == 30.0
        assert manager.enable_background_sync is True

    @pytest.mark.unit
    def test_init_custom_params(self, mock_repository, mock_chroma_client):
        """Test initialization with custom parameters."""
        manager = DualWriteManager(
            mock_repository,
            mock_chroma_client,
            circuit_breaker_threshold=10,
            sync_interval_seconds=60.0,
            enable_background_sync=False,
        )

        assert manager.circuit_breaker_threshold == 10
        assert manager.sync_interval_seconds == 60.0
        assert manager.enable_background_sync is False

    @pytest.mark.unit
    def test_initial_state(self, mock_repository, mock_chroma_client):
        """Test initial state is correct."""
        manager = DualWriteManager(mock_repository, mock_chroma_client)

        assert manager.is_circuit_open is False
        assert manager.queue_size == 0
        assert manager._consecutive_failures == 0
        assert manager._total_writes == 0
        assert manager._successful_writes == 0
        assert manager._failed_writes == 0

    @pytest.mark.unit
    def test_get_metrics(self, mock_repository, mock_chroma_client):
        """Test get_metrics returns correct structure."""
        manager = DualWriteManager(mock_repository, mock_chroma_client)
        metrics = manager.get_metrics()

        assert "total_writes" in metrics
        assert "successful_writes" in metrics
        assert "failed_writes" in metrics
        assert "compensations" in metrics
        assert "queue_size" in metrics
        assert "circuit_open" in metrics
        assert "consecutive_failures" in metrics


class TestDualWriteManagerCircuitBreaker:
    """Tests for DualWriteManager circuit breaker behavior."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock async repository."""
        repo = AsyncMock()
        repo.begin_transaction = AsyncMock()
        repo.commit = AsyncMock()
        repo.rollback = AsyncMock()
        repo.insert = AsyncMock(return_value="id")
        repo.execute = AsyncMock()
        return repo

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client."""
        client = MagicMock()
        return client

    @pytest.mark.unit
    def test_handle_chroma_failure_increments(self, mock_repository, mock_chroma_client):
        """Test _handle_chroma_failure increments failure counter."""
        manager = DualWriteManager(
            mock_repository, mock_chroma_client, circuit_breaker_threshold=3
        )

        manager._handle_chroma_failure()
        assert manager._consecutive_failures == 1
        assert manager._circuit_open is False

    @pytest.mark.unit
    def test_circuit_opens_at_threshold(self, mock_repository, mock_chroma_client):
        """Test circuit opens when threshold reached."""
        manager = DualWriteManager(
            mock_repository, mock_chroma_client, circuit_breaker_threshold=3
        )

        for _ in range(3):
            manager._handle_chroma_failure()

        assert manager._circuit_open is True
        assert manager._consecutive_failures == 3

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_queue_for_sync(self, mock_repository, mock_chroma_client):
        """Test _queue_for_sync adds entry to queue."""
        manager = DualWriteManager(
            mock_repository, mock_chroma_client, enable_background_sync=False
        )

        manager._queue_for_sync("trade", "t-1", "insert", {"data": "test"})
        # Give time for task to be created
        await asyncio.sleep(0.01)

        assert manager.queue_size == 1


class TestDualWriteManagerContextManager:
    """Tests for DualWriteManager async context manager."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock async repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client."""
        return MagicMock()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_context_manager_entry_no_background(
        self, mock_repository, mock_chroma_client
    ):
        """Test context manager entry without background sync."""
        manager = DualWriteManager(
            mock_repository, mock_chroma_client, enable_background_sync=False
        )

        async with manager as m:
            assert m is manager
            assert manager._background_task is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_context_manager_exit(self, mock_repository, mock_chroma_client):
        """Test context manager exit sets shutdown event."""
        manager = DualWriteManager(
            mock_repository, mock_chroma_client, enable_background_sync=False
        )

        async with manager:
            pass

        assert manager._shutdown_event.is_set()
