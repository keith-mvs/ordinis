"""Tests for Order Repository."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ordinis.adapters.storage.repositories.order import OrderRepository


class TestOrderRepositoryInit:
    """Tests for OrderRepository initialization."""

    def test_init(self):
        """Test repository initialization."""
        mock_db = MagicMock()
        repo = OrderRepository(db=mock_db)

        assert repo.db == mock_db


class TestOrderRepositoryGetById:
    """Tests for get_by_id method."""

    @pytest.fixture
    def repo(self):
        """Create repository with mock database."""
        mock_db = MagicMock()
        mock_db.fetch_one = AsyncMock()
        return OrderRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repo):
        """Test getting order by ID when not found."""
        repo.db.fetch_one.return_value = None

        result = await repo.get_by_id("nonexistent")

        assert result is None


class TestOrderRepositoryGetActive:
    """Tests for get_active method."""

    @pytest.fixture
    def repo(self):
        """Create repository with mock database."""
        mock_db = MagicMock()
        mock_db.fetch_all = AsyncMock()
        return OrderRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_get_active_empty(self, repo):
        """Test getting active orders when none exist."""
        repo.db.fetch_all.return_value = []

        result = await repo.get_active()

        assert result == []


class TestOrderRepositoryGetByStatus:
    """Tests for get_by_status method."""

    @pytest.fixture
    def repo(self):
        """Create repository with mock database."""
        mock_db = MagicMock()
        mock_db.fetch_all = AsyncMock()
        return OrderRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_get_by_status(self, repo):
        """Test getting orders by status."""
        repo.db.fetch_all.return_value = []

        result = await repo.get_by_status("filled")

        assert result == []
        repo.db.fetch_all.assert_called_once()


class TestOrderRepositoryGetByBrokerId:
    """Tests for get_by_broker_id method."""

    @pytest.fixture
    def repo(self):
        """Create repository with mock database."""
        mock_db = MagicMock()
        mock_db.fetch_one = AsyncMock()
        return OrderRepository(db=mock_db)

    @pytest.mark.asyncio
    async def test_get_by_broker_id_not_found(self, repo):
        """Test getting order by broker ID when not found."""
        repo.db.fetch_one.return_value = None

        result = await repo.get_by_broker_id("nonexistent")

        assert result is None
