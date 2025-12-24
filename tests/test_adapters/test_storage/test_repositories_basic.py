"""Tests for storage repositories - basic operations."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from ordinis.adapters.storage.models import OrderRow, PositionRow, TradeRow
from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository


@pytest.fixture
def mock_db():
    """Create mock database manager."""
    db = MagicMock()
    db.fetch_one = AsyncMock()
    db.fetch_all = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.begin_transaction = AsyncMock()
    return db


@pytest.fixture
def order_repo(mock_db):
    """Create order repository with mock database."""
    return OrderRepository(mock_db)


@pytest.fixture
def position_repo(mock_db):
    """Create position repository with mock database."""
    return PositionRepository(mock_db)


@pytest.fixture
def trade_repo(mock_db):
    """Create trade repository with mock database."""
    return TradeRepository(mock_db)


class TestOrderRepository:
    """Test OrderRepository operations."""

    async def test_get_by_id_not_found(self, order_repo, mock_db):
        """Test get_by_id when order doesn't exist."""
        mock_db.fetch_one.return_value = None
        result = await order_repo.get_by_id("ORD-999")
        assert result is None

    async def test_get_by_broker_id_not_found(self, order_repo, mock_db):
        """Test get_by_broker_id when order doesn't exist."""
        mock_db.fetch_one.return_value = None
        result = await order_repo.get_by_broker_id("BROKER-999")
        assert result is None

    async def test_get_active_empty(self, order_repo, mock_db):
        """Test get_active when no active orders."""
        mock_db.fetch_all.return_value = []
        result = await order_repo.get_active()
        assert result == []

    async def test_get_by_symbol_empty(self, order_repo, mock_db):
        """Test get_by_symbol when no orders."""
        mock_db.fetch_all.return_value = []
        result = await order_repo.get_by_symbol("AAPL")
        assert result == []

    async def test_get_recent_empty(self, order_repo, mock_db):
        """Test get_recent when no orders."""
        mock_db.fetch_all.return_value = []
        result = await order_repo.get_recent(limit=10)
        assert result == []

    async def test_get_today_order_count(self, order_repo, mock_db):
        """Test get_today_order_count."""
        mock_db.fetch_one.return_value = (42,)
        result = await order_repo.get_today_order_count()
        assert result == 42

    async def test_get_today_fill_count(self, order_repo, mock_db):
        """Test get_today_fill_count."""
        mock_db.fetch_one.return_value = (15,)
        result = await order_repo.get_today_fill_count()
        assert result == 15


class TestPositionRepository:
    """Test PositionRepository operations."""

    async def test_get_by_symbol_not_found(self, position_repo, mock_db):
        """Test get_by_symbol when position doesn't exist."""
        mock_db.fetch_one.return_value = None
        result = await position_repo.get_by_symbol("AAPL")
        assert result is None

    async def test_get_all_empty(self, position_repo, mock_db):
        """Test get_all when no positions."""
        mock_db.fetch_all.return_value = []
        result = await position_repo.get_all()
        assert result == []

    async def test_get_active_empty(self, position_repo, mock_db):
        """Test get_active when no active positions."""
        mock_db.fetch_all.return_value = []
        result = await position_repo.get_active()
        assert result == []

    async def test_get_total_realized_pnl(self, position_repo, mock_db):
        """Test get_total_realized_pnl."""
        mock_db.fetch_one.return_value = (1234.56,)
        result = await position_repo.get_total_realized_pnl()
        assert result == 1234.56

    async def test_get_total_unrealized_pnl(self, position_repo, mock_db):
        """Test get_total_unrealized_pnl."""
        mock_db.fetch_one.return_value = (-567.89,)
        result = await position_repo.get_total_unrealized_pnl()
        assert result == -567.89

    async def test_get_latest_snapshot_not_found(self, position_repo, mock_db):
        """Test get_latest_snapshot when no snapshots."""
        mock_db.fetch_one.return_value = None
        result = await position_repo.get_latest_snapshot()
        assert result is None


class TestTradeRepository:
    """Test TradeRepository operations."""

    async def test_get_by_id_not_found(self, trade_repo, mock_db):
        """Test get_by_id when trade doesn't exist."""
        mock_db.fetch_one.return_value = None
        result = await trade_repo.get_by_id("TRD-999")
        assert result is None

    async def test_get_all_empty(self, trade_repo, mock_db):
        """Test get_all when no trades."""
        mock_db.fetch_all.return_value = []
        result = await trade_repo.get_all()
        assert result == []

    async def test_get_by_symbol_empty(self, trade_repo, mock_db):
        """Test get_by_symbol when no trades."""
        mock_db.fetch_all.return_value = []
        result = await trade_repo.get_by_symbol("AAPL")
        assert result == []

    async def test_get_by_date_range_empty(self, trade_repo, mock_db):
        """Test get_by_date_range when no trades."""
        mock_db.fetch_all.return_value = []
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)
        result = await trade_repo.get_by_date_range(start, end)
        assert result == []

    async def test_get_today_empty(self, trade_repo, mock_db):
        """Test get_today when no trades."""
        mock_db.fetch_all.return_value = []
        result = await trade_repo.get_today()
        assert result == []

    async def test_get_total_pnl(self, trade_repo, mock_db):
        """Test get_total_pnl."""
        mock_db.fetch_one.return_value = (5432.10,)
        result = await trade_repo.get_total_pnl()
        assert result == 5432.10

    async def test_get_today_pnl(self, trade_repo, mock_db):
        """Test get_today_pnl."""
        mock_db.fetch_one.return_value = (123.45,)
        result = await trade_repo.get_today_pnl()
        assert result == 123.45

    async def test_get_win_rate(self, trade_repo, mock_db):
        """Test get_win_rate."""
        mock_db.fetch_one.return_value = (65.5,)
        result = await trade_repo.get_win_rate()
        assert result == 65.5

    async def test_get_trade_count(self, trade_repo, mock_db):
        """Test get_trade_count."""
        mock_db.fetch_one.return_value = (100,)
        result = await trade_repo.get_trade_count()
        assert result == 100

    async def test_get_today_trade_count(self, trade_repo, mock_db):
        """Test get_today_trade_count."""
        mock_db.fetch_one.return_value = (5,)
        result = await trade_repo.get_today_trade_count()
        assert result == 5

    async def test_get_average_pnl(self, trade_repo, mock_db):
        """Test get_average_pnl."""
        mock_db.fetch_one.return_value = (54.32,)
        result = await trade_repo.get_average_pnl()
        assert result == 54.32

    async def test_get_average_winner(self, trade_repo, mock_db):
        """Test get_average_winner."""
        mock_db.fetch_one.return_value = (120.00,)
        result = await trade_repo.get_average_winner()
        assert result == 120.00

    async def test_get_average_loser(self, trade_repo, mock_db):
        """Test get_average_loser."""
        mock_db.fetch_one.return_value = (-80.00,)
        result = await trade_repo.get_average_loser()
        assert result == -80.00

    async def test_get_profit_factor_zero_loss(self, trade_repo, mock_db):
        """Test get_profit_factor when no losses."""
        mock_db.fetch_one.return_value = (1000.0, 0.0)
        result = await trade_repo.get_profit_factor()
        assert result == 0.0

    async def test_get_profit_factor_valid(self, trade_repo, mock_db):
        """Test get_profit_factor with valid data."""
        mock_db.fetch_one.return_value = (2000.0, 1000.0)
        result = await trade_repo.get_profit_factor()
        assert result == 2.0

    async def test_get_largest_winner(self, trade_repo, mock_db):
        """Test get_largest_winner."""
        mock_db.fetch_one.return_value = (500.00,)
        result = await trade_repo.get_largest_winner()
        assert result == 500.00

    async def test_get_largest_loser(self, trade_repo, mock_db):
        """Test get_largest_loser."""
        mock_db.fetch_one.return_value = (-300.00,)
        result = await trade_repo.get_largest_loser()
        assert result == -300.00


class TestModelConstruction:
    """Test model construction with proper fields."""

    def test_order_row_minimal(self):
        """Test OrderRow construction with minimal fields."""
        now = datetime.utcnow().isoformat()
        order = OrderRow(
            order_id="ORD-TEST",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            time_in_force="day",
            status="created",
            filled_quantity=0,
            remaining_quantity=100,
            created_at=now,
        )
        assert order.order_id == "ORD-TEST"
        assert order.symbol == "AAPL"
        assert order.quantity == 100

    def test_position_row_minimal(self):
        """Test PositionRow construction with minimal fields."""
        now = datetime.utcnow().isoformat()
        position = PositionRow(
            symbol="AAPL",
            side="LONG",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            realized_pnl=0.0,
            unrealized_pnl=500.0,
            entry_time=now,
            last_update=now,
        )
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.unrealized_pnl == 500.0

    def test_trade_row_minimal(self):
        """Test TradeRow construction with minimal fields."""
        now = datetime.utcnow().isoformat()
        trade = TradeRow(
            trade_id="TRD-TEST",
            symbol="AAPL",
            side="buy",
            entry_time=now,
            exit_time=now,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=3.33,
            duration_seconds=3600,
        )
        assert trade.trade_id == "TRD-TEST"
        assert trade.symbol == "AAPL"
        assert trade.pnl == 500.0


class TestModelTupleConversion:
    """Test model to_insert_tuple() methods."""

    def test_order_row_to_insert_tuple(self):
        """Test OrderRow to_insert_tuple."""
        now = datetime.utcnow().isoformat()
        order = OrderRow(
            order_id="ORD-TEST",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            time_in_force="day",
            status="created",
            filled_quantity=0,
            remaining_quantity=100,
            created_at=now,
        )
        tuple_data = order.to_insert_tuple()
        assert tuple_data[0] == "ORD-TEST"  # order_id
        assert tuple_data[1] == "AAPL"  # symbol
        assert tuple_data[2] == "buy"  # side
        assert tuple_data[3] == 100  # quantity

    def test_position_row_to_insert_tuple(self):
        """Test PositionRow to_insert_tuple."""
        now = datetime.utcnow().isoformat()
        position = PositionRow(
            symbol="AAPL",
            side="LONG",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            realized_pnl=0.0,
            unrealized_pnl=500.0,
            entry_time=now,
            last_update=now,
        )
        tuple_data = position.to_insert_tuple()
        assert tuple_data[0] == "AAPL"  # symbol
        assert tuple_data[1] == "LONG"  # side
        assert tuple_data[2] == 100  # quantity

    def test_trade_row_to_insert_tuple(self):
        """Test TradeRow to_insert_tuple."""
        now = datetime.utcnow().isoformat()
        trade = TradeRow(
            trade_id="TRD-TEST",
            symbol="AAPL",
            side="buy",
            entry_time=now,
            exit_time=now,
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=3.33,
            duration_seconds=3600,
        )
        tuple_data = trade.to_insert_tuple()
        assert tuple_data[0] == "TRD-TEST"  # trade_id
        assert tuple_data[1] == "AAPL"  # symbol
        assert tuple_data[2] == "buy"  # side
