"""Tests for database manager and repositories."""

from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from ordinis.adapters.storage.database import DatabaseManager
from ordinis.adapters.storage.models import OrderRow, PositionRow, TradeRow
from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.system_state import SystemStateRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
async def db_manager(temp_db_path):
    """Create and initialize database manager."""
    manager = DatabaseManager(db_path=temp_db_path, auto_backup=False)
    await manager.initialize()
    yield manager
    await manager.shutdown()


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    @pytest.mark.asyncio
    async def test_initialize(self, temp_db_path):
        """Test database initialization."""
        manager = DatabaseManager(db_path=temp_db_path, auto_backup=False)
        result = await manager.initialize()
        assert result is True
        assert manager.is_connected
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, temp_db_path):
        """Test graceful shutdown."""
        manager = DatabaseManager(db_path=temp_db_path, auto_backup=False)
        await manager.initialize()
        await manager.shutdown()
        assert not manager.is_connected

    @pytest.mark.asyncio
    async def test_integrity_check(self, db_manager):
        """Test integrity check passes on valid database."""
        result = await db_manager._check_integrity()
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_and_fetch(self, db_manager):
        """Test basic SQL execution."""
        # Insert test data
        await db_manager.execute(
            "INSERT INTO system_state (key, value, value_type) VALUES (?, ?, ?)",
            ("test_key", "test_value", "string"),
        )
        await db_manager.commit()

        # Fetch data
        row = await db_manager.fetch_one(
            "SELECT value FROM system_state WHERE key = ?",
            ("test_key",),
        )
        assert row is not None
        assert row[0] == "test_value"


class TestPositionRepository:
    """Tests for PositionRepository."""

    @pytest.mark.asyncio
    async def test_upsert_position(self, db_manager):
        """Test position upsert."""
        repo = PositionRepository(db_manager)

        position = PositionRow(
            symbol="AAPL",
            side="LONG",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            realized_pnl=0.0,
            unrealized_pnl=500.0,
            last_update=datetime.utcnow().isoformat(),
        )

        result = await repo.upsert(position)
        assert result is True

        # Retrieve and verify
        fetched = await repo.get_by_symbol("AAPL")
        assert fetched is not None
        assert fetched.symbol == "AAPL"
        assert fetched.quantity == 100

    @pytest.mark.asyncio
    async def test_get_active_positions(self, db_manager):
        """Test getting active positions."""
        repo = PositionRepository(db_manager)

        # Create active position
        await repo.upsert(
            PositionRow(
                symbol="MSFT",
                side="LONG",
                quantity=50,
                avg_cost=300.0,
                current_price=310.0,
                last_update=datetime.utcnow().isoformat(),
            )
        )

        # Create flat position
        await repo.upsert(
            PositionRow(
                symbol="GOOGL",
                side="FLAT",
                quantity=0,
                avg_cost=0.0,
                current_price=0.0,
                last_update=datetime.utcnow().isoformat(),
            )
        )

        active = await repo.get_active()
        assert len(active) == 1
        assert active[0].symbol == "MSFT"

    @pytest.mark.asyncio
    async def test_update_price(self, db_manager):
        """Test price update with P&L calculation."""
        repo = PositionRepository(db_manager)

        await repo.upsert(
            PositionRow(
                symbol="TSLA",
                side="LONG",
                quantity=10,
                avg_cost=200.0,
                current_price=200.0,
                unrealized_pnl=0.0,
                last_update=datetime.utcnow().isoformat(),
            )
        )

        # Update price
        await repo.update_price("TSLA", 220.0)

        # Verify P&L calculated
        position = await repo.get_by_symbol("TSLA")
        assert position.current_price == 220.0
        assert position.unrealized_pnl == 200.0  # (220-200) * 10


class TestOrderRepository:
    """Tests for OrderRepository."""

    @pytest.mark.asyncio
    async def test_create_order(self, db_manager):
        """Test order creation."""
        repo = OrderRepository(db_manager)

        order = OrderRow(
            order_id="test-order-001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            remaining_quantity=100,
            created_at=datetime.utcnow().isoformat(),
        )

        result = await repo.create(order)
        assert result is True

        fetched = await repo.get_by_id("test-order-001")
        assert fetched is not None
        assert fetched.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_update_status(self, db_manager):
        """Test order status update."""
        repo = OrderRepository(db_manager)

        await repo.create(
            OrderRow(
                order_id="test-order-002",
                symbol="MSFT",
                side="buy",
                quantity=50,
                order_type="limit",
                limit_price=300.0,
                remaining_quantity=50,
                created_at=datetime.utcnow().isoformat(),
            )
        )

        await repo.update_status("test-order-002", "submitted")

        order = await repo.get_by_id("test-order-002")
        assert order.status == "submitted"

    @pytest.mark.asyncio
    async def test_get_active_orders(self, db_manager):
        """Test getting active orders."""
        repo = OrderRepository(db_manager)

        # Create active order
        await repo.create(
            OrderRow(
                order_id="active-001",
                symbol="AAPL",
                side="buy",
                quantity=100,
                order_type="market",
                status="submitted",
                remaining_quantity=100,
                created_at=datetime.utcnow().isoformat(),
            )
        )

        # Create filled order
        await repo.create(
            OrderRow(
                order_id="filled-001",
                symbol="MSFT",
                side="sell",
                quantity=50,
                order_type="market",
                status="filled",
                remaining_quantity=0,
                created_at=datetime.utcnow().isoformat(),
            )
        )

        active = await repo.get_active()
        assert len(active) == 1
        assert active[0].order_id == "active-001"


class TestSystemStateRepository:
    """Tests for SystemStateRepository."""

    @pytest.mark.asyncio
    async def test_kill_switch_default(self, db_manager):
        """Test kill switch defaults to inactive."""
        repo = SystemStateRepository(db_manager)
        is_active = await repo.is_kill_switch_active()
        assert is_active is False

    @pytest.mark.asyncio
    async def test_activate_kill_switch(self, db_manager):
        """Test kill switch activation."""
        repo = SystemStateRepository(db_manager)

        result = await repo.activate_kill_switch("Test emergency")
        assert result is True

        is_active = await repo.is_kill_switch_active()
        assert is_active is True

        info = await repo.get_kill_switch_info()
        assert info["active"] is True
        assert info["reason"] == "Test emergency"
        assert info["trading_enabled"] is False

    @pytest.mark.asyncio
    async def test_deactivate_kill_switch(self, db_manager):
        """Test kill switch deactivation."""
        repo = SystemStateRepository(db_manager)

        await repo.activate_kill_switch("Test")
        await repo.deactivate_kill_switch()

        is_active = await repo.is_kill_switch_active()
        assert is_active is False

        trading_enabled = await repo.is_trading_enabled()
        assert trading_enabled is True

    @pytest.mark.asyncio
    async def test_checkpoints(self, db_manager):
        """Test checkpoint recording."""
        repo = SystemStateRepository(db_manager)

        await repo.record_startup()
        startup = await repo.get_last_startup()
        assert startup is not None

        await repo.record_shutdown()
        shutdown = await repo.get_last_shutdown()
        assert shutdown is not None

        clean = await repo.was_clean_shutdown()
        assert clean is True

    @pytest.mark.asyncio
    async def test_risk_limits(self, db_manager):
        """Test risk limit configuration."""
        repo = SystemStateRepository(db_manager)

        await repo.set_daily_loss_limit(2000.0)
        limit = await repo.get_daily_loss_limit()
        assert limit == 2000.0

        await repo.set_max_drawdown_pct(10.0)
        drawdown = await repo.get_max_drawdown_pct()
        assert drawdown == 10.0


class TestTradeRepository:
    """Tests for TradeRepository."""

    @pytest.mark.asyncio
    async def test_create_trade(self, db_manager):
        """Test trade creation."""
        repo = TradeRepository(db_manager)

        trade = TradeRow(
            trade_id="trade-001",
            symbol="AAPL",
            side="LONG",
            entry_time=datetime.utcnow().isoformat(),
            exit_time=datetime.utcnow().isoformat(),
            entry_price=150.0,
            exit_price=160.0,
            quantity=100,
            pnl=1000.0,
            pnl_pct=6.67,
            commission=2.0,
            duration_seconds=3600.0,
        )

        result = await repo.create(trade)
        assert result is True

        fetched = await repo.get_by_id("trade-001")
        assert fetched is not None
        assert fetched.pnl == 1000.0

    @pytest.mark.asyncio
    async def test_performance_metrics(self, db_manager):
        """Test performance metric calculations."""
        repo = TradeRepository(db_manager)

        # Create winning trade
        await repo.create(
            TradeRow(
                trade_id="win-001",
                symbol="AAPL",
                side="LONG",
                entry_time=datetime.utcnow().isoformat(),
                exit_time=datetime.utcnow().isoformat(),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=10.0,
                duration_seconds=3600.0,
            )
        )

        # Create losing trade
        await repo.create(
            TradeRow(
                trade_id="loss-001",
                symbol="MSFT",
                side="LONG",
                entry_time=datetime.utcnow().isoformat(),
                exit_time=datetime.utcnow().isoformat(),
                entry_price=200.0,
                exit_price=190.0,
                quantity=50,
                pnl=-500.0,
                pnl_pct=-5.0,
                duration_seconds=1800.0,
            )
        )

        total_pnl = await repo.get_total_pnl()
        assert total_pnl == 500.0

        win_rate = await repo.get_win_rate()
        assert win_rate == 50.0

        summary = await repo.get_performance_summary()
        assert summary["total_trades"] == 2
        assert summary["profit_factor"] == 2.0  # 1000 / 500
