"""
Tests for FlowRoute broker sync and pre-trade validation (Phase 1).

Tests cover:
- Broker state synchronization on startup
- Pre-trade buying power validation
- Position limit enforcement
- Account state caching and refresh
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from ordinis.engines.flowroute.core.engine import (
    AccountState,
    FlowRouteEngine,
)


@pytest.fixture
def mock_broker():
    """Create mock broker adapter."""
    broker = AsyncMock()

    # Mock account response
    account = MagicMock()
    account.equity = Decimal("100000")
    account.cash = Decimal("50000")
    account.buying_power = Decimal("50000")
    account.portfolio_value = Decimal("100000")
    broker.get_account = AsyncMock(return_value=account)

    # Mock positions response
    position1 = MagicMock()
    position1.symbol = "AAPL"
    position1.quantity = Decimal("100")
    position1.side = MagicMock(value="long")
    position1.avg_entry_price = Decimal("150")
    position1.market_value = Decimal("15000")
    position1.unrealized_pnl = Decimal("500")

    position2 = MagicMock()
    position2.symbol = "MSFT"
    position2.quantity = Decimal("50")
    position2.side = MagicMock(value="long")
    position2.avg_entry_price = Decimal("300")
    position2.market_value = Decimal("15000")
    position2.unrealized_pnl = Decimal("-200")

    broker.get_positions = AsyncMock(return_value=[position1, position2])

    # Mock order submission
    broker.submit_order = AsyncMock(
        return_value={
            "success": True,
            "broker_order_id": "BROKER-12345",
            "status": "accepted",
        }
    )

    return broker


@pytest.fixture
def mock_order():
    """Create mock order."""
    order = MagicMock()
    order.order_id = "test-order-001"
    order.symbol = "GOOGL"
    order.side = "buy"
    order.quantity = Decimal("10")
    order.limit_price = Decimal("150")
    order.status = MagicMock(value="created")
    order.events = []
    order.broker_response = None
    order.broker_order_id = None
    order.error_message = None
    order.submitted_at = None
    return order


@pytest.fixture
def engine(mock_broker):
    """Create FlowRoute engine with mock broker."""
    return FlowRouteEngine(
        broker_adapter=mock_broker,
        max_positions=20,
        min_buying_power=Decimal("1000"),
    )


class TestAccountState:
    """Tests for AccountState dataclass."""

    def test_is_stale_when_never_synced(self):
        """Account state is stale when never synced."""
        state = AccountState()
        assert state.is_stale() is True

    def test_is_stale_when_recently_synced(self):
        """Account state is not stale when recently synced."""
        state = AccountState(last_sync=datetime.utcnow())
        assert state.is_stale() is False

    def test_is_stale_when_old(self):
        """Account state is stale when sync is old."""
        old_time = datetime.utcnow() - timedelta(seconds=10)
        state = AccountState(last_sync=old_time)
        assert state.is_stale(max_age_seconds=5.0) is True

    def test_custom_max_age(self):
        """Custom max age is respected."""
        recent_time = datetime.utcnow() - timedelta(seconds=2)
        state = AccountState(last_sync=recent_time)

        # Not stale with 5 second max age
        assert state.is_stale(max_age_seconds=5.0) is False

        # Stale with 1 second max age
        assert state.is_stale(max_age_seconds=1.0) is True


class TestBrokerSync:
    """Tests for broker state synchronization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, engine, mock_broker):
        """Test successful initialization."""
        result = await engine.initialize()

        assert result.success is True
        assert result.positions_synced == 2
        assert result.account_synced is True
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_broker(self):
        """Test initialization fails without broker."""
        engine = FlowRouteEngine(broker_adapter=None)
        result = await engine.initialize()

        assert result.success is False
        assert "No broker adapter" in result.error

    @pytest.mark.asyncio
    async def test_sync_populates_account_state(self, engine, mock_broker):
        """Test sync populates account state correctly."""
        await engine.sync_broker_state()

        state = engine.get_account_state()
        assert state.equity == Decimal("100000")
        assert state.cash == Decimal("50000")
        assert state.buying_power == Decimal("50000")
        assert state.last_sync is not None

    @pytest.mark.asyncio
    async def test_sync_populates_positions(self, engine, mock_broker):
        """Test sync populates positions correctly."""
        await engine.sync_broker_state()

        positions = engine.get_all_positions()
        assert len(positions) == 2

        aapl = engine.get_position("AAPL")
        assert aapl is not None
        assert aapl.quantity == Decimal("100")
        assert aapl.avg_entry_price == Decimal("150")

    @pytest.mark.asyncio
    async def test_sync_detects_new_positions(self, engine, mock_broker):
        """Test sync detects newly opened positions."""
        # First sync
        await engine.sync_broker_state()

        # Add new position to broker
        new_position = MagicMock()
        new_position.symbol = "TSLA"
        new_position.quantity = Decimal("25")
        new_position.side = MagicMock(value="long")
        new_position.avg_entry_price = Decimal("200")
        new_position.market_value = Decimal("5000")
        new_position.unrealized_pnl = Decimal("100")

        current_positions = list(mock_broker.get_positions.return_value)
        current_positions.append(new_position)
        mock_broker.get_positions = AsyncMock(return_value=current_positions)

        # Second sync
        result = await engine.sync_broker_state()

        assert len(result.discrepancies) > 0
        assert any("New positions" in d for d in result.discrepancies)

    @pytest.mark.asyncio
    async def test_sync_detects_closed_positions(self, engine, mock_broker):
        """Test sync detects closed positions."""
        # First sync with 2 positions
        await engine.sync_broker_state()

        # Remove one position
        mock_broker.get_positions = AsyncMock(
            return_value=[mock_broker.get_positions.return_value[0]]
        )

        # Second sync
        result = await engine.sync_broker_state()

        assert len(result.discrepancies) > 0
        assert any("closed" in d.lower() for d in result.discrepancies)

    @pytest.mark.asyncio
    async def test_sync_handles_broker_error(self, engine, mock_broker):
        """Test sync handles broker errors gracefully."""
        mock_broker.get_account = AsyncMock(side_effect=Exception("Connection failed"))

        result = await engine.sync_broker_state()

        assert result.success is False
        assert "Connection failed" in result.error


class TestPreTradeValidation:
    """Tests for pre-trade validation."""

    @pytest.mark.asyncio
    async def test_validates_buying_power(self, engine, mock_broker):
        """Test buying power validation."""
        await engine.initialize()

        # Order within buying power
        valid, reason = engine._validate_pre_trade(
            MagicMock(symbol="GOOGL"),
            estimated_cost=Decimal("10000"),
        )
        assert valid is True

        # Order exceeds buying power
        valid, reason = engine._validate_pre_trade(
            MagicMock(symbol="GOOGL"),
            estimated_cost=Decimal("100000"),
        )
        assert valid is False
        assert "Insufficient buying power" in reason

    @pytest.mark.asyncio
    async def test_validates_position_limit(self, engine, mock_broker):
        """Test position limit validation."""
        # Set low position limit
        engine._max_positions = 2
        await engine.initialize()

        # Already at limit (2 positions)
        valid, reason = engine._validate_pre_trade(
            MagicMock(symbol="GOOGL"),  # New symbol
            estimated_cost=Decimal("1000"),
        )
        assert valid is False
        assert "Position limit" in reason

    @pytest.mark.asyncio
    async def test_allows_adding_to_existing_position(self, engine, mock_broker):
        """Test adding to existing position bypasses position limit."""
        engine._max_positions = 2
        await engine.initialize()

        # Adding to existing position should be allowed
        valid, reason = engine._validate_pre_trade(
            MagicMock(symbol="AAPL"),  # Existing symbol
            estimated_cost=Decimal("1000"),
        )
        assert valid is True

    @pytest.mark.asyncio
    async def test_validates_minimum_buying_power(self, engine, mock_broker):
        """Test minimum buying power threshold."""
        engine._min_buying_power = Decimal("100000")  # Higher than available
        await engine.initialize()

        valid, reason = engine._validate_pre_trade(
            MagicMock(symbol="GOOGL"),
            estimated_cost=Decimal("100"),
        )
        assert valid is False
        assert "below minimum" in reason


class TestSubmitOrderWithValidation:
    """Tests for order submission with pre-trade validation."""

    @pytest.mark.asyncio
    async def test_submit_refreshes_stale_account(self, engine, mock_broker, mock_order):
        """Test submit refreshes stale account state."""
        from ordinis.domain.enums import OrderStatus

        # Initialize but make state stale
        await engine.initialize()
        engine._account_state.last_sync = datetime.utcnow() - timedelta(seconds=10)

        # Mock order with correct status
        mock_order.status = OrderStatus.CREATED

        # Submit should refresh account
        await engine.submit_order(mock_order)

        # Account should have been queried again
        assert mock_broker.get_account.call_count >= 2

    @pytest.mark.asyncio
    async def test_submit_rejects_insufficient_funds(self, engine, mock_broker, mock_order):
        """Test submit rejects order when insufficient funds."""
        from ordinis.domain.enums import OrderStatus

        # Set up low buying power
        account = MagicMock()
        account.equity = Decimal("1000")
        account.cash = Decimal("500")
        account.buying_power = Decimal("500")
        account.portfolio_value = Decimal("1000")
        mock_broker.get_account = AsyncMock(return_value=account)

        await engine.initialize()

        # Order for expensive stock
        mock_order.status = OrderStatus.CREATED
        mock_order.limit_price = Decimal("1000")
        mock_order.quantity = Decimal("10")

        success, reason = await engine.submit_order(mock_order)

        assert success is False
        # Engine checks minimum buying power threshold first
        assert "Buying power" in reason or "buying power" in reason.lower()
        assert mock_order.status == OrderStatus.REJECTED


class TestEngineState:
    """Tests for engine state reporting."""

    @pytest.mark.asyncio
    async def test_to_dict_includes_account_state(self, engine, mock_broker):
        """Test to_dict includes account and position state."""
        await engine.initialize()

        state = engine.to_dict()

        assert "account_state" in state
        assert state["account_state"]["equity"] == "100000"
        assert state["account_state"]["buying_power"] == "50000"

        assert "positions" in state
        assert "AAPL" in state["positions"]
        assert state["position_count"] == 2
        assert state["initialized"] is True
