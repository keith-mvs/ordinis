"""
Tests for Order domain models.
Tests cover:
- Fill model
- ExecutionEvent model
- Order model with validations
- Order lifecycle methods
"""

from datetime import UTC, datetime
from unittest.mock import patch
import pytest

from ordinis.domain.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from ordinis.domain.orders import ExecutionEvent, Fill, Order


class TestFill:
    """Tests for Fill model."""

    @pytest.mark.unit
    def test_fill_creation(self):
        """Test creating a Fill."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.50,
        )

        assert fill.fill_id == "fill_123"
        assert fill.order_id == "order_456"
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == 100
        assert fill.price == 150.50
        assert fill.commission == 0.0
        assert fill.slippage == 0.0

    @pytest.mark.unit
    def test_fill_with_all_fields(self):
        """Test Fill with all optional fields."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            price=155.00,
            commission=1.50,
            slippage=0.001,
            slippage_bps=10.0,
            vs_arrival_bps=5.0,
            multiplier=100.0,
            metadata={"venue": "NYSE"},
        )

        assert fill.commission == 1.50
        assert fill.slippage == 0.001
        assert fill.slippage_bps == 10.0
        assert fill.vs_arrival_bps == 5.0
        assert fill.multiplier == 100.0
        assert fill.metadata == {"venue": "NYSE"}

    @pytest.mark.unit
    def test_fill_total_cost_buy(self):
        """Test total_cost for buy order."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.00,
            commission=1.50,
        )

        # base_cost = 100 * 150 * 1 = 15000
        # total = 15000 + 1.50 = 15001.50
        assert fill.total_cost == 15001.50

    @pytest.mark.unit
    def test_fill_total_cost_sell(self):
        """Test total_cost for sell order."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            price=150.00,
            commission=1.50,
        )

        # base_cost = 100 * 150 * 1 = 15000
        # total = 15000 - 1.50 = 14998.50
        assert fill.total_cost == 14998.50

    @pytest.mark.unit
    def test_fill_total_cost_with_multiplier(self):
        """Test total_cost with multiplier (e.g., options)."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=5.00,
            multiplier=100.0,
            commission=6.50,
        )

        # base_cost = 10 * 5 * 100 = 5000
        # total = 5000 + 6.50 = 5006.50
        assert fill.total_cost == 5006.50

    @pytest.mark.unit
    def test_fill_net_proceeds_sell(self):
        """Test net_proceeds for sell order."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            price=150.00,
            commission=1.50,
        )

        # net = 100 * 150 * 1 - 1.50 = 14998.50
        assert fill.net_proceeds == 14998.50

    @pytest.mark.unit
    def test_fill_net_proceeds_buy(self):
        """Test net_proceeds for buy order (negative)."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_456",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.00,
            commission=1.50,
        )

        # net = -(100 * 150 * 1 + 1.50) = -15001.50
        assert fill.net_proceeds == -15001.50


class TestExecutionEvent:
    """Tests for ExecutionEvent model."""

    @pytest.mark.unit
    def test_execution_event_creation(self):
        """Test creating an ExecutionEvent."""
        event = ExecutionEvent(
            event_id="evt_123",
            order_id="order_456",
            event_type="SUBMITTED",
            status_after=OrderStatus.SUBMITTED,
        )

        assert event.event_id == "evt_123"
        assert event.order_id == "order_456"
        assert event.event_type == "SUBMITTED"
        assert event.status_after == OrderStatus.SUBMITTED
        assert event.status_before is None
        assert event.error_code is None

    @pytest.mark.unit
    def test_execution_event_with_error(self):
        """Test ExecutionEvent with error details."""
        event = ExecutionEvent(
            event_id="evt_123",
            order_id="order_456",
            event_type="ERROR",
            status_before=OrderStatus.SUBMITTED,
            status_after=OrderStatus.ERROR,
            error_code="INSUFFICIENT_FUNDS",
            error_message="Not enough buying power",
            retry_count=2,
        )

        assert event.error_code == "INSUFFICIENT_FUNDS"
        assert event.error_message == "Not enough buying power"
        assert event.retry_count == 2


class TestOrder:
    """Tests for Order model."""

    @pytest.mark.unit
    def test_order_creation_market(self):
        """Test creating a market order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.DAY
        assert order.status == OrderStatus.CREATED
        assert order.filled_quantity == 0
        assert order.order_id is not None

    @pytest.mark.unit
    def test_order_creation_limit(self):
        """Test creating a limit order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.00

    @pytest.mark.unit
    def test_order_limit_requires_price(self):
        """Test limit order requires limit_price."""
        with pytest.raises(ValueError, match="Limit orders require a limit_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
            )

    @pytest.mark.unit
    def test_order_stop_requires_price(self):
        """Test stop order requires stop_price."""
        with pytest.raises(ValueError, match="Stop orders require a stop_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP,
            )

    @pytest.mark.unit
    def test_order_stop_limit_requires_both_prices(self):
        """Test stop-limit order requires both prices."""
        with pytest.raises(ValueError, match="Stop-Limit orders require a limit_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP_LIMIT,
                stop_price=145.00,
            )

        with pytest.raises(ValueError, match="Stop-Limit orders require a stop_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP_LIMIT,
                limit_price=150.00,
            )

    @pytest.mark.unit
    def test_order_stop_limit_valid(self):
        """Test valid stop-limit order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            limit_price=150.00,
            stop_price=145.00,
        )

        assert order.limit_price == 150.00
        assert order.stop_price == 145.00

    @pytest.mark.unit
    def test_order_timestamp_alias(self):
        """Test timestamp property is alias for created_at."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        assert order.timestamp == order.created_at

    @pytest.mark.unit
    def test_order_remaining_quantity(self):
        """Test remaining_quantity calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=40,
        )

        assert order.remaining_quantity == 60

    @pytest.mark.unit
    def test_order_is_terminal_filled(self):
        """Test is_terminal for filled order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.FILLED,
        )

        assert order.is_terminal() is True

    @pytest.mark.unit
    def test_order_is_terminal_cancelled(self):
        """Test is_terminal for cancelled order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.CANCELLED,
        )

        assert order.is_terminal() is True

    @pytest.mark.unit
    def test_order_is_terminal_rejected(self):
        """Test is_terminal for rejected order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.REJECTED,
        )

        assert order.is_terminal() is True

    @pytest.mark.unit
    def test_order_is_terminal_expired(self):
        """Test is_terminal for expired order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.EXPIRED,
        )

        assert order.is_terminal() is True

    @pytest.mark.unit
    def test_order_is_terminal_error(self):
        """Test is_terminal for error order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.ERROR,
        )

        assert order.is_terminal() is True

    @pytest.mark.unit
    def test_order_is_terminal_false(self):
        """Test is_terminal is False for active order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.SUBMITTED,
        )

        assert order.is_terminal() is False

    @pytest.mark.unit
    def test_order_is_active_submitted(self):
        """Test is_active for submitted order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.SUBMITTED,
        )

        assert order.is_active() is True

    @pytest.mark.unit
    def test_order_is_active_acknowledged(self):
        """Test is_active for acknowledged order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.ACKNOWLEDGED,
        )

        assert order.is_active() is True

    @pytest.mark.unit
    def test_order_is_active_partially_filled(self):
        """Test is_active for partially filled order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.PARTIALLY_FILLED,
        )

        assert order.is_active() is True

    @pytest.mark.unit
    def test_order_is_active_false(self):
        """Test is_active is False for terminal order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.FILLED,
        )

        assert order.is_active() is False

    @pytest.mark.unit
    def test_order_is_filled(self):
        """Test is_filled property."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=100,
        )

        assert order.is_filled is True

    @pytest.mark.unit
    def test_order_is_filled_false(self):
        """Test is_filled property when not fully filled."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=50,
        )

        assert order.is_filled is False

    @pytest.mark.unit
    def test_order_is_partial(self):
        """Test is_partial property."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=50,
        )

        assert order.is_partial is True

    @pytest.mark.unit
    def test_order_is_partial_false_empty(self):
        """Test is_partial is False when no fills."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=0,
        )

        assert order.is_partial is False

    @pytest.mark.unit
    def test_order_is_partial_false_full(self):
        """Test is_partial is False when fully filled."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=100,
        )

        assert order.is_partial is False

    @pytest.mark.unit
    def test_order_fill_percentage(self):
        """Test fill_percentage calculation."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=25,
        )

        assert order.fill_percentage() == 0.25

    @pytest.mark.unit
    def test_order_fill_percentage_zero_quantity(self):
        """Test fill_percentage with zero quantity."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.quantity = 0  # Force zero quantity

        assert order.fill_percentage() == 0.0

    @pytest.mark.unit
    def test_order_add_fill(self):
        """Test add_fill updates order correctly."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        fill = Fill(
            fill_id="fill_1",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            price=150.00,
        )

        order.add_fill(fill)

        assert len(order.fills) == 1
        assert order.filled_quantity == 50
        assert order.avg_fill_price == 150.00
        assert order.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.unit
    def test_order_add_fill_complete(self):
        """Test add_fill completes order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        fill1 = Fill(
            fill_id="fill_1",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            price=150.00,
        )

        fill2 = Fill(
            fill_id="fill_2",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            price=151.00,
        )

        order.add_fill(fill1)
        order.add_fill(fill2)

        assert len(order.fills) == 2
        assert order.filled_quantity == 100
        # avg_fill_price = (50*150 + 50*151) / 100 = 150.50
        assert order.avg_fill_price == 150.50
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None

    @pytest.mark.unit
    def test_order_with_metadata(self):
        """Test order with metadata."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            metadata={"strategy": "momentum", "urgency": "high"},
        )

        assert order.metadata["strategy"] == "momentum"
        assert order.metadata["urgency"] == "high"

    @pytest.mark.unit
    def test_order_with_source_tracking(self):
        """Test order with source tracking fields."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            intent_id="intent_123",
            signal_id="signal_456",
            strategy_id="strat_789",
        )

        assert order.intent_id == "intent_123"
        assert order.signal_id == "signal_456"
        assert order.strategy_id == "strat_789"
