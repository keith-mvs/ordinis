"""
Tests for FlowRoute order types and lifecycle.

Tests cover:
- Order creation and validation
- Order lifecycle states
- Fill processing
"""

from datetime import datetime

import pytest

from src.engines.flowroute.core.orders import (
    ExecutionEvent,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
)


@pytest.mark.unit
def test_order_intent_creation():
    """Test creating an order intent."""
    intent = OrderIntent(
        intent_id="test-intent-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
        signal_id="signal-001",
        strategy_id="strategy-001",
    )

    assert intent.intent_id == "test-intent-001"
    assert intent.symbol == "AAPL"
    assert intent.quantity == 100
    assert intent.order_type == OrderType.LIMIT


@pytest.mark.unit
def test_order_creation():
    """Test creating an order."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    assert order.order_id == "order-001"
    assert order.status == OrderStatus.CREATED
    assert order.filled_quantity == 0
    assert order.remaining_quantity == 100


@pytest.mark.unit
def test_order_is_terminal():
    """Test order terminal state detection."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    # Created is not terminal
    assert order.is_terminal() is False

    # Filled is terminal
    order.status = OrderStatus.FILLED
    assert order.is_terminal() is True

    # Cancelled is terminal
    order.status = OrderStatus.CANCELLED
    assert order.is_terminal() is True

    # Submitted is not terminal
    order.status = OrderStatus.SUBMITTED
    assert order.is_terminal() is False


@pytest.mark.unit
def test_order_is_active():
    """Test order active state detection."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    # Created is not active
    assert order.is_active() is False

    # Submitted is active
    order.status = OrderStatus.SUBMITTED
    assert order.is_active() is True

    # Partially filled is active
    order.status = OrderStatus.PARTIALLY_FILLED
    assert order.is_active() is True

    # Filled is not active
    order.status = OrderStatus.FILLED
    assert order.is_active() is False


@pytest.mark.unit
def test_order_add_fill():
    """Test adding fill to order."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    # Add partial fill
    fill1 = Fill(
        fill_id="fill-001",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=50,
        price=150.0,
        commission=0.25,
        timestamp=datetime.utcnow(),
    )

    order.add_fill(fill1)

    assert order.filled_quantity == 50
    assert order.remaining_quantity == 50
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.avg_fill_price == 150.0

    # Add second fill
    fill2 = Fill(
        fill_id="fill-002",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=50,
        price=151.0,
        commission=0.25,
        timestamp=datetime.utcnow(),
    )

    order.add_fill(fill2)

    assert order.filled_quantity == 100
    assert order.remaining_quantity == 0
    assert order.status == OrderStatus.FILLED
    assert order.avg_fill_price == 150.5  # Average of 150 and 151


@pytest.mark.unit
def test_order_fill_percentage():
    """Test calculating fill percentage."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    assert order.fill_percentage() == 0.0

    fill = Fill(
        fill_id="fill-001",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=50,
        price=150.0,
        commission=0.25,
        timestamp=datetime.utcnow(),
    )

    order.add_fill(fill)

    assert order.fill_percentage() == 0.5


@pytest.mark.unit
def test_order_to_dict():
    """Test converting order to dictionary."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
    )

    order_dict = order.to_dict()

    assert order_dict["order_id"] == "order-001"
    assert order_dict["symbol"] == "AAPL"
    assert order_dict["order_type"] == "limit"
    assert order_dict["limit_price"] == 150.0


@pytest.mark.unit
def test_fill_creation():
    """Test creating a fill."""
    fill = Fill(
        fill_id="fill-001",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.0,
        commission=0.50,
        timestamp=datetime.utcnow(),
        slippage_bps=5.0,
    )

    assert fill.fill_id == "fill-001"
    assert fill.quantity == 100
    assert fill.price == 150.0
    assert fill.slippage_bps == 5.0


@pytest.mark.unit
def test_fill_to_dict():
    """Test converting fill to dictionary."""
    now = datetime.utcnow()
    fill = Fill(
        fill_id="fill-001",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.0,
        commission=0.50,
        timestamp=now,
    )

    fill_dict = fill.to_dict()

    assert fill_dict["fill_id"] == "fill-001"
    assert fill_dict["quantity"] == 100
    assert fill_dict["timestamp"] == now.isoformat()


@pytest.mark.unit
def test_execution_event_creation():
    """Test creating an execution event."""
    event = ExecutionEvent(
        event_id="event-001",
        order_id="order-001",
        event_type="order_submitted",
        timestamp=datetime.utcnow(),
        status_before=OrderStatus.CREATED,
        status_after=OrderStatus.SUBMITTED,
    )

    assert event.event_id == "event-001"
    assert event.event_type == "order_submitted"
    assert event.status_before == OrderStatus.CREATED
    assert event.status_after == OrderStatus.SUBMITTED


@pytest.mark.unit
def test_execution_event_to_dict():
    """Test converting event to dictionary."""
    now = datetime.utcnow()
    event = ExecutionEvent(
        event_id="event-001",
        order_id="order-001",
        event_type="order_filled",
        timestamp=now,
        status_before=OrderStatus.SUBMITTED,
        status_after=OrderStatus.FILLED,
    )

    event_dict = event.to_dict()

    assert event_dict["event_id"] == "event-001"
    assert event_dict["status_before"] == "submitted"
    assert event_dict["status_after"] == "filled"
