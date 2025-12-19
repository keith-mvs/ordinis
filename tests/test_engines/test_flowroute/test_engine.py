"""
Tests for FlowRoute engine.

Tests cover:
- Order submission and tracking
- Order cancellation
- Fill processing
- Execution statistics
"""

import pytest

from ordinis.engines.flowroute.adapters.paper import PaperBrokerAdapter
from ordinis.engines.flowroute.core.engine import FlowRouteEngine
from ordinis.engines.flowroute.core.orders import (
    Fill,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
)


@pytest.fixture
def paper_broker():
    """Create paper broker adapter."""
    return PaperBrokerAdapter(slippage_bps=5.0, commission_per_share=0.005)


@pytest.fixture
def engine(paper_broker):
    """Create FlowRoute engine with paper broker."""
    return FlowRouteEngine(broker_adapter=paper_broker)


@pytest.mark.unit
def test_engine_initialization(paper_broker):
    """Test engine initialization."""
    engine = FlowRouteEngine(broker_adapter=paper_broker)

    assert engine._broker is not None
    assert len(engine._orders) == 0
    assert len(engine._active_orders) == 0


@pytest.mark.unit
def test_create_order_from_intent(engine):
    """Test creating order from intent."""
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)

    assert order.symbol == "AAPL"
    assert order.quantity == 100
    assert order.status == OrderStatus.CREATED
    assert order.intent_id == "intent-001"
    assert len(order.events) == 1  # Creation event


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_order(engine):
    """Test submitting order."""
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)
    success, message = await engine.submit_order(order)

    assert success is True
    assert order.status == OrderStatus.SUBMITTED
    assert order.broker_order_id is not None
    assert order.order_id in engine._active_orders


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_order_twice_fails(engine):
    """Test that submitting an order twice fails."""
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)
    await engine.submit_order(order)

    # Try to submit again
    success, message = await engine.submit_order(order)

    assert success is False
    assert "CREATED" in message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_order(engine):
    """Test cancelling order."""
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
    )

    order = engine.create_order_from_intent(intent)
    await engine.submit_order(order)

    success, message = await engine.cancel_order(order.order_id, "User requested")

    assert success is True
    assert order.status == OrderStatus.CANCELLED
    assert order.order_id not in engine._active_orders


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_nonexistent_order(engine):
    """Test cancelling non-existent order fails."""
    success, message = await engine.cancel_order("nonexistent", "Test")

    assert success is False
    assert "not found" in message


@pytest.mark.unit
def test_get_order(engine):
    """Test retrieving order by ID."""
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)

    retrieved = engine.get_order(order.order_id)

    assert retrieved is not None
    assert retrieved.order_id == order.order_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_active_orders(engine):
    """Test getting active orders."""
    # Create and submit two orders
    for i in range(2):
        intent = OrderIntent(
            intent_id=f"intent-{i:03d}",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )
        order = engine.create_order_from_intent(intent)
        await engine.submit_order(order)

    active = engine.get_active_orders()

    assert len(active) == 2
    assert all(o.is_active() for o in active)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_fill(engine):
    """Test processing fill notification."""
    from datetime import datetime

    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)

    fill = Fill(
        fill_id="fill-001",
        order_id=order.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=150.0,
        commission=0.50,
        timestamp=datetime.utcnow(),
    )

    await engine.process_fill(fill)

    assert order.filled_quantity == 100
    assert order.status == OrderStatus.FILLED
    assert len(order.fills) == 1
    assert order.order_id not in engine._active_orders


@pytest.mark.unit
def test_get_execution_stats_empty(engine):
    """Test execution stats with no orders."""
    stats = engine.get_execution_stats()

    assert stats["total_orders"] == 0
    assert stats["filled_orders"] == 0
    assert stats["fill_rate"] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_execution_stats_with_orders(engine):
    """Test execution stats with filled orders."""
    from datetime import datetime

    # Create and fill an order
    intent = OrderIntent(
        intent_id="intent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )

    order = engine.create_order_from_intent(intent)

    fill = Fill(
        fill_id="fill-001",
        order_id=order.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=150.0,
        commission=0.50,
        timestamp=datetime.utcnow(),
        slippage_bps=5.0,
    )

    await engine.process_fill(fill)

    stats = engine.get_execution_stats()

    assert stats["total_orders"] == 1
    assert stats["filled_orders"] == 1
    assert stats["fill_rate"] == 1.0
    assert stats["total_fills"] == 1
    assert stats["avg_slippage_bps"] == 5.0


@pytest.mark.unit
def test_engine_to_dict(engine):
    """Test converting engine state to dictionary."""
    state = engine.to_dict()

    assert "total_orders" in state
    assert "active_orders" in state
    assert "has_broker" in state
    assert state["has_broker"] is True
