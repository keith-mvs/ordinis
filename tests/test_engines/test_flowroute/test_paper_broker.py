"""
Tests for paper trading broker adapter.

Tests cover:
- Order submission and cancellation
- Fill simulation
- Position tracking
- Account management
"""

import pytest

from ordinis.engines.flowroute.adapters.paper import PaperBrokerAdapter
from ordinis.engines.flowroute.core.orders import Order, OrderType


@pytest.fixture
def paper_broker():
    """Create paper broker adapter."""
    return PaperBrokerAdapter(slippage_bps=5.0, commission_per_share=0.005, fill_delay_ms=100.0)


@pytest.mark.unit
def test_paper_broker_initialization(paper_broker):
    """Test paper broker initialization."""
    assert paper_broker.slippage_bps == 5.0
    assert paper_broker.commission_per_share == 0.005
    assert paper_broker._cash == 100000.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_order(paper_broker):
    """Test submitting order to paper broker."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    response = await paper_broker.submit_order(order)

    assert response["success"] is True
    assert "broker_order_id" in response
    assert response["status"] == "accepted"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_order(paper_broker):
    """Test cancelling order with paper broker."""
    response = await paper_broker.cancel_order("PAPER-12345678")

    assert response["success"] is True
    assert response["status"] == "cancelled"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_positions_empty(paper_broker):
    """Test getting positions when none exist."""
    positions = await paper_broker.get_positions()

    assert len(positions) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_account(paper_broker):
    """Test getting account information."""
    account = await paper_broker.get_account()

    assert account["cash"] == 100000.0
    assert account["total_equity"] == 100000.0
    assert account["positions"] == 0


@pytest.mark.unit
def test_simulate_fill_buy(paper_broker):
    """Test simulating buy order fill."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    fill = paper_broker.simulate_fill(order, fill_price=150.0, current_price=150.0)

    assert fill.symbol == "AAPL"
    assert fill.quantity == 100
    assert fill.price == 150.0
    assert fill.commission == 0.5  # 100 * 0.005

    # Check position created
    assert "AAPL" in paper_broker._positions
    position = paper_broker._positions["AAPL"]
    assert position["quantity"] == 100
    assert position["avg_price"] == 150.0

    # Check cash reduced
    assert paper_broker._cash < 100000.0


@pytest.mark.unit
def test_simulate_fill_sell(paper_broker):
    """Test simulating sell order fill."""
    # First buy to create position
    buy_order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(buy_order, fill_price=150.0)

    # Then sell
    sell_order = Order(
        order_id="order-002",
        symbol="AAPL",
        side="sell",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    fill = paper_broker.simulate_fill(sell_order, fill_price=155.0)

    assert fill.quantity == 100
    assert fill.price == 155.0

    # Position should be closed
    assert "AAPL" not in paper_broker._positions

    # Cash should be higher than initial
    assert paper_broker._cash > 100000.0


@pytest.mark.unit
def test_simulate_partial_sell(paper_broker):
    """Test simulating partial sell."""
    # Buy 100 shares
    buy_order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(buy_order, fill_price=150.0)

    # Sell 50 shares
    sell_order = Order(
        order_id="order-002",
        symbol="AAPL",
        side="sell",
        quantity=50,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(sell_order, fill_price=155.0)

    # Position should remain with 50 shares
    assert "AAPL" in paper_broker._positions
    assert paper_broker._positions["AAPL"]["quantity"] == 50


@pytest.mark.unit
def test_multiple_fills_average_price(paper_broker):
    """Test average price calculation with multiple fills."""
    # First buy
    order1 = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(order1, fill_price=150.0)

    # Second buy at different price
    order2 = Order(
        order_id="order-002",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(order2, fill_price=160.0)

    # Average price should be 155.0
    position = paper_broker._positions["AAPL"]
    assert position["quantity"] == 200
    assert position["avg_price"] == 155.0


@pytest.mark.unit
def test_get_fills(paper_broker):
    """Test getting all fills."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    paper_broker.simulate_fill(order, fill_price=150.0)

    fills = paper_broker.get_fills()

    assert len(fills) == 1
    assert fills[0].symbol == "AAPL"


@pytest.mark.unit
def test_reset(paper_broker):
    """Test resetting paper broker state."""
    # Create some fills and positions
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(order, fill_price=150.0)

    # Reset
    paper_broker.reset(initial_cash=200000.0)

    assert paper_broker._cash == 200000.0
    assert len(paper_broker._positions) == 0
    assert len(paper_broker._fills) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_positions_with_position(paper_broker):
    """Test getting positions after creating one."""
    order = Order(
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    paper_broker.simulate_fill(order, fill_price=150.0)

    positions = await paper_broker.get_positions()

    assert len(positions) == 1
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["quantity"] == 100
