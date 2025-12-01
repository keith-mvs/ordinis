"""
Test paper broker with simulated and real market data.

This script tests the paper trading broker adapter:
1. Without market data (manual fills)
2. With market data plugin (auto fills)
"""

import asyncio
import sys

sys.path.insert(0, "src")

from engines.flowroute.adapters.paper import PaperBrokerAdapter
from engines.flowroute.core.orders import Order, OrderType


async def test_manual_fills():
    """Test paper broker with manual fill simulation."""
    print("\n" + "=" * 60)
    print("[TEST 1] Paper Broker - Manual Fills")
    print("=" * 60)

    broker = PaperBrokerAdapter(
        slippage_bps=5.0,
        commission_per_share=0.005,
        fill_delay_ms=100.0,
    )

    # Check initial account
    account = await broker.get_account()
    print(f"\n[INIT] Cash: ${account['cash']:,.2f}")

    # Submit order
    order = Order(
        order_id="TEST-001",
        symbol="SPY",
        side="buy",
        quantity=10,
        order_type=OrderType.MARKET,
    )

    print(f"\n[ORDER] Submitting: {order.side.upper()} {order.quantity} {order.symbol}")
    result = await broker.submit_order(order)
    print(f"[RESULT] Status: {result['status']}, ID: {result['broker_order_id']}")

    # Manually fill the order
    fill_price = 450.50
    print(f"\n[FILL] Simulating fill at ${fill_price}")
    fill = broker.simulate_fill(order, fill_price)

    print(f"[FILL] ID: {fill.fill_id}")
    print(f"[FILL] Price: ${fill.price:.2f}")
    print(f"[FILL] Commission: ${fill.commission:.2f}")
    print(f"[FILL] Slippage: {fill.slippage_bps:.2f} bps")

    # Check positions
    positions = await broker.get_positions()
    print(f"\n[POSITIONS] Count: {len(positions)}")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        print(f"    Unrealized P&L: ${pos['unrealized_pnl']:.2f}")

    # Check account
    account = await broker.get_account()
    print("\n[ACCOUNT]")
    print(f"  Cash: ${account['cash']:,.2f}")
    print(f"  Position Value: ${account['total_position_value']:,.2f}")
    print(f"  Total Equity: ${account['total_equity']:,.2f}")

    print("\n[OK] Manual fills test complete")


async def test_auto_fills_without_plugin():
    """Test paper broker without market data (should stay pending)."""
    print("\n" + "=" * 60)
    print("[TEST 2] Paper Broker - No Market Data Plugin")
    print("=" * 60)

    broker = PaperBrokerAdapter()

    order = Order(
        order_id="TEST-002",
        symbol="SPY",
        side="buy",
        quantity=5,
        order_type=OrderType.MARKET,
    )

    print(f"\n[ORDER] Submitting: {order.side.upper()} {order.quantity} {order.symbol}")
    result = await broker.submit_order(order)
    print(f"[RESULT] Status: {result['status']}")

    # Check pending orders
    pending = broker.get_pending_orders()
    print(f"\n[PENDING] Count: {len(pending)}")
    for p in pending:
        print(f"  {p['broker_order_id']}: {p['order'].symbol}")

    print("\n[OK] Order stays pending without market data plugin")


async def test_with_sample_data():
    """Test paper broker using sample data as price source."""
    print("\n" + "=" * 60)
    print("[TEST 3] Paper Broker - Sample Data Simulation")
    print("=" * 60)

    # Create a mock market data plugin
    class MockMarketDataPlugin:
        """Mock plugin that returns fixed prices."""

        async def get_quote(self, symbol: str):
            """Return mock quote data."""
            return {
                "symbol": symbol,
                "bid": 450.45,
                "ask": 450.55,
                "last": 450.50,
            }

    broker = PaperBrokerAdapter(market_data_plugin=MockMarketDataPlugin(), price_cache_seconds=0.5)

    # Buy order
    buy_order = Order(
        order_id="TEST-003",
        symbol="SPY",
        side="buy",
        quantity=20,
        order_type=OrderType.MARKET,
    )

    print(f"\n[ORDER] Submitting BUY: {buy_order.quantity} {buy_order.symbol}")
    result = await broker.submit_order(buy_order)
    print(f"[RESULT] Status: {result['status']}")

    if result["status"] == "filled":
        print(f"[FILL] Price: ${result['fill']['price']:.2f}")
        print(f"[FILL] Commission: ${result['fill']['commission']:.2f}")

    # Check positions
    positions = await broker.get_positions()
    print("\n[POSITIONS]")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")

    # Sell order
    sell_order = Order(
        order_id="TEST-004",
        symbol="SPY",
        side="sell",
        quantity=10,
        order_type=OrderType.MARKET,
    )

    print(f"\n[ORDER] Submitting SELL: {sell_order.quantity} {sell_order.symbol}")
    result = await broker.submit_order(sell_order)
    print(f"[RESULT] Status: {result['status']}")

    if result["status"] == "filled":
        print(f"[FILL] Price: ${result['fill']['price']:.2f}")

    # Final state
    positions = await broker.get_positions()
    account = await broker.get_account()

    print("\n[FINAL STATE]")
    print(f"  Positions: {len(positions)}")
    for pos in positions:
        print(f"    {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        print(f"      Unrealized P&L: ${pos['unrealized_pnl']:.2f}")

    print(f"  Cash: ${account['cash']:,.2f}")
    print(f"  Total Equity: ${account['total_equity']:,.2f}")

    # Get fills history
    fills = broker.get_fills()
    print(f"\n[FILLS HISTORY] Count: {len(fills)}")
    for fill in fills:
        print(f"  {fill.side.upper()} {fill.quantity} {fill.symbol} @ ${fill.price:.2f}")

    print("\n[OK] Sample data simulation complete")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PAPER BROKER TEST SUITE")
    print("=" * 60)

    await test_manual_fills()
    await test_auto_fills_without_plugin()
    await test_with_sample_data()

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
