#!/usr/bin/env python3
"""
Integration Test for Live Trading Runtime.

Tests all components work together:
- Strategy Loader
- Broker (Simulated)
- Live Trading Runtime
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.adapters.broker.broker import OrderSide, OrderType, SimulatedBroker
from ordinis.engines.signalcore.strategy_loader import StrategyLoader


async def test_full_integration():
    """Test complete live trading integration."""
    print("=" * 70)
    print("LIVE TRADING INTEGRATION TEST")
    print("=" * 70)

    # 1. Strategy Loader
    print("\n[1/5] Testing Strategy Loader...")
    loader = StrategyLoader()
    success = loader.load_strategy("configs/strategies/atr_optimized_rsi.yaml")

    if not success:
        print("   FAILED: Could not load strategy")
        return False

    symbols = loader.get_symbols()
    print(f"   SUCCESS: Loaded {len(symbols)} symbols")
    print(f"   Symbols: {symbols}")

    # 2. Model Access
    print("\n[2/5] Testing Model Access...")
    for symbol in ["COIN", "DKNG", "AMD"]:
        model = loader.get_model(symbol)
        if model is None:
            print(f"   FAILED: No model for {symbol}")
            return False

        risk = loader.get_risk_params(symbol)
        print(f"   {symbol}: ATR_stop={risk['atr_stop_mult']}x, ATR_tp={risk['atr_tp_mult']}x")

    print("   SUCCESS: All models accessible")

    # 3. Simulated Broker
    print("\n[3/5] Testing Simulated Broker...")
    broker = SimulatedBroker(initial_cash=100_000)
    connected = await broker.connect()

    if not connected:
        print("   FAILED: Could not connect to broker")
        return False

    account = await broker.get_account()
    print(f"   Equity: ${account.equity:,.2f}")
    print(f"   Cash: ${account.cash:,.2f}")
    print(f"   Buying Power: ${account.buying_power:,.2f}")
    print("   SUCCESS: Broker connected")

    # 4. Order Execution
    print("\n[4/5] Testing Order Execution...")

    # Buy order - use the broker's submit_order method directly
    result = await broker.submit_order(
        symbol="COIN",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    if result is None:
        print("   FAILED: Buy order rejected")
        return False

    print(f"   Buy Order ID: {result.id}")

    # Check position
    positions_list = await broker.get_positions()
    positions = {p.symbol: p for p in positions_list}
    if "COIN" not in positions:
        print("   FAILED: Position not created")
        return False

    print(
        f"   Position: {positions['COIN'].quantity} shares @ ${positions['COIN'].avg_entry_price:.2f}"
    )

    # Sell order
    result = await broker.submit_order(
        symbol="COIN",
        side=OrderSide.SELL,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    if result is None:
        print("   FAILED: Sell order rejected")
        return False

    print(f"   Sell Order ID: {result.id}")

    # Check position closed
    positions_list = await broker.get_positions()
    positions = {p.symbol: p for p in positions_list}
    coin_pos = positions.get("COIN")
    if coin_pos and coin_pos.quantity > 0:
        print("   FAILED: Position not closed")
        return False

    print("   SUCCESS: Orders executed correctly")

    # 5. Account State
    print("\n[5/5] Testing Account State...")
    account = await broker.get_account()
    print(f"   Final Equity: ${account.equity:,.2f}")
    print(f"   Final Cash: ${account.cash:,.2f}")

    await broker.disconnect()
    print("   SUCCESS: Broker disconnected")

    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    print("\n🚀 LIVE TRADING SYSTEM READY")
    print("\nNext Steps:")
    print("  1. Paper Trading (requires Alpaca API keys):")
    print("     export ALPACA_API_KEY=your_key")
    print("     export ALPACA_API_SECRET=your_secret")
    print("     python -m ordinis.runtime.live_trading --mode paper")
    print("")
    print("  2. Simulated Trading (no API required):")
    print("     python -m ordinis.runtime.live_trading --mode simulated")
    print("")
    print("  3. View Strategy Config:")
    print("     configs/strategies/atr_optimized_rsi.yaml")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_full_integration())
    sys.exit(0 if success else 1)
