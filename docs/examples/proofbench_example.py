"""
Example usage of ProofBench backtesting engine.

This example demonstrates how to:
1. Load historical data
2. Define a trading strategy
3. Run a backtest
4. Analyze results
"""

import numpy as np
import pandas as pd

from ordinis.engines.proofbench import (
    ExecutionConfig,
    Order,
    OrderSide,
    OrderType,
    SimulationConfig,
    SimulationEngine,
)


def load_sample_data():
    """Create sample OHLCV data for demonstration."""
    # Generate 252 trading days (1 year)
    dates = pd.date_range("2024-01-01", periods=252, freq="1d")

    # Create realistic price movement with trend + noise
    trend = np.linspace(100, 120, 252)
    noise = np.random.normal(0, 2, 252)
    close_prices = trend + noise

    data = pd.DataFrame(
        {
            "open": close_prices * 0.99,  # Slightly below close
            "high": close_prices * 1.01,  # Slightly above close
            "low": close_prices * 0.98,  # Slightly below open
            "close": close_prices,
            "volume": np.random.randint(1000000, 5000000, 252),
        },
        index=dates,
    )

    return data


def simple_momentum_strategy(engine, symbol, bar):
    """Simple momentum strategy: buy on strength, sell on weakness.

    This strategy:
    - Buys when price crosses above 20-day moving average
    - Sells when price crosses below 20-day moving average
    - Only holds one position at a time
    """
    # Get current position
    pos = engine.get_position(symbol)

    # Calculate simple moving average
    # In a real strategy, you'd maintain this externally
    # For demo purposes, we'll use a simple rule
    if not hasattr(engine, "price_history"):
        engine.price_history = []

    engine.price_history.append(bar.close)

    if len(engine.price_history) < 20:
        return  # Not enough data yet

    # Calculate 20-period SMA
    sma_20 = np.mean(engine.price_history[-20:])

    # Entry logic
    if bar.close > sma_20 and (pos is None or pos.quantity == 0):
        # Buy signal: price above SMA and no position
        # Use 10% of portfolio equity
        cash_to_invest = engine.get_equity() * 0.1
        quantity = int(cash_to_invest / bar.close)

        if quantity > 0 and engine.portfolio.can_trade(symbol, quantity, bar.close, OrderSide.BUY):
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)

    # Exit logic
    elif bar.close < sma_20 and pos and pos.quantity > 0:
        # Sell signal: price below SMA and we have a position
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
        )
        engine.submit_order(order)


def buy_and_hold_strategy(engine, symbol, bar):
    """Simple buy-and-hold strategy.

    Buy on first bar and hold until end.
    """
    if not hasattr(engine, "bought"):
        engine.bought = False

    if not engine.bought:
        # Invest 90% of capital
        cash_to_invest = engine.get_equity() * 0.9
        quantity = int(cash_to_invest / bar.close)

        if quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            engine.bought = True


def main():
    """Run backtest examples."""
    print("ProofBench Backtesting Example")
    print("=" * 50)

    # Load data
    print("\n1. Loading sample data...")
    data = load_sample_data()
    print(f"   Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Configure simulation
    print("\n2. Configuring simulation...")
    config = SimulationConfig(
        initial_capital=100000.0,
        execution_config=ExecutionConfig(
            estimated_spread=0.001,  # 0.1% spread
            commission_per_share=0.01,  # $0.01 per share
            commission_pct=0.001,  # 0.1% of trade value
        ),
        bar_frequency="1d",
        risk_free_rate=0.02,  # 2% annual risk-free rate
    )
    print(f"   Initial capital: ${config.initial_capital:,.2f}")

    # Test 1: Momentum Strategy
    print("\n3. Running momentum strategy backtest...")
    engine = SimulationEngine(config)
    engine.load_data("AAPL", data)
    engine.set_strategy(simple_momentum_strategy)

    results = engine.run()

    print("\n   === Momentum Strategy Results ===")
    print(f"   Final equity: ${results.metrics.equity_final:,.2f}")
    print(f"   Total return: {results.metrics.total_return:.2f}%")
    print(f"   Annualized return: {results.metrics.annualized_return:.2f}%")
    print(f"   Sharpe ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"   Max drawdown: {results.metrics.max_drawdown:.2f}%")
    print(f"   Number of trades: {results.metrics.num_trades}")
    print(f"   Win rate: {results.metrics.win_rate:.2f}%")
    print(f"   Profit factor: {results.metrics.profit_factor:.2f}")

    # Test 2: Buy-and-Hold Strategy
    print("\n4. Running buy-and-hold strategy backtest...")
    engine2 = SimulationEngine(config)
    engine2.load_data("AAPL", data)
    engine2.set_strategy(buy_and_hold_strategy)

    results2 = engine2.run()

    print("\n   === Buy-and-Hold Results ===")
    print(f"   Final equity: ${results2.metrics.equity_final:,.2f}")
    print(f"   Total return: {results2.metrics.total_return:.2f}%")
    print(f"   Annualized return: {results2.metrics.annualized_return:.2f}%")
    print(f"   Sharpe ratio: {results2.metrics.sharpe_ratio:.2f}")
    print(f"   Max drawdown: {results2.metrics.max_drawdown:.2f}%")
    print(f"   Number of trades: {results2.metrics.num_trades}")

    # Compare strategies
    print("\n5. Strategy Comparison:")
    print(f"   Momentum return: {results.metrics.total_return:.2f}%")
    print(f"   Buy-hold return: {results2.metrics.total_return:.2f}%")
    print(f"   Momentum Sharpe: {results.metrics.sharpe_ratio:.2f}")
    print(f"   Buy-hold Sharpe: {results2.metrics.sharpe_ratio:.2f}")

    # Show trade history
    if not results.trades.empty:
        print("\n6. Recent trades (momentum strategy):")
        print(results.trades.head())

    print("\n" + "=" * 50)
    print("Backtest complete!")


if __name__ == "__main__":
    main()
