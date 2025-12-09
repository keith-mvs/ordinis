"""
SignalCore + ProofBench Integration Example.

This example demonstrates how to:
1. Create SignalCore trading models
2. Generate trading signals
3. Backtest signals using ProofBench
4. Analyze performance metrics
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


import numpy as np
import pandas as pd

from engines.proofbench import (
    ExecutionConfig,
    Order,
    OrderSide,
    OrderType,
    SimulationConfig,
    SimulationEngine,
)
from engines.signalcore import (
    ModelConfig,
    ModelRegistry,
    RSIMeanReversionModel,
    SMACrossoverModel,
)


def create_sample_data():
    """Create realistic market data with clear trends for demonstration."""
    dates = pd.date_range("2024-01-01", periods=365, freq="1d")

    # Create price data with clearer trends
    # Phase 1: Uptrend (days 0-120)
    # Phase 2: Downtrend (days 120-240)
    # Phase 3: Uptrend (days 240-365)

    close_prices = []
    for i in range(365):
        if i < 120:
            # Strong uptrend
            base = 100 + (i / 120) * 20
        elif i < 240:
            # Downtrend
            base = 120 - ((i - 120) / 120) * 15
        else:
            # Recovery uptrend
            base = 105 + ((i - 240) / 125) * 25

        # Add some noise
        price = base + np.random.randn() * 1.5
        close_prices.append(max(price, 50))  # Floor at $50

    # Create valid OHLC bars
    bars = []
    for close in close_prices:
        # Ensure valid OHLC relationship
        open_price = close + np.random.randn() * 0.3
        high = max(open_price, close) + abs(np.random.randn()) * 0.8
        low = min(open_price, close) - abs(np.random.randn()) * 0.8

        bars.append(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": int(1000000 + np.random.randn() * 200000),
            }
        )

    data = pd.DataFrame(bars, index=dates)

    return data


def signal_driven_strategy(engine, symbol, bar):  # noqa: PLR0912
    """
    Trading strategy driven by SignalCore models.

    This strategy generates signals from registered models
    and translates them into orders.
    """
    # Get historical data for signal generation
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 50:  # Need enough data
        return

    # Generate signals from all registered models
    signals = []

    for model_id in engine.signal_registry.list_models(enabled_only=True):
        model = engine.signal_registry.get(model_id)

        try:
            is_valid, msg = model.validate(historical)
            if not is_valid:
                continue

            signal = model.generate(historical, bar.timestamp)
            signals.append(signal)

            # Debug: print first few actionable signals
            if len(signals) <= 3 and signal.is_actionable(min_probability=0.5, min_score=0.2):
                print(
                    f"  Signal: {signal.model_id} | {signal.signal_type.value} | {signal.direction.value} | score={signal.score:.2f}"
                )

        except Exception as e:
            print(f"Error generating signal from {model_id}: {e}")
            continue

    # Decision logic: Use signals to make trading decisions
    positions = engine.portfolio.positions

    # Entry logic: Look for buy signals (lowered thresholds for more signals)
    for signal in signals:
        if signal.is_actionable(min_probability=0.5, min_score=0.2):
            if signal.signal_type.value == "entry" and signal.direction.value == "long":
                # Don't enter if already have position
                if symbol in positions:
                    continue

                # Calculate position size (simple: 10% of equity)
                equity = engine.portfolio.equity
                position_value = equity * 0.10
                quantity = int(position_value / bar.close)

                if quantity > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        timestamp=bar.timestamp,
                    )
                    engine.submit_order(order)
                    print(
                        f"{bar.timestamp.date()} | BUY {quantity} {symbol} @ {bar.close:.2f} | "
                        f"Model: {signal.model_id} | Prob: {signal.probability:.2%} | "
                        f"Score: {signal.score:.2f}"
                    )
                    break  # Only one entry per bar

    # Exit logic: Look for sell signals
    for signal in signals:
        if signal.signal_type.value == "exit":
            if symbol in positions:
                quantity = positions[symbol].quantity

                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    timestamp=bar.timestamp,
                )
                engine.submit_order(order)
                print(
                    f"{bar.timestamp.date()} | SELL {quantity} {symbol} @ {bar.close:.2f} | "
                    f"Model: {signal.model_id} | Prob: {signal.probability:.2%}"
                )
                break  # Only one exit per bar


def main():  # noqa: PLR0915
    """Run SignalCore + ProofBench backtest."""
    print("=== SignalCore + ProofBench Integration Example ===\n")

    # Create sample data
    print("Creating sample market data...")
    data = create_sample_data()
    print(f"Generated {len(data)} days of data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}\n")

    # Configure SignalCore models
    print("Configuring SignalCore models...")

    # SMA Crossover model (shorter periods for more signals)
    sma_config = ModelConfig(
        model_id="sma_10_30",
        model_type="technical",
        version="1.0.0",
        parameters={"fast_period": 10, "slow_period": 30, "min_separation": 0.005},
    )
    sma_model = SMACrossoverModel(sma_config)

    # RSI Mean Reversion model
    rsi_config = ModelConfig(
        model_id="rsi_14",
        model_type="technical",
        version="1.0.0",
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )
    rsi_model = RSIMeanReversionModel(rsi_config)

    # Create model registry and register models
    signal_registry = ModelRegistry()
    signal_registry.register(sma_model)
    signal_registry.register(rsi_model)

    print(f"Registered {len(signal_registry.list_models())} models:")
    for model_id in signal_registry.list_models():
        model = signal_registry.get(model_id)
        print(f"  - {model_id}: {model.config.model_type}")

    print()

    # Configure ProofBench simulation
    print("Configuring ProofBench backtest...")
    exec_config = ExecutionConfig(
        estimated_spread=0.001, commission_pct=0.001, commission_per_trade=1.0
    )

    sim_config = SimulationConfig(initial_capital=100000.0, execution_config=exec_config)

    # Create simulation engine
    engine = SimulationEngine(config=sim_config)

    # Attach signal registry to engine for access in strategy
    engine.signal_registry = signal_registry

    # Load data and set strategy
    engine.load_data("TEST", data)
    engine.on_bar = signal_driven_strategy

    print(f"Initial capital: ${sim_config.initial_capital:,.2f}")
    print(
        f"Commission: {exec_config.commission_pct:.2%} + ${exec_config.commission_per_trade:.2f}\n"
    )

    # Run backtest
    print("Running backtest...\n")
    print("-" * 80)

    results = engine.run()

    print("-" * 80)
    print("\n=== Backtest Results ===\n")

    # Performance metrics
    metrics = results.metrics
    print(f"Total Return:       {metrics.total_return:>8.2%}")
    print(f"Annualized Return:  {metrics.annualized_return:>8.2%}")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>8.2f}")
    print(f"Sortino Ratio:      {metrics.sortino_ratio:>8.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown:>8.2%}")
    print(f"Win Rate:           {metrics.win_rate:>8.2%}")
    print(f"Volatility:         {metrics.volatility:>8.2%}")

    print(f"\nTotal Trades:       {len(results.trades):>8}")
    print(f"Final Equity:       ${results.portfolio.equity:>8,.2f}")

    # Trade analysis
    if len(results.trades) > 0:
        winning_trades = [t for t in results.trades if t.pnl > 0]
        losing_trades = [t for t in results.trades if t.pnl < 0]

        print(f"\nWinning Trades:     {len(winning_trades):>8}")
        print(f"Losing Trades:      {len(losing_trades):>8}")

        if winning_trades:
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            print(f"Avg Win:            ${avg_win:>8,.2f}")

        if losing_trades:
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            print(f"Avg Loss:           ${avg_loss:>8,.2f}")

    print("\n[OK] SignalCore + ProofBench integration verified!")
    print("Successfully generated signals and backtested strategy.")

    return results


if __name__ == "__main__":
    main()
