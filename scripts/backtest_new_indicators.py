#!/usr/bin/env python3
"""
Backtest New Technical Indicator Strategies.

Tests the new ADX, Fibonacci, and Parabolic SAR strategies using ProofBench.
Generates comprehensive performance report comparing all strategies.

Usage:
    python scripts/backtest_new_indicators.py
    python scripts/backtest_new_indicators.py --data data/real_spy_daily.csv
    python scripts/backtest_new_indicators.py --capital 50000 --output results/
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.proofbench import ExecutionConfig, SimulationConfig, SimulationEngine
from engines.signalcore import ModelConfig
from engines.signalcore.models import (
    ADXTrendModel,
    FibonacciRetracementModel,
    ParabolicSARModel,
)


def create_sample_data():
    """Create realistic sample data if no file provided."""
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=500, freq="1D")

    # Create trending data with pullbacks
    trend = np.linspace(100, 150, 500)
    noise = np.random.randn(500) * 2
    pullbacks = np.sin(np.linspace(0, 20, 500)) * 5

    close = trend + noise + pullbacks
    high = close + np.abs(np.random.randn(500)) * 1.5
    low = close - np.abs(np.random.randn(500)) * 1.5
    volume = np.random.randint(1000000, 5000000, 500)

    data = pd.DataFrame(
        {
            "open": close - (np.random.randn(500) * 0.5),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return data


def strategy_adx_trend(engine, symbol, bar):
    """ADX trend filter strategy."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 60:
        return

    # Generate ADX signal
    model_id = "adx-trend"
    if model_id not in engine.signal_registry.list_models():
        config = ModelConfig(
            model_id=model_id,
            model_type="trend",
            version="1.0.0",
            parameters={
                "adx_period": 14,
                "adx_threshold": 25,
                "strong_trend": 40,
            },
        )
        engine.signal_registry.register(ADXTrendModel(config))

    model = engine.signal_registry.get(model_id)
    signal = model.generate(historical, bar.timestamp)

    # Only trade on strong trends
    if signal.signal_type.value == "entry":
        # Calculate position size
        equity = engine.portfolio.equity
        position_value = equity * 0.15
        quantity = int(position_value / bar.close)

        if quantity > 0:
            from engines.proofbench import Order, OrderSide, OrderType

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)


def strategy_fibonacci_retracement(engine, symbol, bar):
    """Fibonacci retracement strategy."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 70:
        return

    # Generate Fibonacci signal
    model_id = "fibonacci-retracement"
    if model_id not in engine.signal_registry.list_models():
        config = ModelConfig(
            model_id=model_id,
            model_type="static_level",
            version="1.0.0",
            parameters={
                "swing_lookback": 50,
                "key_levels": [0.382, 0.5, 0.618],
                "tolerance": 0.015,
            },
        )
        engine.signal_registry.register(FibonacciRetracementModel(config))

    model = engine.signal_registry.get(model_id)
    signal = model.generate(historical, bar.timestamp)

    if signal.signal_type.value == "entry":
        equity = engine.portfolio.equity
        position_value = equity * 0.12
        quantity = int(position_value / bar.close)

        if quantity > 0:
            from engines.proofbench import Order, OrderSide, OrderType

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)


def strategy_parabolic_sar(engine, symbol, bar):
    """Parabolic SAR trend following strategy."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 50:
        return

    # Generate Parabolic SAR signal
    model_id = "parabolic-sar"
    if model_id not in engine.signal_registry.list_models():
        config = ModelConfig(
            model_id=model_id,
            model_type="trend",
            version="1.0.0",
            parameters={
                "acceleration": 0.02,
                "maximum": 0.2,
                "min_trend_bars": 3,
            },
        )
        engine.signal_registry.register(ParabolicSARModel(config))

    model = engine.signal_registry.get(model_id)
    signal = model.generate(historical, bar.timestamp)

    positions = engine.portfolio.positions

    if signal.signal_type.value == "entry" and signal.metadata.get("reversal_detected"):
        # New trend detected
        equity = engine.portfolio.equity
        position_value = equity * 0.18
        quantity = int(position_value / bar.close)

        if quantity > 0:
            from engines.proofbench import Order, OrderSide, OrderType

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

    elif symbol in positions and signal.metadata.get("reversal_detected"):
        # Exit on reversal
        quantity = positions[symbol].quantity
        from engines.proofbench import Order, OrderSide, OrderType

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=bar.timestamp,
        )
        engine.submit_order(order)


def run_backtest(strategy_func, strategy_name, data, symbol, initial_capital):
    """Run a single backtest."""
    print(f"\n{'='*80}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*80}")

    # Configure execution
    exec_config = ExecutionConfig(
        estimated_spread=0.001,
        commission_pct=0.001,
        commission_per_trade=1.0,
    )

    sim_config = SimulationConfig(
        initial_capital=initial_capital,
        execution_config=exec_config,
    )

    # Create engine
    engine = SimulationEngine(config=sim_config)

    # Load data
    engine.load_data(symbol, data)
    engine.on_bar = strategy_func

    # Add signal registry for models
    from engines.signalcore.core.model import ModelRegistry

    engine.signal_registry = ModelRegistry()

    # Run backtest
    print(f"Running {strategy_name} backtest...")
    results = engine.run()

    # Print results
    metrics = results.metrics

    print(f"\n{strategy_name} Results:")
    print(f"  Total Return:       {metrics.total_return:>8.2%}")
    print(f"  Annualized Return:  {metrics.annualized_return:>8.2%}")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>8.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:>8.2f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:>8.2%}")
    print(f"  Win Rate:           {metrics.win_rate:>8.2%}")
    print(f"  Total Trades:       {len(results.trades):>8}")
    print(f"  Final Equity:       ${results.portfolio.equity:>8,.2f}")

    return {
        "strategy": strategy_name,
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "total_trades": len(results.trades),
        "final_equity": results.portfolio.equity,
    }


def main():  # noqa: PLR0915
    """Run backtests for all new indicator strategies."""
    parser = argparse.ArgumentParser(description="Backtest new technical indicator strategies")
    parser.add_argument("--data", default=None, help="Path to CSV file with OHLCV data")
    parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital (default: 100000)"
    )
    parser.add_argument("--output", default=None, help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NEW TECHNICAL INDICATORS BACKTEST SUITE")
    print("=" * 80)
    print("\nTesting Strategies:")
    print("  1. ADX Trend Filter")
    print("  2. Fibonacci Retracement")
    print("  3. Parabolic SAR Trend Following")
    print()

    # Load or create data
    if args.data:
        print(f"Loading data from {args.data}...")
        data = pd.read_csv(args.data)

        # Handle date column
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data.set_index("date", inplace=True)
        elif "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)

        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "SPY"
    else:
        print("No data file provided. Creating sample data...")
        data = create_sample_data()
        symbol = "TEST"

    print(f"Data: {len(data)} bars")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Initial Capital: ${args.capital:,.2f}")

    # Run backtests
    strategies = [
        (strategy_adx_trend, "ADX Trend Filter"),
        (strategy_fibonacci_retracement, "Fibonacci Retracement"),
        (strategy_parabolic_sar, "Parabolic SAR"),
    ]

    all_results = []

    for strategy_func, strategy_name in strategies:
        try:
            result = run_backtest(
                strategy_func,
                strategy_name,
                data.copy(),
                symbol,
                args.capital,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {strategy_name} failed: {e}")
            import traceback

            traceback.print_exc()

    # Generate comparison report
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + df.to_string(index=False))

        # Find best performers
        print("\n" + "=" * 80)
        print("BEST PERFORMERS")
        print("=" * 80)

        best_return = df.loc[df["total_return"].idxmax()]
        print(f"\nHighest Return: {best_return['strategy']}")
        print(f"  Return: {best_return['total_return']:.2%}")

        best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
        print(f"\nBest Risk-Adjusted: {best_sharpe['strategy']}")
        print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}")

        min_dd = df.loc[df["max_drawdown"].abs().idxmin()]
        print(f"\nLowest Drawdown: {min_dd['strategy']}")
        print(f"  Max DD: {min_dd['max_drawdown']:.2%}")

        # Save results if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"backtest_results_{timestamp}.csv"

            df.to_csv(output_file, index=False)
            print(f"\n[OK] Results saved to {output_file}")

    print("\n" + "=" * 80)
    print("[COMPLETE] Backtest suite finished")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
