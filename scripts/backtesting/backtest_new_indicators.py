#!/usr/bin/env python3
"""
Backtest New Technical Indicator Strategies - Version 2.

Improved version with proper position management and risk controls.

Usage:
    python scripts/backtest_new_indicators_v2.py
    python scripts/backtest_new_indicators_v2.py --data data/real_spy_daily.csv
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.proofbench import (
    ExecutionConfig,
    Order,
    OrderSide,
    OrderType,
    SimulationConfig,
    SimulationEngine,
)
from engines.signalcore import ModelConfig
from engines.signalcore.core.model import ModelRegistry
from engines.signalcore.models import ADXTrendModel, FibonacciRetracementModel, ParabolicSARModel


class StrategyState:
    """Track strategy state across bars."""

    def __init__(self):
        self.positions = {}  # symbol -> {'quantity', 'entry_price', 'entry_bar'}
        self.last_signal = {}  # symbol -> signal_type


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


def strategy_adx_trend_v2(engine, symbol, bar):
    """ADX trend filter strategy with proper risk management."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 60:
        return

    # Initialize state if needed
    if not hasattr(engine, "strategy_state"):
        engine.strategy_state = StrategyState()

    state = engine.strategy_state

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

    # Check if we have a position
    has_position = symbol in state.positions

    # ENTRY logic
    if signal.signal_type.value == "entry" and not has_position:
        # Calculate position size with limits
        equity = engine.portfolio.equity
        cash = engine.portfolio.cash

        # Maximum 10% of equity per position
        max_position_value = equity * 0.10
        # Use available cash limit
        position_value = min(max_position_value, cash * 0.9)  # Keep 10% cash buffer

        quantity = int(position_value / bar.close)

        if quantity > 0 and position_value < cash:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

            # Track position
            state.positions[symbol] = {
                "quantity": quantity,
                "entry_price": bar.close,
                "entry_bar": len(historical),
                "direction": signal.direction.value,
            }

    # EXIT logic
    elif signal.signal_type.value == "exit" and has_position:
        position = state.positions[symbol]
        quantity = position["quantity"]

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL if position["direction"] == "long" else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=bar.timestamp,
        )
        engine.submit_order(order)

        # Remove position
        del state.positions[symbol]

    # Store last signal
    state.last_signal[symbol] = signal.signal_type.value


def strategy_fibonacci_retracement_v2(engine, symbol, bar):
    """Fibonacci retracement strategy with proper risk management."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 70:
        return

    # Initialize state if needed
    if not hasattr(engine, "strategy_state"):
        engine.strategy_state = StrategyState()

    state = engine.strategy_state

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
                "tolerance": 0.01,  # Reduced from 0.015 to 0.01 (1%)
            },
        )
        engine.signal_registry.register(FibonacciRetracementModel(config))

    model = engine.signal_registry.get(model_id)
    signal = model.generate(historical, bar.timestamp)

    # Check if we have a position
    has_position = symbol in state.positions

    # ENTRY logic
    if signal.signal_type.value == "entry" and not has_position:
        # Calculate position size with limits
        equity = engine.portfolio.equity
        cash = engine.portfolio.cash

        # Maximum 8% of equity per position (more conservative)
        max_position_value = equity * 0.08
        position_value = min(max_position_value, cash * 0.9)

        quantity = int(position_value / bar.close)

        if quantity > 0 and position_value < cash:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

            # Track position
            state.positions[symbol] = {
                "quantity": quantity,
                "entry_price": bar.close,
                "entry_bar": len(historical),
                "direction": signal.direction.value,
            }

    # Simple profit target exit (15% gain)
    elif has_position:
        position = state.positions[symbol]
        entry_price = position["entry_price"]
        current_price = bar.close

        profit_pct = (current_price - entry_price) / entry_price

        # Exit if 15% profit or -5% stop loss
        if (position["direction"] == "long" and (profit_pct >= 0.15 or profit_pct <= -0.05)) or (
            position["direction"] == "short" and (profit_pct <= -0.15 or profit_pct >= 0.05)
        ):
            quantity = position["quantity"]

            order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position["direction"] == "long" else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

            # Remove position
            del state.positions[symbol]


def strategy_parabolic_sar_v2(engine, symbol, bar):
    """Parabolic SAR trend following strategy with proper risk management."""
    data = engine.data[symbol]
    historical = data.loc[: bar.timestamp]

    if len(historical) < 50:
        return

    # Initialize state if needed
    if not hasattr(engine, "strategy_state"):
        engine.strategy_state = StrategyState()

    state = engine.strategy_state

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
                "min_trend_bars": 2,  # Reduced from 3 to 2 for more signals
            },
        )
        engine.signal_registry.register(ParabolicSARModel(config))

    model = engine.signal_registry.get(model_id)
    signal = model.generate(historical, bar.timestamp)

    # Check if we have a position
    has_position = symbol in state.positions

    # ENTRY logic - only on reversals
    if (
        signal.signal_type.value == "entry"
        and not has_position
        and signal.metadata.get("reversal_detected")
    ):
        # Calculate position size with limits
        equity = engine.portfolio.equity
        cash = engine.portfolio.cash

        # Maximum 12% of equity per position
        max_position_value = equity * 0.12
        position_value = min(max_position_value, cash * 0.9)

        quantity = int(position_value / bar.close)

        if quantity > 0 and position_value < cash:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

            # Track position
            state.positions[symbol] = {
                "quantity": quantity,
                "entry_price": bar.close,
                "entry_bar": len(historical),
                "direction": signal.direction.value,
                "sar_level": signal.metadata.get("current_sar"),
            }

    # EXIT logic - on SAR reversal or stop loss
    elif has_position:
        position = state.positions[symbol]

        # Check for reversal
        if signal.metadata.get("reversal_detected"):
            # SAR reversed - exit position
            quantity = position["quantity"]

            order = Order(
                symbol=symbol,
                side=OrderSide.SELL if position["direction"] == "long" else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=bar.timestamp,
            )
            engine.submit_order(order)

            # Remove position
            del state.positions[symbol]


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
    engine.signal_registry = ModelRegistry()

    # Run backtest
    print(f"Running {strategy_name} backtest...")
    results = engine.run()

    # Print results
    metrics = results.metrics

    print(f"\n{strategy_name} Results:")
    print(f"  Total Return:       {metrics.total_return:>7.2f}%")
    print(f"  Annualized Return:  {metrics.annualized_return:>7.2f}%")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>8.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:>8.2f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:>7.2f}%")
    print(f"  Win Rate:           {metrics.win_rate:>7.2f}%")
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
    parser = argparse.ArgumentParser(
        description="Backtest new technical indicator strategies (v2 with risk management)"
    )
    parser.add_argument("--data", default=None, help="Path to CSV file with OHLCV data")
    parser.add_argument(
        "--capital", type=float, default=100000, help="Initial capital (default: 100000)"
    )
    parser.add_argument("--output", default=None, help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NEW TECHNICAL INDICATORS BACKTEST SUITE - V2")
    print("=" * 80)
    print("\nImprovements:")
    print("  - Position tracking to prevent over-leveraging")
    print("  - 10% max position size limit")
    print("  - Stop loss and profit targets")
    print("  - Cash buffer (90% usage max)")
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
        (strategy_adx_trend_v2, "ADX Trend Filter V2"),
        (strategy_fibonacci_retracement_v2, "Fibonacci Retracement V2"),
        (strategy_parabolic_sar_v2, "Parabolic SAR V2"),
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

        if len(df) > 0:
            best_return = df.loc[df["total_return"].idxmax()]
            print(f"\nHighest Return: {best_return['strategy']}")
            print(f"  Return: {best_return['total_return']:.2f}%")
            print(f"  Trades: {best_return['total_trades']:.0f}")

            if df["sharpe_ratio"].max() > 0:
                best_sharpe = df.loc[df["sharpe_ratio"].idxmax()]
                print(f"\nBest Risk-Adjusted: {best_sharpe['strategy']}")
                print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}")

            min_dd = df.loc[df["max_drawdown"].abs().idxmin()]
            print(f"\nLowest Drawdown: {min_dd['strategy']}")
            print(f"  Max DD: {min_dd['max_drawdown']:.2f}%")

        # Save results if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"backtest_results_v2_{timestamp}.csv"

            df.to_csv(output_file, index=False)
            print(f"\n[OK] Results saved to {output_file}")

    print("\n" + "=" * 80)
    print("[COMPLETE] Backtest suite finished")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
