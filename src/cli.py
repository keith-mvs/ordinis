"""
Command-Line Interface for Intelligent Investor.

Provides easy-to-use commands for running backtests, managing strategies,
and analyzing results.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

from engines.proofbench import SimulationConfig, SimulationEngine
from engines.proofbench.analytics import LLMPerformanceNarrator
from strategies import (
    MomentumBreakoutStrategy,
    MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy,
)


def load_market_data(file_path: str) -> tuple[str, pd.DataFrame]:
    """
    Load market data from CSV file.

    Args:
        file_path: Path to CSV file with OHLCV data

    Returns:
        (symbol, data) tuple
    """
    # Read CSV
    data = pd.read_csv(file_path)

    # Extract symbol if present, otherwise use filename
    if "symbol" in data.columns:
        symbol = data["symbol"].iloc[0]
    else:
        symbol = Path(file_path).stem.upper()

    # Set timestamp as index
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data.set_index("timestamp", inplace=True)
    elif "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)
    else:
        raise ValueError("Data must have 'timestamp' or 'date' column")

    # Validate required columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return symbol, data


def create_strategy(strategy_name: str, **params):
    """
    Create strategy instance by name.

    Args:
        strategy_name: Strategy name (rsi, ma, momentum)
        **params: Strategy parameters

    Returns:
        Strategy instance
    """
    strategies = {
        "rsi": RSIMeanReversionStrategy,
        "ma": MovingAverageCrossoverStrategy,
        "momentum": MomentumBreakoutStrategy,
    }

    if strategy_name.lower() not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {', '.join(strategies.keys())}"
        )

    strategy_class = strategies[strategy_name.lower()]
    return strategy_class(name=f"{strategy_name}-backtest", **params)  # type: ignore[abstract]


def run_backtest(args):  # noqa: PLR0915
    """Run backtest command."""
    print("\n" + "=" * 80)
    print("INTELLIGENT INVESTOR - BACKTEST")
    print("=" * 80)

    # Load data
    print(f"\n[1/5] Loading market data from {args.data}...")
    try:
        symbol, data = load_market_data(args.data)
        print(f"  Symbol: {symbol}")
        print(f"  Bars: {len(data)}")
        print(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return 1

    # Create strategy
    print(f"\n[2/5] Creating {args.strategy} strategy...")
    try:
        # Parse strategy parameters
        strategy_params = {}
        if args.params:
            for param in args.params:
                key, value = param.split("=")
                # Try to convert to number
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                strategy_params[key] = value

        strategy = create_strategy(args.strategy, **strategy_params)
        print(f"  Strategy: {strategy.name}")
        print(f"  Required bars: {strategy.get_required_bars()}")

        if len(data) < strategy.get_required_bars():
            print(f"[WARNING] Insufficient data: {len(data)} < {strategy.get_required_bars()}")
            print("  Proceeding anyway - may produce incomplete results")

    except Exception as e:
        print(f"[ERROR] Failed to create strategy: {e}")
        return 1

    # Setup simulation
    print("\n[3/5] Configuring backtest...")
    config = SimulationConfig(
        initial_capital=args.capital, bar_frequency="1d", risk_free_rate=args.risk_free
    )
    sim = SimulationEngine(config=config)
    sim.load_data(symbol, data)

    print(f"  Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  Risk-Free Rate: {config.risk_free_rate:.1%}")

    # Trading logic
    signals_generated = 0
    signals_executed = 0

    def trading_strategy(engine, sym, bar):
        nonlocal signals_generated, signals_executed

        # Need enough data
        current_bar = len(engine.portfolio.equity_curve)
        if current_bar < strategy.get_required_bars():
            return

        # Check signal frequency
        if current_bar % args.signal_frequency != 0:
            return

        # Get data up to current bar
        recent_data = data.iloc[:current_bar]

        # Generate signal
        signal = strategy.generate_signal(recent_data, bar.timestamp)
        if not signal:
            return

        signals_generated += 1

        # Execute entry signals
        if signal.signal_type.value == "entry" and signal.direction.value == "long":
            # Calculate position size
            position_size = int((engine.portfolio.cash * args.position_size) / bar.close)

            if position_size > 0:
                from engines.proofbench.core.execution import Order, OrderSide, OrderType

                order = Order(
                    symbol=sym,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    timestamp=bar.timestamp,
                )
                engine.pending_orders.append(order)
                signals_executed += 1

    sim.on_bar = trading_strategy

    # Run backtest
    print("\n[4/5] Running backtest...")
    print(f"  Processing {len(data)} bars...")

    try:
        results = sim.run()
        print(f"\n  Signals Generated: {signals_generated}")
        print(f"  Signals Executed: {signals_executed}")
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Analyze results
    print("\n[5/5] Analyzing results...")

    metrics = results.metrics
    print(f"\n{'=' * 80}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 80}")

    print("\nPerformance:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")

    print("\nRisk:")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Downside Deviation: {metrics.downside_deviation:.2%}")

    print("\nTrades:")
    print(f"  Total Trades: {metrics.num_trades}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Avg Win: ${metrics.avg_win:,.2f}")
    print(f"  Avg Loss: ${metrics.avg_loss:,.2f}")

    # AI narration if requested
    if args.ai and args.nvidia_key:
        print(f"\n{'=' * 80}")
        print("AI PERFORMANCE ANALYSIS")
        print(f"{'=' * 80}")

        narrator = LLMPerformanceNarrator(nvidia_api_key=args.nvidia_key)
        narration = narrator.narrate_results(results)

        print(f"\n{narration['narration']}")

        if args.suggestions:
            print(f"\n{'=' * 80}")
            print("OPTIMIZATION SUGGESTIONS")
            print(f"{'=' * 80}")

            suggestions = narrator.suggest_optimizations(results, focus=args.focus)
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"{i}. {suggestion}")

    # Save results if requested
    if args.output:
        print(f"\n[OK] Saving results to {args.output}...")
        results_df = pd.DataFrame(
            {
                "metric": [
                    "total_return",
                    "annualized_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "num_trades",
                ],
                "value": [
                    metrics.total_return,
                    metrics.annualized_return,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.win_rate,
                    metrics.num_trades,
                ],
            }
        )
        results_df.to_csv(args.output, index=False)

    print(f"\n{'=' * 80}")
    print("BACKTEST COMPLETE")
    print(f"{'=' * 80}\n")

    return 0


def list_strategies(args):
    """List available strategies."""
    print("\nAvailable Strategies:\n")

    strategies = [
        ("rsi", "RSI Mean Reversion", "Counter-trend using RSI indicator"),
        ("ma", "Moving Average Crossover", "Trend following with MA signals"),
        ("momentum", "Momentum Breakout", "Volatility breakouts with confirmation"),
    ]

    for name, title, desc in strategies:
        print(f"  {name:12} - {title}")
        print(f"               {desc}\n")

    print("Use 'intelligent-investor backtest --help' for usage details.\n")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Investor - AI-Powered Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run RSI backtest
  intelligent-investor backtest --data data.csv --strategy rsi

  # Run MA crossover with custom parameters
  intelligent-investor backtest --data data.csv --strategy ma --params fast_period=50 slow_period=200

  # Run with AI analysis
  intelligent-investor backtest --data data.csv --strategy momentum --ai --nvidia-key nvapi-...

  # Save results to file
  intelligent-investor backtest --data data.csv --strategy rsi --output results.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run strategy backtest")
    backtest_parser.add_argument(
        "--data", required=True, help="Path to CSV file with market data (OHLCV)"
    )
    backtest_parser.add_argument(
        "--strategy",
        required=True,
        choices=["rsi", "ma", "momentum"],
        help="Strategy to backtest",
    )
    backtest_parser.add_argument(
        "--params",
        nargs="*",
        help="Strategy parameters (e.g., rsi_period=14 oversold_threshold=30)",
    )
    backtest_parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    backtest_parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size as fraction of capital (default: 0.1)",
    )
    backtest_parser.add_argument(
        "--signal-frequency",
        type=int,
        default=1,
        help="Check for signals every N bars (default: 1)",
    )
    backtest_parser.add_argument(
        "--risk-free",
        type=float,
        default=0.02,
        help="Risk-free rate for Sharpe calculation (default: 0.02)",
    )
    backtest_parser.add_argument("--ai", action="store_true", help="Enable AI-powered analysis")
    backtest_parser.add_argument("--nvidia-key", help="NVIDIA API key for AI features")
    backtest_parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Generate optimization suggestions (requires --ai)",
    )
    backtest_parser.add_argument(
        "--focus",
        choices=["returns", "risk", "consistency", "general"],
        default="general",
        help="Focus area for optimization suggestions",
    )
    backtest_parser.add_argument("--output", help="Save results to CSV file")

    # List strategies command
    list_parser = subparsers.add_parser("list", help="List available strategies")

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "backtest":
        return run_backtest(args)
    if args.command == "list":
        return list_strategies(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
