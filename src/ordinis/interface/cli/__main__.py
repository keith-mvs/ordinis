"""
Command-Line Interface for Intelligent Investor.

Provides easy-to-use commands for running backtests, managing strategies,
and analyzing results.
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

from ordinis.analysis.technical import (
    BreakoutDetector,
    CandlestickPatterns,
    CompositeIndicator,
    MultiTimeframeAnalyzer,
    SupportResistanceLocator,
    TechnicalIndicators,
)
from ordinis.application.strategies import (
    MomentumBreakoutStrategy,
    MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy,
)
from ordinis.engines.proofbench import SimulationConfig, SimulationEngine
from ordinis.engines.proofbench.analytics import LLMPerformanceNarrator
from ordinis.visualization.indicators import IndicatorChart

_logger = logging.getLogger(__name__)


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


def run_backtest(args):
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
                from ordinis.engines.proofbench.core.execution import Order, OrderSide, OrderType

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


def analyze_market(args):
    """Run technical analysis (Phase 3 indicators/patterns)."""
    print("\n" + "=" * 80)
    print("INTELLIGENT INVESTOR - TECHNICAL ANALYSIS")
    print("=" * 80)

    # Load data
    print(f"\n[1/4] Loading market data from {args.data}...")
    try:
        symbol, data = load_market_data(args.data)
        print(f"  Symbol: {symbol}")
        print(f"  Bars: {len(data)}")
        print(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:  # pragma: no cover - CLI guard rail
        print(f"[ERROR] Failed to load data: {e}")
        return 1

    # Full technical snapshot (includes Ichimoku Cloud)
    print("\n[2/4] Running core indicators (Ichimoku, MA/vol/osc)...")
    tech = TechnicalIndicators()
    snapshot = tech.analyze(data)
    print(f"  Trend: {snapshot.trend_direction} | Bias: {snapshot.overall_bias}")
    print(
        f"  Ichimoku: {snapshot.ichimoku.trend} | Position: {snapshot.ichimoku.position} | "
        f"Baseline cross: {snapshot.ichimoku.baseline_cross}"
    )
    print(f"  Signals: {', '.join(snapshot.signals) if snapshot.signals else 'None'}")

    # Patterns and breakouts
    print("\n[3/4] Detecting patterns and breakouts...")
    patterns = CandlestickPatterns().detect(data.tail(120))
    if isinstance(patterns, dict):
        active_patterns = [name for name, matched in patterns.items() if matched]
    else:
        active_patterns = [p for p in patterns if p]
    sr = SupportResistanceLocator().find_levels(
        high=data["high"], low=data["low"], window=3, tolerance=0.003
    )
    breakout_signal = BreakoutDetector.detect(
        close=data["close"],
        support=sr.support,
        resistance=sr.resistance,
        tolerance=0.002,
    )
    print(f"  Candlestick patterns: {', '.join(active_patterns) if active_patterns else 'None'}")
    brk_text = breakout_signal.direction or "none"
    level_text = f"{breakout_signal.level:.2f}" if breakout_signal.level else "n/a"
    print(f"  Breakout: {brk_text} | level: {level_text} | confirmed: {breakout_signal.confirmed}")

    # Multi-timeframe alignment + composite score
    print("\n[4/4] Multi-timeframe alignment + composite score...")
    resampled = {
        "1d": data,
        "1w": data.resample("1W")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(),
        "1m": data.resample("1M")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(),
    }
    resampled = {k: v for k, v in resampled.items() if len(v) > 10}

    if resampled:
        mtf = MultiTimeframeAnalyzer()
        mtf_result = mtf.analyze(resampled)
        print(
            f"  MTF majority: {mtf_result.majority_trend} | bias: {mtf_result.bias} "
            f"| agreement: {mtf_result.agreement_score:.2f}"
        )
        for sig in mtf_result.signals:
            print(f"    {sig.timeframe}: {sig.trend_direction} (strength={sig.trend_strength:.2f})")
    else:
        print("  Not enough data to compute multi-timeframe alignment.")

    bias_score = (
        1
        if snapshot.overall_bias in {"buy", "strong_buy"}
        else -1
        if snapshot.overall_bias in {"sell", "strong_sell"}
        else 0
    )
    ichimoku_score = (
        1
        if snapshot.ichimoku.trend == "bullish"
        else -1
        if snapshot.ichimoku.trend == "bearish"
        else 0
    )
    comp = CompositeIndicator.weighted_sum(
        {"bias": bias_score, "ichimoku": ichimoku_score}, {"bias": 0.6, "ichimoku": 0.4}
    )
    # Map score to simple signal
    comp_signal = "buy" if comp.value > 0 else "sell" if comp.value < 0 else "neutral"
    print(f"\nComposite Score: {comp.value:.2f} | Signal: {comp_signal}")

    print(f"\n{'=' * 80}")
    print("TECHNICAL ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")

    # Optional: save Ichimoku plot
    if args.save_ichimoku:
        try:
            fig = IndicatorChart.plot_ichimoku_cloud(data, title=f"{symbol} Ichimoku")
            fig.write_html(args.save_ichimoku)
            print(f"[saved] Ichimoku plot -> {args.save_ichimoku}")
        except Exception as plot_err:  # pragma: no cover - best-effort for CLI convenience
            print(f"[warning] Failed to save Ichimoku plot: {plot_err}")

    return 0


def main():
    """Main CLI entry point."""
    # Initialize tracing
    tracing_enabled = setup_tracing(
        TracingConfig(
            service_name="ordinis-cli",
            enabled=True,  # Can be controlled via environment variable
        )
    )
    if tracing_enabled:
        _logger.info("Distributed tracing enabled - view traces in AI Toolkit")

    try:
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

        # Technical analysis command
        analyze_parser = subparsers.add_parser(
            "analyze", help="Run Phase 3 technical analysis on OHLCV CSV"
        )
        analyze_parser.add_argument(
            "--data",
            required=True,
            help="Path to CSV file with OHLCV data (requires timestamp/date index column)",
        )
        analyze_parser.add_argument(
            "--breakout-lookback",
            type=int,
            default=20,
            help="Lookback window for breakout detection (default: 20)",
        )
        analyze_parser.add_argument(
            "--breakout-volume-mult",
            type=float,
            default=1.5,
            help="Volume multiplier for breakout confirmation (default: 1.5x)",
        )
        analyze_parser.add_argument(
            "--save-ichimoku",
            help="Optional path to save Ichimoku cloud HTML plot",
        )

        # Parse arguments
        args = parser.parse_args()

        # Execute command
        result = 0
        if args.command == "backtest":
            result = run_backtest(args)
        elif args.command == "list":
            result = list_strategies(args)
        elif args.command == "analyze":
            result = analyze_market(args)
        else:
            parser.print_help()

        return result

    finally:
        # Shutdown tracing gracefully
        if tracing_enabled:
            shutdown_tracing()


if __name__ == "__main__":
    sys.exit(main())
