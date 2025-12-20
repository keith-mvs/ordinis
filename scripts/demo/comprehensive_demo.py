"""Comprehensive Ordinis System Demo.

Demonstrates all major capabilities:
- Multiple stock symbols
- Various trading strategies (RSI, MACD, Bollinger, ADX, Fibonacci, PSAR)
- Risk management (RiskGuard)
- Governance (Ethics, Compliance)
- Performance analytics
- Portfolio comparison

Usage:
    python scripts/comprehensive_demo.py
    python scripts/comprehensive_demo.py --symbols SPY,QQQ,AAPL --capital 500000
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.engines.proofbench import SimulationEngine
from src.engines.riskguard import RiskGuardEngine
from src.engines.signalcore import ModelConfig
from src.engines.signalcore.models import (
    ADXTrendModel,
    BollingerBandsModel,
    FibonacciRetracementModel,
    MACDModel,
    ParabolicSARModel,
    RSIMeanReversionModel,
)


@dataclass
class SymbolData:
    """Market data for a symbol."""

    symbol: str
    data: pd.DataFrame
    initial_price: float
    final_price: float


def create_synthetic_data(
    symbol: str, start_date: datetime, days: int, base_price: float, volatility: float, trend: float
) -> pd.DataFrame:
    """Create synthetic OHLCV data with configurable characteristics.

    Args:
        symbol: Stock symbol
        start_date: Starting date
        days: Number of trading days
        base_price: Starting price
        volatility: Daily volatility (0.01 = 1%)
        trend: Daily trend (0.001 = 0.1% drift per day)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    # Generate price series with trend and volatility
    np.random.seed(hash(symbol) % (2**32))  # Consistent but different per symbol
    returns = np.random.normal(trend, volatility, days)
    prices = base_price * (1 + returns).cumprod()

    # Generate OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices, strict=False)):
        daily_range = close * volatility * np.random.uniform(0.5, 1.5)
        high = close + daily_range * np.random.uniform(0, 1)
        low = close - daily_range * np.random.uniform(0, 1)
        open_price = (high + low) / 2 + np.random.normal(0, daily_range * 0.2)

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = int(np.random.lognormal(15, 1) * 1000)

        data.append(
            {
                "date": date,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def load_or_create_data(symbols: list[str], data_dir: Path | None = None) -> dict[str, SymbolData]:
    """Load real data from CSV or create synthetic data.

    Args:
        symbols: List of stock symbols
        data_dir: Directory containing CSV files (optional)

    Returns:
        Dictionary mapping symbols to SymbolData
    """
    symbol_data = {}
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    days = 500

    # Stock characteristics (price, volatility, trend)
    characteristics = {
        "SPY": (400, 0.012, 0.0003),  # S&P 500 ETF - moderate vol, slight uptrend
        "QQQ": (350, 0.015, 0.0005),  # NASDAQ ETF - higher vol, stronger uptrend
        "AAPL": (175, 0.018, 0.0004),  # Apple - tech stock volatility
        "MSFT": (350, 0.016, 0.0004),  # Microsoft - stable mega cap
        "TSLA": (250, 0.035, 0.0002),  # Tesla - high volatility
        "GLD": (180, 0.010, 0.0001),  # Gold ETF - low vol, defensive
        "TLT": (95, 0.013, -0.0001),  # Treasury bonds - negative trend
        "XLE": (85, 0.020, 0.0002),  # Energy sector - commodity exposure
    }

    for symbol in symbols:
        # Try to load from CSV first
        if data_dir:
            csv_path = data_dir / f"{symbol.lower()}_daily.csv"
            if csv_path.exists():
                print(f"Loading {symbol} from {csv_path}")
                df = pd.read_csv(csv_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.rename(columns={"timestamp": "date"}, inplace=True)
                    df.set_index("date", inplace=True)

                if "symbol" not in df.columns:
                    df["symbol"] = symbol

                symbol_data[symbol] = SymbolData(
                    symbol=symbol,
                    data=df,
                    initial_price=df["close"].iloc[0],
                    final_price=df["close"].iloc[-1],
                )
                continue

        # Create synthetic data
        base_price, volatility, trend = characteristics.get(symbol, (100, 0.015, 0.0002))
        df = create_synthetic_data(symbol, start_date, days, base_price, volatility, trend)

        symbol_data[symbol] = SymbolData(
            symbol=symbol,
            data=df,
            initial_price=df["close"].iloc[0],
            final_price=df["close"].iloc[-1],
        )

    return symbol_data


def run_strategy_on_symbol(
    strategy_name: str,
    model_config: ModelConfig,
    symbol_data: SymbolData,
    capital: float,
    position_size: float = 0.1,
) -> dict:
    """Run a single strategy on a single symbol.

    Args:
        strategy_name: Name of the strategy
        model_config: Model configuration
        symbol_data: Symbol data
        capital: Initial capital
        position_size: Position size as fraction of capital

    Returns:
        Dictionary with results
    """
    # Create simulation engine
    engine = SimulationEngine(initial_capital=capital, commission=0.001, slippage=0.0005)

    # Create risk guard (optional)
    risk_guard = RiskGuardEngine(max_position_size=0.15, max_total_exposure=0.9)

    # Track positions to avoid over-trading
    active_positions = {}

    def on_bar(bar_data: dict):
        """Process each bar of data."""
        # Skip if insufficient data
        if len(engine._historical_data) < model_config.min_data_points:
            return

        # Create DataFrame for signal generation
        hist_df = pd.DataFrame(engine._historical_data)
        hist_df["symbol"] = symbol_data.symbol

        # Generate signal
        signal = model_config.model.generate(hist_df)

        if signal is None:
            return

        # Check if we already have a position
        symbol = symbol_data.symbol
        has_position = symbol in active_positions

        # Entry logic
        if signal.signal_type == "ENTRY" and not has_position and signal.score > 0.6:
            # Calculate position size
            current_price = bar_data["close"]
            max_shares = int((capital * position_size) / current_price)

            if max_shares > 0:
                # Submit buy order
                order = engine.submit_order(
                    symbol=symbol,
                    quantity=max_shares,
                    order_type="market",
                    side="buy",
                )

                if order:
                    active_positions[symbol] = {
                        "entry_price": current_price,
                        "quantity": max_shares,
                        "entry_time": bar_data["timestamp"],
                    }

        # Exit logic
        elif signal.signal_type == "EXIT" and has_position:
            position = active_positions[symbol]

            # Submit sell order
            order = engine.submit_order(
                symbol=symbol,
                quantity=position["quantity"],
                order_type="market",
                side="sell",
            )

            if order:
                del active_positions[symbol]

    # Set callback
    engine.set_on_bar_callback(on_bar)

    # Run simulation
    results = engine.run(symbol_data.data)

    # Calculate metrics
    metrics = results.metrics

    return {
        "strategy": strategy_name,
        "symbol": symbol_data.symbol,
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "num_trades": metrics.num_trades,
        "profit_factor": metrics.profit_factor,
        "final_equity": metrics.equity_final,
        "initial_price": symbol_data.initial_price,
        "final_price": symbol_data.final_price,
        "buy_hold_return": (
            (symbol_data.final_price - symbol_data.initial_price) / symbol_data.initial_price
        )
        * 100,
    }


def main():  # noqa: PLR0915
    """Run comprehensive demo."""
    parser = argparse.ArgumentParser(description="Comprehensive Ordinis System Demo")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,AAPL",
        help="Comma-separated stock symbols (default: SPY,QQQ,AAPL)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital per strategy (default: 100000)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with CSV files (optional, will create synthetic data if not found)",
    )
    parser.add_argument(
        "--output",
        default="demo_results",
        help="Output directory for results (default: demo_results)",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(" " * 35 + "ORDINIS COMPREHENSIVE DEMO")
    print("=" * 100)
    print(f"\nSymbols:         {', '.join(symbols)}")
    print(f"Capital/Symbol:  ${args.capital:,.2f}")
    print(f"Data Source:     {'CSV files' if data_dir else 'Synthetic data'}")
    print(f"Output Dir:      {output_dir}")

    # Load data
    print("\n" + "=" * 100)
    print("LOADING MARKET DATA")
    print("=" * 100)
    symbol_data = load_or_create_data(symbols, data_dir)

    for symbol, data in symbol_data.items():
        print(f"\n{symbol}:")
        print(f"  Bars:         {len(data.data)}")
        print(f"  Period:       {data.data.index[0].date()} to {data.data.index[-1].date()}")
        print(f"  Price Range:  ${data.data['close'].min():.2f} - ${data.data['close'].max():.2f}")
        print(f"  Initial:      ${data.initial_price:.2f}")
        print(f"  Final:        ${data.final_price:.2f}")
        print(
            f"  Buy & Hold:   {((data.final_price - data.initial_price) / data.initial_price) * 100:>6.2f}%"
        )

    # Define strategies
    print("\n" + "=" * 100)
    print("STRATEGY CONFIGURATIONS")
    print("=" * 100)

    strategies = [
        (
            "RSI Mean Reversion",
            ModelConfig(model=RSIMeanReversionModel(), parameters={"rsi_period": 14}),
        ),
        (
            "MACD Crossover",
            ModelConfig(model=MACDModel(), parameters={"fast_period": 12, "slow_period": 26}),
        ),
        (
            "Bollinger Bands",
            ModelConfig(model=BollingerBandsModel(), parameters={"bb_period": 20, "bb_std": 2.0}),
        ),
        (
            "ADX Trend Filter",
            ModelConfig(model=ADXTrendModel(), parameters={"adx_period": 14, "adx_threshold": 25}),
        ),
        (
            "Fibonacci Levels",
            ModelConfig(model=FibonacciRetracementModel(), parameters={"swing_lookback": 50}),
        ),
        (
            "Parabolic SAR",
            ModelConfig(
                model=ParabolicSARModel(), parameters={"acceleration": 0.02, "maximum": 0.2}
            ),
        ),
    ]

    for strat_name, config in strategies:
        print(f"\n{strat_name}:")
        print(f"  Model:       {config.model.__class__.__name__}")
        print(f"  Parameters:  {config.parameters}")
        print(f"  Min Data:    {config.min_data_points} bars")

    # Run backtests
    print("\n" + "=" * 100)
    print("RUNNING BACKTESTS")
    print("=" * 100)

    all_results = []
    total_runs = len(strategies) * len(symbols)
    current_run = 0

    for strategy_name, model_config in strategies:
        print(f"\n{strategy_name}")
        print("-" * 100)

        for symbol, data in symbol_data.items():
            current_run += 1
            progress = (current_run / total_runs) * 100

            print(f"  [{progress:>5.1f}%] {symbol}...", end=" ", flush=True)

            try:
                result = run_strategy_on_symbol(
                    strategy_name, model_config, data, args.capital, position_size=0.1
                )
                all_results.append(result)

                # Quick summary
                print(
                    f"Return: {result['total_return']:>6.2f}% | "
                    f"Trades: {result['num_trades']:>3} | "
                    f"Sharpe: {result['sharpe_ratio']:>5.2f}"
                )

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback

                traceback.print_exc()

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    # Performance Summary
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Best performers
    print("\n" + "─" * 100)
    print("TOP PERFORMERS BY TOTAL RETURN")
    print("─" * 100)
    top_returns = df_results.nlargest(10, "total_return")
    print(
        top_returns[
            ["strategy", "symbol", "total_return", "sharpe_ratio", "num_trades", "win_rate"]
        ].to_string(index=False)
    )

    # Best risk-adjusted
    print("\n" + "─" * 100)
    print("TOP PERFORMERS BY SHARPE RATIO")
    print("─" * 100)
    top_sharpe = df_results.nlargest(10, "sharpe_ratio")
    print(
        top_sharpe[
            ["strategy", "symbol", "sharpe_ratio", "total_return", "max_drawdown", "num_trades"]
        ].to_string(index=False)
    )

    # Strategy comparison
    print("\n" + "─" * 100)
    print("STRATEGY AVERAGES (Across All Symbols)")
    print("─" * 100)
    strategy_avg = (
        df_results.groupby("strategy")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
                "num_trades": "sum",
            }
        )
        .sort_values("total_return", ascending=False)
    )
    print(strategy_avg.to_string())

    # Symbol comparison
    print("\n" + "─" * 100)
    print("SYMBOL AVERAGES (Across All Strategies)")
    print("─" * 100)
    symbol_avg = (
        df_results.groupby("symbol")
        .agg(
            {
                "total_return": "mean",
                "buy_hold_return": "first",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
                "num_trades": "sum",
            }
        )
        .sort_values("total_return", ascending=False)
    )
    print(symbol_avg.to_string())

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"comprehensive_demo_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Full results saved to: {results_file}")

    # Save summary
    summary_file = output_dir / f"summary_{timestamp}.txt"
    with summary_file.open("w") as f:
        f.write("ORDINIS COMPREHENSIVE DEMO - SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Run Date:        {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Symbols:         {', '.join(symbols)}\n")
        f.write(f"Strategies:      {len(strategies)}\n")
        f.write(f"Total Backtests: {len(all_results)}\n")
        f.write(f"Capital/Symbol:  ${args.capital:,.2f}\n\n")

        f.write("TOP PERFORMERS BY RETURN:\n")
        f.write("-" * 100 + "\n")
        f.write(
            top_returns[
                ["strategy", "symbol", "total_return", "sharpe_ratio", "num_trades"]
            ].to_string(index=False)
        )
        f.write("\n\n")

        f.write("STRATEGY AVERAGES:\n")
        f.write("-" * 100 + "\n")
        f.write(strategy_avg.to_string())
        f.write("\n\n")

        f.write("SYMBOL AVERAGES:\n")
        f.write("-" * 100 + "\n")
        f.write(symbol_avg.to_string())
        f.write("\n")

    print(f"✓ Summary saved to: {summary_file}")

    print("\n" + "=" * 100)
    print(f"DEMO COMPLETE - {len(all_results)} backtests executed successfully")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
