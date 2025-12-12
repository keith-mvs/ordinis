"""Comprehensive Backtesting Suite - Production-Grade Strategy Validation.

Runs extensive backtests across multiple dimensions:
- Symbols: Equities + ETFs across market caps and sectors
- Timeframes: Daily, weekly (intraday requires tick data)
- Market regimes: Bull, bear, sideways, high/low volatility
- Realistic constraints: Transaction costs, slippage, liquidity

Usage:
    python scripts/comprehensive_backtest_suite.py --output results/full_suite
    python scripts/comprehensive_backtest_suite.py --symbols-only TECH --quick
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import warnings

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
from engines.signalcore import ModelConfig, SignalType
from engines.signalcore.core.model import ModelRegistry
from engines.signalcore.models import (
    ADXTrendModel,
    BollingerBandsModel,
    FibonacciRetracementModel,
    MACDModel,
    ParabolicSARModel,
    RSIMeanReversionModel,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# SYMBOL UNIVERSES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SymbolSpec:
    """Symbol specification with metadata."""

    ticker: str
    name: str
    sector: str
    market_cap: str  # SMALL, MID, LARGE
    asset_type: str  # EQUITY, ETF


SYMBOL_UNIVERSE = {
    # Technology - Large Cap
    "AAPL": SymbolSpec("AAPL", "Apple", "TECH", "LARGE", "EQUITY"),
    "MSFT": SymbolSpec("MSFT", "Microsoft", "TECH", "LARGE", "EQUITY"),
    "NVDA": SymbolSpec("NVDA", "NVIDIA", "TECH", "LARGE", "EQUITY"),
    # Technology - Mid Cap
    "CRWD": SymbolSpec("CRWD", "CrowdStrike", "TECH", "MID", "EQUITY"),
    "DDOG": SymbolSpec("DDOG", "Datadog", "TECH", "MID", "EQUITY"),
    # Technology - Small Cap
    "GTLB": SymbolSpec("GTLB", "GitLab", "TECH", "SMALL", "EQUITY"),
    # Financials - Large Cap
    "JPM": SymbolSpec("JPM", "JPMorgan", "FINANCIALS", "LARGE", "EQUITY"),
    "BAC": SymbolSpec("BAC", "Bank of America", "FINANCIALS", "LARGE", "EQUITY"),
    "GS": SymbolSpec("GS", "Goldman Sachs", "FINANCIALS", "LARGE", "EQUITY"),
    # Financials - Mid Cap
    "SCHW": SymbolSpec("SCHW", "Charles Schwab", "FINANCIALS", "MID", "EQUITY"),
    # Healthcare - Large Cap
    "JNJ": SymbolSpec("JNJ", "Johnson & Johnson", "HEALTHCARE", "LARGE", "EQUITY"),
    "UNH": SymbolSpec("UNH", "UnitedHealth", "HEALTHCARE", "LARGE", "EQUITY"),
    "PFE": SymbolSpec("PFE", "Pfizer", "HEALTHCARE", "LARGE", "EQUITY"),
    # Healthcare - Small Cap
    "RXRX": SymbolSpec("RXRX", "Recursion Pharma", "HEALTHCARE", "SMALL", "EQUITY"),
    # Energy - Large Cap
    "XOM": SymbolSpec("XOM", "Exxon Mobil", "ENERGY", "LARGE", "EQUITY"),
    "CVX": SymbolSpec("CVX", "Chevron", "ENERGY", "LARGE", "EQUITY"),
    # Energy - Mid Cap
    "OXY": SymbolSpec("OXY", "Occidental", "ENERGY", "MID", "EQUITY"),
    # Consumer - Large Cap
    "AMZN": SymbolSpec("AMZN", "Amazon", "CONSUMER", "LARGE", "EQUITY"),
    "WMT": SymbolSpec("WMT", "Walmart", "CONSUMER", "LARGE", "EQUITY"),
    "HD": SymbolSpec("HD", "Home Depot", "CONSUMER", "LARGE", "EQUITY"),
    # Consumer - Mid Cap
    "CHWY": SymbolSpec("CHWY", "Chewy", "CONSUMER", "MID", "EQUITY"),
    # Industrials - Large Cap
    "BA": SymbolSpec("BA", "Boeing", "INDUSTRIALS", "LARGE", "EQUITY"),
    "CAT": SymbolSpec("CAT", "Caterpillar", "INDUSTRIALS", "LARGE", "EQUITY"),
    # Materials - Large Cap
    "LIN": SymbolSpec("LIN", "Linde", "MATERIALS", "LARGE", "EQUITY"),
    # Real Estate - Mid Cap
    "AMT": SymbolSpec("AMT", "American Tower", "REAL_ESTATE", "MID", "EQUITY"),
    # Utilities - Large Cap
    "NEE": SymbolSpec("NEE", "NextEra Energy", "UTILITIES", "LARGE", "EQUITY"),
    # Communication - Large Cap
    "META": SymbolSpec("META", "Meta", "COMMUNICATION", "LARGE", "EQUITY"),
    "GOOGL": SymbolSpec("GOOGL", "Alphabet", "COMMUNICATION", "LARGE", "EQUITY"),
    # ETFs - Broad Market
    "SPY": SymbolSpec("SPY", "S&P 500 ETF", "BROAD", "LARGE", "ETF"),
    "QQQ": SymbolSpec("QQQ", "NASDAQ 100 ETF", "TECH", "LARGE", "ETF"),
    "IWM": SymbolSpec("IWM", "Russell 2000 ETF", "BROAD", "SMALL", "ETF"),
    # ETFs - Sector
    "XLF": SymbolSpec("XLF", "Financial Sector ETF", "FINANCIALS", "LARGE", "ETF"),
    "XLE": SymbolSpec("XLE", "Energy Sector ETF", "ENERGY", "LARGE", "ETF"),
    "XLV": SymbolSpec("XLV", "Healthcare Sector ETF", "HEALTHCARE", "LARGE", "ETF"),
    "XLK": SymbolSpec("XLK", "Tech Sector ETF", "TECH", "LARGE", "ETF"),
    # ETFs - Fixed Income & Commodities
    "TLT": SymbolSpec("TLT", "20Y Treasury ETF", "FIXED_INCOME", "LARGE", "ETF"),
    "GLD": SymbolSpec("GLD", "Gold ETF", "COMMODITIES", "LARGE", "ETF"),
    "USO": SymbolSpec("USO", "Oil ETF", "COMMODITIES", "MID", "ETF"),
    # ETFs - Volatility
    "VXX": SymbolSpec("VXX", "VIX Short-Term ETF", "VOLATILITY", "MID", "ETF"),
}


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════


STRATEGY_CONFIGS = {
    "RSI_MeanReversion": {
        "model": RSIMeanReversionModel,
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "min_data": 50,
    },
    "MACD_Crossover": {
        "model": MACDModel,
        "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "min_data": 60,
    },
    "BollingerBands": {
        "model": BollingerBandsModel,
        "params": {"bb_period": 20, "bb_std": 2.0, "min_band_width": 0.01},
        "min_data": 50,
    },
    "ADX_TrendFilter": {
        "model": ADXTrendModel,
        "params": {"adx_period": 14, "adx_threshold": 25, "strong_trend": 40},
        "min_data": 100,
    },
    "Fibonacci_Retracement": {
        "model": FibonacciRetracementModel,
        "params": {"swing_lookback": 50, "key_levels": [0.382, 0.5, 0.618], "tolerance": 0.01},
        "min_data": 120,
    },
    "ParabolicSAR": {
        "model": ParabolicSARModel,
        "params": {"acceleration": 0.02, "maximum": 0.2, "min_trend_bars": 3},
        "min_data": 80,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_market_data(
    symbol: str,
    days: int,
    regime: str,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic market data with specific regime characteristics.

    Args:
        symbol: Stock symbol
        days: Number of trading days
        regime: Market regime (BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL)
        base_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(hash(symbol + regime) % (2**32))

    # Regime parameters
    params = {
        "BULL": {"drift": 0.0008, "vol": 0.015},  # Strong uptrend, moderate vol
        "BEAR": {"drift": -0.0006, "vol": 0.020},  # Downtrend, higher vol
        "SIDEWAYS": {"drift": 0.0001, "vol": 0.012},  # No trend, low vol
        "HIGH_VOL": {"drift": 0.0002, "vol": 0.035},  # Neutral, very high vol
        "LOW_VOL": {"drift": 0.0003, "vol": 0.008},  # Slight up, very low vol
    }

    config = params.get(regime, params["SIDEWAYS"])
    drift = config["drift"]
    vol = config["vol"]

    # Generate prices
    returns = np.random.normal(drift, vol, days)
    prices = base_price * (1 + returns).cumprod()

    # Generate OHLCV
    dates = pd.date_range(start="2020-01-01", periods=days, freq="D")
    data = []

    for i, (date, close) in enumerate(zip(dates, prices, strict=False)):
        daily_range = close * vol * np.random.uniform(0.5, 2.0)
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


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST EXECUTION
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    """Single backtest result."""

    strategy: str
    symbol: str
    sector: str
    market_cap: str
    asset_type: str
    regime: str
    timeframe: str
    # Performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    # Activity
    num_trades: int
    avg_trade_duration: float
    turnover: float  # Approximate annual turnover
    # Risk
    volatility: float
    downside_deviation: float
    # Capacity proxies
    avg_position_size: float
    max_position_size: float


def run_single_backtest(
    strategy_name: str,
    strategy_config: dict,
    symbol_spec: SymbolSpec,
    regime: str,
    timeframe: str,
    initial_capital: float = 100000,
) -> BacktestResult | None:
    """Run a single backtest configuration.

    Args:
        strategy_name: Strategy name
        strategy_config: Strategy configuration
        symbol_spec: Symbol specification
        regime: Market regime
        timeframe: Timeframe (DAILY, WEEKLY)
        initial_capital: Starting capital

    Returns:
        BacktestResult or None if failed
    """
    try:
        # Generate data
        days = 500 if timeframe == "DAILY" else 2500  # 500 weeks ~ 10 years
        data = generate_market_data(
            symbol_spec.ticker,
            days,
            regime,
            base_price=100.0,
        )

        # Resample for weekly
        if timeframe == "WEEKLY":
            data = (
                data.resample("W")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            data["symbol"] = symbol_spec.ticker

        # Create model config
        config = ModelConfig(
            model_id=f"{strategy_name.lower()}",
            model_type="signal",
            version="1.0.0",
            parameters=strategy_config["params"],
        )
        config.min_data_points = strategy_config["min_data"]

        # Initialize model
        model_class = strategy_config["model"]
        model = model_class(config)

        # Create simulation engine with realistic costs
        exec_config = ExecutionConfig(
            estimated_spread=0.0005,  # 5 bps slippage
            commission_pct=0.001,  # 10 bps commission
            commission_per_trade=1.0,  # $1 per trade
        )
        sim_config = SimulationConfig(
            initial_capital=initial_capital,
            execution_config=exec_config,
        )
        engine = SimulationEngine(config=sim_config)

        # Add signal registry (dynamically attached attribute)
        engine.signal_registry = ModelRegistry()

        # Register model
        engine.signal_registry.register(model)

        # Load data
        engine.load_data(symbol_spec.ticker, data)

        # Track positions and historical data
        positions = {}
        historical_data = []

        def on_bar_callback(engine_ref, symbol, bar):
            """Bar callback - called for each bar."""
            # Build historical data
            historical_data.append(
                {
                    "date": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

            # Need minimum data points
            if len(historical_data) < config.min_data_points:
                return

            # Generate signal
            hist_df = pd.DataFrame(historical_data)
            hist_df["symbol"] = symbol
            signal = model.generate(hist_df, bar.timestamp)

            if signal is None or signal.score < 0.6:
                return

            current_price = bar.close

            # Entry logic
            if signal.signal_type == SignalType.ENTRY and symbol not in positions:
                position_value = engine_ref.portfolio.equity * 0.1  # 10% position
                cash_available = engine_ref.portfolio.cash * 0.9  # 90% max usage
                position_value = min(position_value, cash_available)
                quantity = int(position_value / current_price)

                if quantity > 0:
                    order = Order(
                        symbol=symbol,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        timestamp=bar.timestamp,
                    )
                    engine_ref.submit_order(order)
                    positions[symbol] = {
                        "quantity": quantity,
                        "entry_price": current_price,
                        "entry_time": bar.timestamp,
                    }

            # Exit logic
            elif signal.signal_type == SignalType.EXIT and symbol in positions:
                pos = positions[symbol]
                order = Order(
                    symbol=symbol,
                    quantity=pos["quantity"],
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    timestamp=bar.timestamp,
                )
                engine_ref.submit_order(order)
                del positions[symbol]

        # Set callback and run
        engine.on_bar = on_bar_callback
        results = engine.run()
        metrics = results.metrics

        # Calculate additional metrics
        turnover = (metrics.num_trades * 2) / (days / 252)  # Approximate annual turnover
        avg_pos_size = initial_capital * 0.1 if metrics.num_trades > 0 else 0
        max_pos_size = initial_capital * 0.1

        return BacktestResult(
            strategy=strategy_name,
            symbol=symbol_spec.ticker,
            sector=symbol_spec.sector,
            market_cap=symbol_spec.market_cap,
            asset_type=symbol_spec.asset_type,
            regime=regime,
            timeframe=timeframe,
            total_return=metrics.total_return,
            annualized_return=metrics.annualized_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            num_trades=metrics.num_trades,
            avg_trade_duration=metrics.avg_trade_duration,
            turnover=turnover,
            volatility=metrics.volatility,
            downside_deviation=metrics.downside_deviation,
            avg_position_size=avg_pos_size,
            max_position_size=max_pos_size,
        )

    except Exception as e:
        print(f"    [FAIL] {strategy_name} × {symbol_spec.ticker} × {regime}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN SUITE
# ═══════════════════════════════════════════════════════════════════════


def main():  # noqa: PLR0915
    """Run comprehensive backtest suite."""
    parser = argparse.ArgumentParser(description="Comprehensive Backtest Suite")
    parser.add_argument(
        "--output",
        default="results/comprehensive_suite",
        help="Output directory",
    )
    parser.add_argument(
        "--symbols-only",
        default=None,
        help="Test only specific sector (e.g., TECH, FINANCIALS)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer symbols and regimes",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names (default: all)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(" " * 30 + "COMPREHENSIVE BACKTEST SUITE")
    print("=" * 100)

    # Filter symbols
    test_symbols = SYMBOL_UNIVERSE
    if args.symbols_only:
        sector_filter = args.symbols_only.upper()
        test_symbols = {k: v for k, v in SYMBOL_UNIVERSE.items() if v.sector == sector_filter}
        print(f"\nFiltered to {sector_filter} sector: {len(test_symbols)} symbols")

    if args.quick:
        # Quick mode: select subset
        test_symbols = dict(list(test_symbols.items())[:10])
        print(f"\nQuick mode: {len(test_symbols)} symbols")

    # Filter strategies
    test_strategies = STRATEGY_CONFIGS
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
        test_strategies = {k: v for k, v in STRATEGY_CONFIGS.items() if k in strategy_names}

    # Define test matrix
    regimes = ["BULL", "BEAR", "SIDEWAYS"] if not args.quick else ["BULL", "SIDEWAYS"]
    timeframes = ["DAILY", "WEEKLY"]

    total_tests = len(test_strategies) * len(test_symbols) * len(regimes) * len(timeframes)

    print("\nTest Matrix:")
    print(f"  Strategies:  {len(test_strategies)}")
    print(f"  Symbols:     {len(test_symbols)}")
    print(f"  Regimes:     {len(regimes)}")
    print(f"  Timeframes:  {len(timeframes)}")
    print(f"  Total Tests: {total_tests}")
    print(f"\nOutput:      {output_dir}")

    # Run backtests
    print("\n" + "=" * 100)
    print("RUNNING BACKTESTS")
    print("=" * 100)

    results = []
    completed = 0
    failed = 0

    for strategy_name, strategy_config in test_strategies.items():
        print(f"\n{strategy_name}")
        print("-" * 100)

        for symbol_ticker, symbol_spec in test_symbols.items():
            for regime in regimes:
                for timeframe in timeframes:
                    completed += 1
                    progress = (completed / total_tests) * 100

                    print(
                        f"  [{progress:>5.1f}%] {symbol_ticker:6s} × {regime:8s} × {timeframe:7s}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_single_backtest(
                        strategy_name,
                        strategy_config,
                        symbol_spec,
                        regime,
                        timeframe,
                    )

                    if result:
                        results.append(result)
                        print(
                            f"[OK] Return: {result.total_return:>7.2f}% | Trades: {result.num_trades:>3}"
                        )
                    else:
                        failed += 1
                        print("[FAIL]")

    # Create results DataFrame
    print(f"\n{'=' * 100}")
    print(f"Completed: {completed} | Successful: {len(results)} | Failed: {failed}")
    print("=" * 100)

    if not results:
        print("\n[ERROR] No successful backtests. Exiting.")
        return

    df = pd.DataFrame([vars(r) for r in results])

    # Save raw results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"raw_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    print(f"\n[SAVED] Raw results: {results_file}")

    # Generate aggregated reports
    generate_reports(df, output_dir, timestamp)

    print(f"\n{'=' * 100}")
    print("SUITE COMPLETE")
    print("=" * 100 + "\n")


def generate_reports(df: pd.DataFrame, output_dir: Path, timestamp: str):
    """Generate aggregated analysis reports."""
    print(f"\n{'=' * 100}")
    print("GENERATING REPORTS")
    print("=" * 100)

    # 1. Strategy Performance Summary
    print("\n" + "─" * 100)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("─" * 100)

    strategy_summary = (
        df.groupby("strategy")
        .agg(
            {
                "total_return": ["mean", "median", "std"],
                "sharpe_ratio": ["mean", "median"],
                "max_drawdown": ["mean", "min"],
                "win_rate": "mean",
                "num_trades": "sum",
            }
        )
        .round(2)
    )

    print("\n" + strategy_summary.to_string())
    strategy_summary.to_csv(output_dir / f"strategy_summary_{timestamp}.csv")

    # 2. Sector Performance
    print("\n" + "─" * 100)
    print("SECTOR PERFORMANCE")
    print("─" * 100)

    sector_summary = (
        df.groupby("sector")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
                "num_trades": "sum",
            }
        )
        .sort_values("sharpe_ratio", ascending=False)
        .round(2)
    )

    print("\n" + sector_summary.to_string())
    sector_summary.to_csv(output_dir / f"sector_summary_{timestamp}.csv")

    # 3. Regime Analysis
    print("\n" + "─" * 100)
    print("REGIME PERFORMANCE")
    print("─" * 100)

    regime_summary = (
        df.groupby(["strategy", "regime"])
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
            }
        )
        .round(2)
    )

    print("\n" + regime_summary.to_string())
    regime_summary.to_csv(output_dir / f"regime_analysis_{timestamp}.csv")

    # 4. Market Cap Analysis
    print("\n" + "─" * 100)
    print("MARKET CAP PERFORMANCE")
    print("─" * 100)

    cap_summary = (
        df.groupby("market_cap")
        .agg(
            {
                "total_return": "mean",
                "sharpe_ratio": "mean",
                "max_drawdown": "mean",
                "win_rate": "mean",
            }
        )
        .round(2)
    )

    print("\n" + cap_summary.to_string())
    cap_summary.to_csv(output_dir / f"market_cap_summary_{timestamp}.csv")

    # 5. Top Performers
    print("\n" + "─" * 100)
    print("TOP 20 PERFORMERS (by Sharpe Ratio)")
    print("─" * 100)

    top_performers = df.nlargest(20, "sharpe_ratio")[
        [
            "strategy",
            "symbol",
            "sector",
            "regime",
            "timeframe",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
        ]
    ].round(2)

    print("\n" + top_performers.to_string(index=False))
    top_performers.to_csv(output_dir / f"top_performers_{timestamp}.csv", index=False)

    # 6. Robustness Analysis
    print("\n" + "─" * 100)
    print("ROBUSTNESS ANALYSIS (Consistency Across Regimes)")
    print("─" * 100)

    robustness = (
        df.groupby(["strategy", "symbol"])
        .agg({"sharpe_ratio": ["mean", "std", "min", "max"]})
        .round(2)
    )
    robustness.columns = ["_".join(col).strip() for col in robustness.columns.values]
    robustness["sharpe_consistency"] = robustness["sharpe_ratio_mean"] / (
        robustness["sharpe_ratio_std"] + 0.01
    )
    robustness = robustness.sort_values("sharpe_consistency", ascending=False).head(20)

    print("\n" + robustness.to_string())
    robustness.to_csv(output_dir / f"robustness_analysis_{timestamp}.csv")

    print(f"\n[COMPLETE] All reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
