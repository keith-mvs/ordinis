"""
Real data validation script for backtesting framework.

This script downloads historical market data (real or synthetic realistic data)
and runs comprehensive backtests to establish baseline IC scores and model
performance metrics before live trading deployment.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ordinis.backtesting import BacktestConfig, BacktestRunner


async def generate_realistic_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic market data using geometric Brownian motion.

    Args:
        symbol: Stock symbol
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(hash(symbol) % 2**32)  # Deterministic seed per symbol

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start, end)

    # Geometric Brownian Motion parameters
    initial_price = 100 + (hash(symbol) % 100)  # Different base price per symbol
    mu = 0.0001  # Daily drift
    sigma = 0.02  # Daily volatility (2%)

    # Generate price movements
    dt = 1
    prices = [initial_price]

    for _ in range(len(dates) - 1):
        # GBM: dS/S = mu*dt + sigma*sqrt(dt)*Z
        z = np.random.normal(0, 1)
        price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(price)

    prices = np.array(prices)

    # Generate OHLC from close prices
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.005, len(prices))),
            "high": prices * (1 + abs(np.random.normal(0.01, 0.01, len(prices)))),
            "low": prices * (1 - abs(np.random.normal(0.01, 0.01, len(prices)))),
            "close": prices,
            "volume": np.random.normal(1e6, 1e5, len(prices)).astype(int),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    df["high"] = df[["open", "close", "high"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close", "low"]].min(axis=1) * 0.999

    return df


async def download_real_data(symbols: list[str], start_date: str, end_date: str) -> dict:
    """Download or generate realistic data for validation.

    Note: In production, this would use yfinance or Alpha Vantage.
    For now, we generate realistic synthetic data.

    Args:
        symbols: List of symbols
        start_date: Start date
        end_date: End date

    Returns:
        Dict of {symbol: DataFrame}
    """
    print(f"\n[data] Generating realistic historical data for {len(symbols)} symbols...")

    data = {}
    for symbol in symbols:
        print(f"  ⧖ {symbol}", end="", flush=True)
        df = await generate_realistic_data(symbol, start_date, end_date)
        data[symbol] = df
        print(f" ✓ ({len(df)} bars)")

    return data


async def run_backtest_scenario(
    name: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    **kwargs,
) -> dict:
    """Run a backtest scenario.

    Args:
        name: Scenario name
        symbols: Trading symbols
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        **kwargs: Additional config params

    Returns:
        Dict with metrics and analysis
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")

    config = BacktestConfig(
        name=name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **kwargs,
    )

    print("\nConfig:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: ${initial_capital:,.0f}")
    print(f"  Commission: {config.commission_pct * 100:.2f}%")
    print(f"  Slippage: {config.slippage_bps:.1f} bps")
    print(f"  Max Position: {config.max_position_size * 100:.1f}%")

    runner = BacktestRunner(config)

    try:
        print("\nRunning backtest...")
        metrics = await runner.run()

        print("\n✓ Backtest complete")
        print(f"  Total Return: {metrics.total_return:.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"  Win Rate: {metrics.win_rate:.2f}%")
        print(f"  Trades: {metrics.num_trades:.0f}")

        # Extract model metrics
        model_metrics = metrics.model_metrics if hasattr(metrics, "model_metrics") else {}

        return {
            "name": name,
            "config": config.__dict__,
            "metrics": {
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "num_trades": metrics.num_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "ic_mean": metrics.ic_mean if hasattr(metrics, "ic_mean") else 0.0,
                "ic_std": metrics.ic_std if hasattr(metrics, "ic_std") else 0.0,
                "hit_rate": metrics.hit_rate if hasattr(metrics, "hit_rate") else 0.0,
                "equity_final": metrics.equity_final,
            },
            "model_metrics": model_metrics,
            "output_dir": str(runner.output_dir),
        }

    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "name": name,
            "error": str(e),
        }


async def main():
    """Run comprehensive validation suite."""

    print("=" * 70)
    print("REAL DATA VALIDATION & BASELINE ESTABLISHMENT")
    print("=" * 70)

    # Phase 1: Generate realistic data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    data = await download_real_data(symbols, start_date, end_date)

    # Save to parquet for reuse
    data_dir = Path("data/validation")
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in data.items():
        df.to_parquet(data_dir / f"{symbol}.parquet")

    print(f"\n✓ Data saved to {data_dir}")

    # Phase 2: Run baseline scenarios
    scenarios = []

    # Scenario 1: Conservative (low position size, high risk limits)
    result1 = await run_backtest_scenario(
        name="Conservative",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission_pct=0.001,
        slippage_bps=5.0,
        max_position_size=0.05,  # 5% max
        max_portfolio_exposure=0.8,  # 80% total
    )
    scenarios.append(result1)

    # Scenario 2: Moderate (balanced)
    result2 = await run_backtest_scenario(
        name="Moderate",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission_pct=0.001,
        slippage_bps=5.0,
        max_position_size=0.10,  # 10% max
        max_portfolio_exposure=1.0,  # 100% total
    )
    scenarios.append(result2)

    # Scenario 3: Aggressive (higher leverage)
    result3 = await run_backtest_scenario(
        name="Aggressive",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission_pct=0.001,
        slippage_bps=5.0,
        max_position_size=0.20,  # 20% max
        max_portfolio_exposure=1.5,  # 150% total (margin)
    )
    scenarios.append(result3)

    # Phase 3: Generate summary report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    # Create comparison table
    summary_data = []
    for scenario in scenarios:
        if "error" in scenario:
            continue

        metrics = scenario["metrics"]
        summary_data.append(
            {
                "Scenario": scenario["name"],
                "Return": f"{metrics['total_return']:>8.2f}%",
                "Sharpe": f"{metrics['sharpe_ratio']:>8.2f}",
                "Drawdown": f"{metrics['max_drawdown']:>8.2f}%",
                "Trades": f"{metrics['num_trades']:>8.0f}",
                "Win Rate": f"{metrics['win_rate']:>8.2f}%",
                "IC": f"{metrics['ic_mean']:>8.3f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save detailed report
    report_path = Path("validation_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "data_period": f"{start_date} to {end_date}",
                "symbols": symbols,
                "scenarios": scenarios,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n✓ Detailed report saved to {report_path}")

    # Phase 4: Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR LIVE DEPLOYMENT")
    print(f"{'='*70}\n")

    best_scenario = max(
        [s for s in scenarios if "error" not in s], key=lambda x: x["metrics"]["sharpe_ratio"]
    )

    print(f"✓ Best scenario by Sharpe ratio: {best_scenario['name']}")
    print("\nSettings for live trading:")
    config = best_scenario["config"]
    print(f"  - Commission: {config['commission_pct'] * 100:.2f}%")
    print(f"  - Slippage: {config['slippage_bps']:.1f} bps")
    print(f"  - Max Position Size: {config['max_position_size'] * 100:.1f}%")
    print(f"  - Max Portfolio Exposure: {config['max_portfolio_exposure'] * 100:.1f}%")

    print("\nModel Performance (IC scores):")
    model_metrics = best_scenario.get("model_metrics", {})
    if model_metrics:
        for model_name, metrics in sorted(
            model_metrics.items(), key=lambda x: x[1].get("ic", 0), reverse=True
        ):
            ic = metrics.get("ic", 0)
            hit_rate = metrics.get("hit_rate", 0)
            print(f"  - {model_name:<20} IC={ic:.3f}, Hit Rate={hit_rate:.1f}%")
    else:
        print("  (No model metrics available)")

    print("\n✓ Next step: Use IC scores to initialize live ensemble weights")
    print("✓ Monitor live trading for drift vs backtest performance")
    print("✓ Retrain models when Sharpe ratio degrades > 0.5")


if __name__ == "__main__":
    asyncio.run(main())
