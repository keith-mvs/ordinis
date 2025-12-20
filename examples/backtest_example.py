"""Example backtest script demonstrating full pipeline."""

import asyncio
from pathlib import Path

import pandas as pd

from ordinis.backtesting import BacktestConfig, BacktestRunner


async def main():
    """Run example backtest."""
    print("=" * 70)
    print("SIGNALCORE BACKTEST EXAMPLE")
    print("=" * 70)

    # Create sample data for demo (in production, data would come from files)
    print("\n[setup] Generating sample historical data...")

    # Create synthetic data directory
    data_dir = Path("data/historical")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample OHLCV data for 3 symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range("2022-01-01", "2024-01-01", freq="D")

    for i, symbol in enumerate(symbols):
        df = pd.DataFrame(
            {
                "date": dates,
                "open": 100 + i * 50 + pd.Series(range(len(dates))) * 0.1,
                "high": 102 + i * 50 + pd.Series(range(len(dates))) * 0.1,
                "low": 98 + i * 50 + pd.Series(range(len(dates))) * 0.1,
                "close": 101 + i * 50 + pd.Series(range(len(dates))) * 0.1,
                "volume": 1000000,
            }
        )
        df.set_index("date", inplace=True)
        df.to_parquet(data_dir / f"{symbol}.parquet")
        print(f"  ✓ Generated data for {symbol}")

    # Configure backtest
    config = BacktestConfig(
        name="example_signalcore",
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        commission_pct=0.001,  # 0.1% commission
        slippage_bps=5.0,  # 5 bps slippage
        max_position_size=0.1,  # 10% max per symbol
        max_portfolio_exposure=1.0,  # 100% max total
        rebalance_freq="1d",
    )

    print("\n[backtest] Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Capital: ${config.initial_capital:,.0f}")
    print(f"  Commission: {config.commission_pct * 100:.2f}%")
    print(f"  Slippage: {config.slippage_bps:.1f} bps")

    # Run backtest
    print("\n[backtest] Running backtest...")
    runner = BacktestRunner(config)

    try:
        metrics = await runner.run()

        # Print results
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        print("\nRETURNS:")
        print(f"  Total Return:      {metrics.total_return:>10.2f}%")
        print(f"  Annualized Return: {metrics.annualized_return:>10.2f}%")

        print("\nRISK METRICS:")
        print(f"  Volatility:        {metrics.volatility:>10.2f}%")
        print(f"  Max Drawdown:      {metrics.max_drawdown:>10.2f}%")
        print(f"  Avg Drawdown:      {metrics.avg_drawdown:>10.2f}%")
        print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")

        print("\nTRADING PERFORMANCE:")
        print(f"  Total Trades:      {metrics.num_trades:>10.0f}")
        print(f"  Win Rate:          {metrics.win_rate:>10.2f}%")
        print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
        print(f"  Avg Win:           ${metrics.avg_win:>9,.0f}")
        print(f"  Avg Loss:          ${metrics.avg_loss:>9,.0f}")
        print(f"  Largest Win:       ${metrics.largest_win:>9,.0f}")
        print(f"  Largest Loss:      ${metrics.largest_loss:>9,.0f}")
        print(f"  Expectancy:        ${metrics.expectancy:>9,.0f}")
        print(f"  Recovery Factor:   {metrics.recovery_factor:>10.2f}")

        print(f"\nFINAL EQUITY:        ${metrics.equity_final:>10,.0f}")
        print(f"\nARTIFACTS SAVED TO:  {runner.output_dir}")

        # List artifact files
        print("\nGenerated files:")
        for file in sorted(runner.output_dir.glob("*")):
            size = file.stat().st_size
            print(f"  - {file.name:<30} ({size:>8,} bytes)")

    except Exception as e:
        print(f"\n[error] Backtest failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
