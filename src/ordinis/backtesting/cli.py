#!/usr/bin/env python
"""CLI entry point for backtesting SignalCore strategies."""

import argparse
import asyncio
from pathlib import Path

from ordinis.backtesting import BacktestConfig, BacktestRunner


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run backtests of SignalCore strategies")
    parser.add_argument("--name", required=True, help="Backtest name")
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols (e.g., AAPL,MSFT)"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission % (default 0.1%)"
    )
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps (default 5)")
    parser.add_argument(
        "--max-position", type=float, default=0.1, help="Max position size % (default 10%)"
    )
    parser.add_argument("--output", type=Path, help="Output directory")

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    config = BacktestConfig(
        name=args.name,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        commission_pct=args.commission,
        slippage_bps=args.slippage,
        max_position_size=args.max_position,
    )

    runner = BacktestRunner(config, args.output)
    metrics = await runner.run()

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {config.name}")
    print("=" * 60)
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print()
    print("RETURNS")
    print(f"  Total Return: {metrics.total_return:>8.2f}%")
    print(f"  Annualized:   {metrics.annualized_return:>8.2f}%")
    print()
    print("RISK")
    print(f"  Volatility:    {metrics.volatility:>8.2f}%")
    print(f"  Max Drawdown:  {metrics.max_drawdown:>8.2f}%")
    print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:>8.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:>8.2f}")
    print()
    print("TRADING")
    print(f"  Total Trades:  {metrics.num_trades:>8.0f}")
    print(f"  Win Rate:      {metrics.win_rate:>8.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:>8.2f}")
    print(f"  Avg Win:       ${metrics.avg_win:>7,.0f}")
    print(f"  Avg Loss:      ${metrics.avg_loss:>7,.0f}")
    print()
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
