"""Backtest on REAL market data - no synthetic BS."""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402
from run_backtest_demo import (  # noqa: E402
    create_bollinger_strategy,
    create_ma_crossover_strategy,
    create_macd_strategy,
    create_momentum_strategy,
    create_rsi_strategy,
    run_backtest,
)


def main():
    """Run backtest on REAL SPY data."""
    print("=" * 70)
    print("BACKTEST ON REAL SPY DATA")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Initial Capital: $100,000")
    print("Data Source: Twelve Data API (real market data)")
    print()

    # Load REAL data
    data_path = project_root / "data" / "real_spy_daily.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"[DATA] {len(df)} trading days")
    print(f"[DATA] Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"[DATA] Start Price: ${df['close'].iloc[0]:.2f}")
    print(f"[DATA] End Price: ${df['close'].iloc[-1]:.2f}")

    # Buy and hold benchmark
    bh_return = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100
    bh_final = 100000 * (1 + bh_return / 100)
    print(f"[BENCHMARK] Buy & Hold: {bh_return:+.2f}% (${bh_final:,.2f})")
    print()

    # Define strategies
    strategies = [
        ("MA Crossover (20/50)", create_ma_crossover_strategy(20, 50)),
        ("RSI (14, 30/70)", create_rsi_strategy(14, 30, 70)),
        ("Momentum (20d, 5%)", create_momentum_strategy(20, 0.05)),
        ("Bollinger (20, 2std)", create_bollinger_strategy(20, 2.0)),
        ("MACD (12/26/9)", create_macd_strategy(12, 26, 9)),
    ]

    # Run backtests
    print("=" * 70)
    print("STRATEGY RESULTS (REAL DATA)")
    print("=" * 70)
    print()
    print(
        f"{'Strategy':<22} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}"
    )
    print("-" * 70)

    results = []
    for name, callback in strategies:
        try:
            result = run_backtest(df, callback, name)
            results.append(result)
            alpha = result.total_return - bh_return
            print(
                f"{name:<22} "
                f"{result.total_return:>+9.2f}% "
                f"{result.sharpe_ratio:>8.2f} "
                f"{result.max_drawdown:>7.2f}% "
                f"{result.win_rate:>7.1f}% "
                f"{result.total_trades:>7}"
            )
        except Exception as e:
            print(f"{name:<22} [ERROR: {e}]")

    # Summary
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Sort by Sharpe
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

    print("Ranked by Sharpe Ratio:")
    for i, r in enumerate(results, 1):
        alpha = r.total_return - bh_return
        status = "BEAT B&H" if alpha > 0 else "LOST TO B&H"
        print(f"  {i}. {r.name}: Sharpe={r.sharpe_ratio:.2f}, Alpha={alpha:+.2f}% [{status}]")

    print()
    print(f"BENCHMARK: Buy & Hold = {bh_return:+.2f}%")
    print()

    # Best performer
    best = results[0]
    best_alpha = best.total_return - bh_return
    print(f"[BEST] {best.name}")
    print(f"  Return: {best.total_return:+.2f}%")
    print(f"  Sharpe: {best.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {best.max_drawdown:.2f}%")
    print(f"  Win Rate: {best.win_rate:.1f}%")
    print(f"  Trades: {best.total_trades}")
    print(f"  Final Equity: ${best.final_equity:,.2f}")
    print(f"  Alpha vs B&H: {best_alpha:+.2f}%")
    print()

    # Honest assessment
    winners = [r for r in results if r.total_return > bh_return]
    print(f"[VERDICT] {len(winners)}/{len(results)} strategies beat buy-and-hold")
    if len(winners) == 0:
        print("[VERDICT] Active trading FAILED to add value on this dataset")


if __name__ == "__main__":
    main()
