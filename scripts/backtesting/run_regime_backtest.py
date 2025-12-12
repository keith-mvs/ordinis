"""
Comprehensive Regime-Stratified Backtest.

Tests all strategies across diverse market conditions using
multi-timeframe training data chunks.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.training_data_generator import (
    DataChunk,
    MarketRegime,
    TrainingConfig,
    TrainingDataGenerator,
)
from src.engines.proofbench.core.execution import Order, OrderSide, OrderType
from src.engines.proofbench.core.simulator import SimulationConfig, SimulationEngine


@dataclass
class RegimeBacktestResult:
    """Result from backtesting on a single chunk."""

    chunk_id: int
    regime: MarketRegime
    strategy_name: str
    duration_months: int
    start_date: str
    end_date: str
    strategy_return: float
    benchmark_return: float
    alpha: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


def create_ma_crossover(fast: int = 20, slow: int = 50):
    """Create MA crossover strategy."""
    position = {"shares": 0}
    ma_data = {"fast_ma": [], "slow_ma": []}

    def strategy(engine, symbol, bar):
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < slow:
            return

        fast_ma = closes.iloc[-fast:].mean()
        slow_ma = closes.iloc[-slow:].mean()

        ma_data["fast_ma"].append(fast_ma)
        ma_data["slow_ma"].append(slow_ma)

        if len(ma_data["fast_ma"]) < 2:
            return

        prev_fast = ma_data["fast_ma"][-2]
        prev_slow = ma_data["slow_ma"][-2]

        if prev_fast <= prev_slow and fast_ma > slow_ma:
            if position["shares"] == 0:
                shares = int(engine.portfolio.cash * 0.95 / bar.close)
                if shares > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                    )
                    engine.submit_order(order)
                    position["shares"] = shares

        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            if position["shares"] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

    return strategy


def create_rsi_strategy(period: int = 14, oversold: int = 30, overbought: int = 70):
    """Create RSI mean reversion strategy."""
    position = {"shares": 0}

    def calculate_rsi(closes: pd.Series, period: int) -> float:
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def strategy(engine, symbol, bar):
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < period + 1:
            return

        rsi = calculate_rsi(closes, period)

        if rsi < oversold and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        elif rsi > overbought and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_momentum_strategy(lookback: int = 20, threshold: float = 0.05):
    """Create momentum breakout strategy."""
    position = {"shares": 0}

    def strategy(engine, symbol, bar):
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < lookback:
            return

        momentum = (closes.iloc[-1] - closes.iloc[-lookback]) / closes.iloc[-lookback]

        if momentum > threshold and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        elif momentum < 0 and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_bollinger_strategy(period: int = 20, std_dev: float = 2.0):
    """Create Bollinger Bands strategy."""
    position = {"shares": 0}

    def strategy(engine, symbol, bar):
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < period:
            return

        sma = closes.iloc[-period:].mean()
        std = closes.iloc[-period:].std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        current_price = bar.close

        if current_price < lower and position["shares"] == 0:
            shares = int(engine.portfolio.cash * 0.95 / bar.close)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares

        elif current_price > upper and position["shares"] > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position["shares"],
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)
            position["shares"] = 0

    return strategy


def create_macd_strategy(fast: int = 12, slow: int = 26, signal: int = 9):
    """Create MACD strategy."""
    position = {"shares": 0}
    macd_history = {"macd": [], "signal": []}

    def strategy(engine, symbol, bar):
        data = engine.data[symbol]
        closes = data.loc[: bar.timestamp, "close"]

        if len(closes) < slow + signal:
            return

        exp1 = closes.ewm(span=fast, adjust=False).mean()
        exp2 = closes.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]

        macd_history["macd"].append(current_macd)
        macd_history["signal"].append(current_signal)

        if len(macd_history["macd"]) < 2:
            return

        prev_macd = macd_history["macd"][-2]
        prev_signal = macd_history["signal"][-2]

        if prev_macd <= prev_signal and current_macd > current_signal:
            if position["shares"] == 0:
                shares = int(engine.portfolio.cash * 0.95 / bar.close)
                if shares > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                    )
                    engine.submit_order(order)
                    position["shares"] = shares

        elif prev_macd >= prev_signal and current_macd < current_signal:
            if position["shares"] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

    return strategy


def run_chunk_backtest(
    chunk: DataChunk,
    strategy_factory,
    strategy_name: str,
    chunk_id: int,
    initial_capital: float = 100000.0,
) -> RegimeBacktestResult:
    """Run backtest on a single data chunk."""
    config = SimulationConfig(
        initial_capital=initial_capital,
        bar_frequency="1d",
        enable_logging=False,
    )

    engine = SimulationEngine(config)
    engine.load_data(chunk.symbol, chunk.data)
    engine.set_strategy(strategy_factory())

    results = engine.run()
    metrics = results.metrics

    benchmark_return = chunk.metrics["total_return"] * 100
    alpha = metrics.total_return - benchmark_return

    return RegimeBacktestResult(
        chunk_id=chunk_id,
        regime=chunk.regime,
        strategy_name=strategy_name,
        duration_months=chunk.duration_months,
        start_date=chunk.start_date.strftime("%Y-%m-%d"),
        end_date=chunk.end_date.strftime("%Y-%m-%d"),
        strategy_return=metrics.total_return,
        benchmark_return=benchmark_return,
        alpha=alpha,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        num_trades=metrics.num_trades,
    )


def print_regime_summary(results: list[RegimeBacktestResult], strategy_name: str):
    """Print regime-stratified performance summary."""
    print(f"\n{'='*80}")
    print(f"REGIME ANALYSIS: {strategy_name}")
    print("=" * 80)

    regime_data = {}
    for regime in MarketRegime:
        regime_results = [r for r in results if r.regime == regime]
        if regime_results:
            alphas = [r.alpha for r in regime_results]
            returns = [r.strategy_return for r in regime_results]
            sharpes = [r.sharpe_ratio for r in regime_results]
            win_vs_bh = sum(1 for r in regime_results if r.alpha > 0) / len(regime_results) * 100

            regime_data[regime] = {
                "count": len(regime_results),
                "avg_return": np.mean(returns),
                "avg_alpha": np.mean(alphas),
                "avg_sharpe": np.mean(sharpes),
                "win_vs_bh": win_vs_bh,
            }

    print(
        f"\n{'Regime':<12} {'Tests':>6} {'Avg Ret':>10} {'Avg Alpha':>11} {'Avg Sharpe':>11} {'Beat B&H':>10}"
    )
    print("-" * 70)

    for regime in MarketRegime:
        if regime in regime_data:
            d = regime_data[regime]
            print(
                f"{regime.value:<12} {d['count']:>6} {d['avg_return']:>+9.2f}% "
                f"{d['avg_alpha']:>+10.2f}% {d['avg_sharpe']:>11.2f} {d['win_vs_bh']:>9.1f}%"
            )

    # Overall
    all_alphas = [r.alpha for r in results]
    all_returns = [r.strategy_return for r in results]
    overall_win = sum(1 for a in all_alphas if a > 0) / len(all_alphas) * 100

    print("-" * 70)
    print(
        f"{'OVERALL':<12} {len(results):>6} {np.mean(all_returns):>+9.2f}% "
        f"{np.mean(all_alphas):>+10.2f}% {np.mean([r.sharpe_ratio for r in results]):>11.2f} "
        f"{overall_win:>9.1f}%"
    )


def main():
    print("=" * 80)
    print("COMPREHENSIVE REGIME-STRATIFIED BACKTEST")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Generate training data
    print("\n[1/2] Generating training data chunks...")
    config = TrainingConfig(
        symbols=["SPY"],
        chunk_sizes_months=[2, 3, 4, 6, 8, 10, 12],
        lookback_years=[5, 10, 15, 20],
        random_seed=42,
    )

    generator = TrainingDataGenerator(config)
    chunks = generator.generate_chunks("SPY", num_chunks=100, balance_regimes=True)

    print(f"Generated {len(chunks)} chunks across {len([r for r in MarketRegime])} regimes")
    print("\nRegime Distribution:")
    for regime, count in generator.get_regime_distribution(chunks).items():
        pct = count / len(chunks) * 100 if chunks else 0
        print(f"  {regime.value:<12}: {count:>4} ({pct:.1f}%)")

    # Define strategies
    strategies = [
        ("MA Crossover (20/50)", create_ma_crossover),
        ("RSI (14, 30/70)", create_rsi_strategy),
        ("Momentum (20d, 5%)", create_momentum_strategy),
        ("Bollinger (20, 2std)", create_bollinger_strategy),
        ("MACD (12/26/9)", create_macd_strategy),
    ]

    # Run backtests
    print("\n[2/2] Running backtests across all chunks...")

    all_results = {}
    for strategy_name, strategy_factory in strategies:
        print(f"\n  Testing {strategy_name}...", end=" ", flush=True)

        results = []
        for i, chunk in enumerate(chunks):
            try:
                result = run_chunk_backtest(
                    chunk=chunk,
                    strategy_factory=strategy_factory,
                    strategy_name=strategy_name,
                    chunk_id=i,
                )
                results.append(result)
            except Exception as e:
                pass  # Skip failed chunks

        all_results[strategy_name] = results
        win_rate = sum(1 for r in results if r.alpha > 0) / len(results) * 100 if results else 0
        print(f"{len(results)} chunks, {win_rate:.1f}% beat B&H")

    # Print regime summaries
    for strategy_name, results in all_results.items():
        print_regime_summary(results, strategy_name)

    # Final comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Strategy':<22} {'Chunks':>8} {'Avg Alpha':>11} {'Beat B&H':>10} {'Consistency':>12}"
    )
    print("-" * 70)

    for strategy_name, results in all_results.items():
        if not results:
            continue

        alphas = [r.alpha for r in results]
        win_rate = sum(1 for a in alphas if a > 0) / len(alphas) * 100

        # Consistency = performance in different regimes
        regime_wins = {}
        for regime in MarketRegime:
            regime_results = [r for r in results if r.regime == regime]
            if regime_results:
                regime_wins[regime] = sum(1 for r in regime_results if r.alpha > 0) / len(
                    regime_results
                )

        consistency = np.std(list(regime_wins.values())) if regime_wins else 1.0
        consistency_score = max(0, (1 - consistency) * 100)

        print(
            f"{strategy_name:<22} {len(results):>8} {np.mean(alphas):>+10.2f}% "
            f"{win_rate:>9.1f}% {consistency_score:>11.1f}"
        )

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    best_strategy = max(
        all_results.items(), key=lambda x: sum(r.alpha for r in x[1]) / len(x[1]) if x[1] else -999
    )
    best_name = best_strategy[0]
    best_results = best_strategy[1]
    best_alpha = np.mean([r.alpha for r in best_results]) if best_results else 0
    best_win = (
        sum(1 for r in best_results if r.alpha > 0) / len(best_results) * 100 if best_results else 0
    )

    print(f"\nBest Strategy: {best_name}")
    print(f"  Avg Alpha: {best_alpha:+.2f}%")
    print(f"  Beat B&H: {best_win:.1f}% of the time")

    if best_win >= 60:
        print("\n[PROMISING] Strategy shows potential for outperformance")
    elif best_win >= 50:
        print("\n[MARGINAL] Strategy needs improvement to consistently beat passive")
    else:
        print("\n[WEAK] No strategy reliably beats buy-and-hold across market conditions")

    print("\nRecommendation: Consider ensemble methods or regime-adaptive strategies")


if __name__ == "__main__":
    main()
