"""
Backtest and ML Optimization for GS-Quant Strategies.

This script:
1. Runs backtests on BollingerRSIConfluenceStrategy with various parameter sets
2. Uses grid search and optionally Bayesian optimization for hyperparameter tuning
3. Generates performance reports and visualizations

Usage:
    python scripts/backtest_gs_quant_strategies.py
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.application.strategies.bollinger_rsi_confluence import (
    BollingerRSIConfluenceStrategy,
)
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.quant import generate_series, max_drawdown, returns, sharpe_ratio, volatility

# =============================================================================
# Data Generation
# =============================================================================


def generate_synthetic_ohlcv(
    n_days: int = 504,  # ~2 years
    start_price: float = 100.0,
    daily_volatility: float = 0.015,
    trend: float = 0.0002,
    mean_reversion_strength: float = 0.02,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with configurable characteristics.

    Args:
        n_days: Number of trading days
        start_price: Starting price
        daily_volatility: Daily return volatility
        trend: Daily drift (positive for uptrend)
        mean_reversion_strength: Strength of mean reversion (0-1)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate prices with mean reversion and trend
    prices = [start_price]
    target_price = start_price

    for _ in range(n_days - 1):
        # Mean reversion component
        reversion = mean_reversion_strength * (target_price - prices[-1])
        # Random walk component
        random_return = np.random.normal(trend, daily_volatility)
        # Combined return
        new_price = prices[-1] * (1 + random_return + reversion / prices[-1])
        prices.append(max(new_price, 1.0))  # Floor at 1

        # Occasionally shift target for regime changes
        if np.random.random() < 0.01:
            target_price = prices[-1] * np.random.uniform(0.9, 1.1)

    prices = np.array(prices)

    # Generate OHLC from close prices
    daily_range = daily_volatility * 1.5
    high = prices * (1 + np.random.uniform(0, daily_range, n_days))
    low = prices * (1 - np.random.uniform(0, daily_range, n_days))
    open_prices = np.roll(prices, 1)
    open_prices[0] = start_price

    # Generate volume with some correlation to price movement
    base_volume = 1_000_000
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    volume = base_volume * (1 + price_changes / prices * 10) * np.random.uniform(0.5, 1.5, n_days)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume.astype(int),
        },
        index=dates,
    )


# =============================================================================
# Backtesting Framework
# =============================================================================


@dataclass
class BacktestResult:
    """Results from a single backtest run."""

    strategy_name: str
    params: dict[str, Any]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    profit_factor: float
    calmar_ratio: float
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "params": self.params,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
        }


class SimpleBacktester:
    """Simple vectorized backtester for strategy evaluation."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size: float = 0.1,
        commission: float = 0.001,
        slippage_bps: int = 5,
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage_bps / 10000

    async def run(
        self,
        strategy: BollingerRSIConfluenceStrategy,
        data: pd.DataFrame,
        symbol: str = "TEST",
    ) -> BacktestResult:
        """
        Run backtest on strategy with given data.

        Args:
            strategy: Strategy instance to test
            data: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            BacktestResult with performance metrics
        """
        # Generate signals for each bar
        signals = []
        min_bars = strategy.get_required_bars()

        for i in range(min_bars, len(data)):
            window = data.iloc[: i + 1].copy()
            timestamp = data.index[i]

            try:
                signal = await strategy.generate_signal(window, timestamp)
                signals.append(
                    {
                        "timestamp": timestamp,
                        "signal": signal,
                        "price": data.iloc[i]["close"],
                    }
                )
            except Exception:
                signals.append(
                    {"timestamp": timestamp, "signal": None, "price": data.iloc[i]["close"]}
                )

        # Simulate trading
        cash = self.initial_capital
        position = 0
        equity_curve = []
        trades = []

        entry_price = None
        entry_time = None

        for sig_data in signals:
            signal = sig_data["signal"]
            price = sig_data["price"]
            ts = sig_data["timestamp"]

            # Calculate current equity
            equity = cash + position * price
            equity_curve.append({"timestamp": ts, "equity": equity})

            if signal is None:
                continue

            # Skip non-actionable signals
            if signal.signal_type == SignalType.HOLD:
                continue

            # Entry signals
            if signal.signal_type == SignalType.ENTRY:
                if signal.direction == Direction.LONG and position <= 0:
                    # Close any short, go long
                    if position < 0:
                        # Cover short
                        cover_cost = abs(position) * price * (1 + self.slippage + self.commission)
                        pnl = entry_price * abs(position) - cover_cost
                        cash += pnl
                        trades.append(
                            {
                                "entry_time": entry_time,
                                "exit_time": ts,
                                "side": "SHORT",
                                "entry_price": entry_price,
                                "exit_price": price,
                                "quantity": abs(position),
                                "pnl": pnl,
                                "return_pct": (entry_price - price) / entry_price * 100,
                            }
                        )

                    # Enter long
                    target_value = equity * self.position_size
                    qty = int(target_value / (price * (1 + self.slippage + self.commission)))
                    if qty > 0 and cash >= qty * price * (1 + self.slippage + self.commission):
                        cost = qty * price * (1 + self.slippage + self.commission)
                        cash -= cost
                        position = qty
                        entry_price = price
                        entry_time = ts

                elif signal.direction == Direction.SHORT and position >= 0:
                    # Close any long, go short
                    if position > 0:
                        # Sell long
                        proceeds = position * price * (1 - self.slippage - self.commission)
                        pnl = proceeds - entry_price * position
                        cash += proceeds
                        trades.append(
                            {
                                "entry_time": entry_time,
                                "exit_time": ts,
                                "side": "LONG",
                                "entry_price": entry_price,
                                "exit_price": price,
                                "quantity": position,
                                "pnl": pnl,
                                "return_pct": (price - entry_price) / entry_price * 100,
                            }
                        )

                    # Enter short
                    target_value = equity * self.position_size
                    qty = int(target_value / price)
                    if qty > 0:
                        proceeds = qty * price * (1 - self.slippage - self.commission)
                        cash += proceeds
                        position = -qty
                        entry_price = price
                        entry_time = ts

        # Close any remaining position at end
        if position != 0 and len(signals) > 0:
            final_price = signals[-1]["price"]
            if position > 0:
                proceeds = position * final_price * (1 - self.slippage - self.commission)
                pnl = proceeds - entry_price * position
                cash += proceeds
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": signals[-1]["timestamp"],
                        "side": "LONG",
                        "entry_price": entry_price,
                        "exit_price": final_price,
                        "quantity": position,
                        "pnl": pnl,
                        "return_pct": (final_price - entry_price) / entry_price * 100,
                    }
                )
            else:
                cover_cost = abs(position) * final_price * (1 + self.slippage + self.commission)
                pnl = entry_price * abs(position) - cover_cost
                cash += entry_price * abs(position) - cover_cost
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": signals[-1]["timestamp"],
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": final_price,
                        "quantity": abs(position),
                        "pnl": pnl,
                        "return_pct": (entry_price - final_price) / entry_price * 100,
                    }
                )
            position = 0

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
        equity_series = equity_df["equity"]

        if len(equity_series) < 2:
            return self._empty_result(strategy.name, strategy.params)

        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        n_years = len(equity_series) / 252
        annualized_return = ((1 + total_return / 100) ** (1 / max(n_years, 0.01)) - 1) * 100

        # Use gs-quant functions for metrics
        equity_returns = returns(equity_series)
        vol = volatility(equity_series, w=min(22, len(equity_series) - 1), returns_type=None)
        dd = max_drawdown(equity_series)
        sr = sharpe_ratio(equity_series)

        # Trade metrics
        num_trades = len(trades)
        if num_trades > 0:
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] <= 0]
            win_rate = len(winning_trades) / num_trades * 100
            avg_trade_return = sum(t["return_pct"] for t in trades) / num_trades

            gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 1
            profit_factor = gross_profit / max(gross_loss, 1)
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0

        max_dd = abs(dd.min()) if len(dd) > 0 else 0
        calmar = annualized_return / max(max_dd * 100, 0.01) if max_dd > 0 else 0

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.params.copy(),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=float(sr) if not np.isnan(sr) else 0,
            max_drawdown=max_dd * 100,
            volatility=float(vol.iloc[-1]) if len(vol) > 0 and not np.isnan(vol.iloc[-1]) else 0,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            equity_curve=equity_series,
            trades=trades,
        )

    def _empty_result(self, name: str, params: dict) -> BacktestResult:
        return BacktestResult(
            strategy_name=name,
            params=params,
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            volatility=0,
            num_trades=0,
            win_rate=0,
            avg_trade_return=0,
            profit_factor=0,
            calmar_ratio=0,
        )


# =============================================================================
# Hyperparameter Optimization
# =============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Bollinger Bands params
    bb_periods: list[int] = field(default_factory=lambda: [10, 15, 20, 25])
    bb_stds: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])

    # RSI params
    rsi_periods: list[int] = field(default_factory=lambda: [7, 10, 14, 21])
    rsi_oversold: list[int] = field(default_factory=lambda: [25, 30, 35])
    rsi_overbought: list[int] = field(default_factory=lambda: [65, 70, 75])

    # Volatility filter
    min_volatilities: list[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])

    def get_param_grid(self) -> list[dict]:
        """Generate all parameter combinations."""
        params = []
        for bb_period, bb_std, rsi_period, rsi_os, rsi_ob, min_vol in product(
            self.bb_periods,
            self.bb_stds,
            self.rsi_periods,
            self.rsi_oversold,
            self.rsi_overbought,
            self.min_volatilities,
        ):
            # Skip invalid combinations
            if rsi_os >= rsi_ob:
                continue
            params.append(
                {
                    "bb_period": bb_period,
                    "bb_std": bb_std,
                    "rsi_period": rsi_period,
                    "rsi_oversold": rsi_os,
                    "rsi_overbought": rsi_ob,
                    "min_volatility": min_vol,
                }
            )
        return params


class GridSearchOptimizer:
    """Grid search optimizer for strategy parameters."""

    def __init__(
        self,
        backtester: SimpleBacktester,
        optimization_metric: str = "sharpe_ratio",
    ):
        self.backtester = backtester
        self.optimization_metric = optimization_metric
        self.results: list[BacktestResult] = []

    async def optimize(
        self,
        data: pd.DataFrame,
        config: OptimizationConfig,
        max_iterations: int | None = None,
    ) -> list[BacktestResult]:
        """
        Run grid search optimization.

        Args:
            data: OHLCV data for backtesting
            config: Optimization configuration
            max_iterations: Max parameter combinations to test

        Returns:
            List of BacktestResults sorted by optimization metric
        """
        param_grid = config.get_param_grid()
        if max_iterations:
            param_grid = param_grid[:max_iterations]

        print(f"[Optimizer] Testing {len(param_grid)} parameter combinations...")

        self.results = []
        for i, params in enumerate(param_grid):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(param_grid)}")

            strategy = BollingerRSIConfluenceStrategy(
                name=f"bb_rsi_opt_{i}",
                **params,
            )

            result = await self.backtester.run(strategy, data)
            self.results.append(result)

        # Sort by optimization metric (descending)
        self.results.sort(key=lambda r: getattr(r, self.optimization_metric), reverse=True)

        return self.results

    def get_best_params(self) -> dict:
        """Get parameters of best performing run."""
        if not self.results:
            return {}
        return self.results[0].params

    def get_summary(self, top_n: int = 10) -> pd.DataFrame:
        """Get summary of top N results."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results[:top_n]:
            row = {
                "bb_period": r.params.get("bb_period"),
                "bb_std": r.params.get("bb_std"),
                "rsi_period": r.params.get("rsi_period"),
                "rsi_oversold": r.params.get("rsi_oversold"),
                "rsi_overbought": r.params.get("rsi_overbought"),
                "min_volatility": r.params.get("min_volatility"),
                "sharpe_ratio": round(r.sharpe_ratio, 3),
                "total_return": round(r.total_return, 2),
                "max_drawdown": round(r.max_drawdown, 2),
                "win_rate": round(r.win_rate, 1),
                "num_trades": r.num_trades,
                "calmar_ratio": round(r.calmar_ratio, 3),
            }
            rows.append(row)

        return pd.DataFrame(rows)


# =============================================================================
# Walk-Forward Analysis
# =============================================================================


class WalkForwardAnalyzer:
    """Walk-forward optimization and out-of-sample testing."""

    def __init__(
        self,
        backtester: SimpleBacktester,
        train_ratio: float = 0.7,
        n_folds: int = 5,
    ):
        self.backtester = backtester
        self.train_ratio = train_ratio
        self.n_folds = n_folds

    async def run(
        self,
        data: pd.DataFrame,
        config: OptimizationConfig,
    ) -> dict:
        """
        Run walk-forward analysis.

        Args:
            data: Full OHLCV data
            config: Optimization config

        Returns:
            Dictionary with in-sample and out-of-sample results
        """
        n = len(data)
        fold_size = n // self.n_folds

        results = []

        for fold in range(self.n_folds):
            fold_start = fold * fold_size
            fold_end = min((fold + 1) * fold_size, n)
            fold_data = data.iloc[fold_start:fold_end]

            train_size = int(len(fold_data) * self.train_ratio)
            train_data = fold_data.iloc[:train_size]
            test_data = fold_data.iloc[train_size:]

            if len(train_data) < 60 or len(test_data) < 20:
                continue

            print(f"\n[Walk-Forward] Fold {fold + 1}/{self.n_folds}")
            print(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()}")
            print(f"  Test:  {test_data.index[0].date()} to {test_data.index[-1].date()}")

            # Optimize on training data
            optimizer = GridSearchOptimizer(self.backtester, "sharpe_ratio")
            await optimizer.optimize(train_data, config, max_iterations=50)

            best_params = optimizer.get_best_params()
            if not best_params:
                continue

            # Test on out-of-sample data
            strategy = BollingerRSIConfluenceStrategy(name=f"wf_fold_{fold}", **best_params)
            oos_result = await self.backtester.run(strategy, test_data)

            results.append(
                {
                    "fold": fold,
                    "best_params": best_params,
                    "is_sharpe": optimizer.results[0].sharpe_ratio if optimizer.results else 0,
                    "is_return": optimizer.results[0].total_return if optimizer.results else 0,
                    "oos_sharpe": oos_result.sharpe_ratio,
                    "oos_return": oos_result.total_return,
                    "oos_max_dd": oos_result.max_drawdown,
                    "oos_trades": oos_result.num_trades,
                }
            )

        # Summary statistics
        if results:
            avg_oos_sharpe = np.mean([r["oos_sharpe"] for r in results])
            avg_oos_return = np.mean([r["oos_return"] for r in results])
            avg_oos_dd = np.mean([r["oos_max_dd"] for r in results])

            print("\n" + "=" * 60)
            print("WALK-FORWARD SUMMARY")
            print("=" * 60)
            print(f"Avg OOS Sharpe:     {avg_oos_sharpe:.3f}")
            print(f"Avg OOS Return:     {avg_oos_return:.2f}%")
            print(f"Avg OOS Max DD:     {avg_oos_dd:.2f}%")
            print("=" * 60)

        return {"folds": results}


# =============================================================================
# Main Script
# =============================================================================


async def main():
    """Run backtest and optimization."""
    print("=" * 70)
    print("GS-QUANT STRATEGY BACKTEST & OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    output_dir = Path("backtest_results") / "gs_quant_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data for multiple market regimes
    print("\n[1] Generating Synthetic Market Data...")

    # Regime 1: Mean-reverting (favorable for our strategy)
    data_mean_rev = generate_synthetic_ohlcv(
        n_days=252,
        daily_volatility=0.015,
        trend=0.0001,
        mean_reversion_strength=0.05,
        seed=42,
    )
    print(f"  Mean-reverting data: {len(data_mean_rev)} bars")

    # Regime 2: Trending (challenging for mean reversion)
    data_trending = generate_synthetic_ohlcv(
        n_days=252,
        daily_volatility=0.02,
        trend=0.001,
        mean_reversion_strength=0.01,
        seed=123,
    )
    print(f"  Trending data: {len(data_trending)} bars")

    # Regime 3: Mixed (realistic)
    data_mixed = generate_synthetic_ohlcv(
        n_days=504,
        daily_volatility=0.018,
        trend=0.0003,
        mean_reversion_strength=0.03,
        seed=456,
    )
    print(f"  Mixed data: {len(data_mixed)} bars")

    # Initialize backtester
    backtester = SimpleBacktester(
        initial_capital=100_000,
        position_size=0.1,
        commission=0.001,
        slippage_bps=5,
    )

    # ==========================================================================
    # Single Backtest with Default Parameters
    # ==========================================================================
    print("\n[2] Running Single Backtest (Default Parameters)...")

    strategy = BollingerRSIConfluenceStrategy(name="default_bb_rsi")
    result = await backtester.run(strategy, data_mixed)

    print(f"\n  Strategy: {result.strategy_name}")
    print(f"  Total Return:     {result.total_return:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.2f}%")
    print(f"  Win Rate:         {result.win_rate:.1f}%")
    print(f"  Num Trades:       {result.num_trades}")
    print(f"  Calmar Ratio:     {result.calmar_ratio:.3f}")

    # Save equity curve
    if len(result.equity_curve) > 0:
        result.equity_curve.to_csv(output_dir / "default_equity_curve.csv")

    # ==========================================================================
    # Grid Search Optimization
    # ==========================================================================
    print("\n[3] Running Grid Search Optimization...")

    opt_config = OptimizationConfig(
        bb_periods=[10, 15, 20],
        bb_stds=[1.5, 2.0, 2.5],
        rsi_periods=[10, 14],
        rsi_oversold=[25, 30],
        rsi_overbought=[70, 75],
        min_volatilities=[3.0, 5.0],
    )

    optimizer = GridSearchOptimizer(backtester, optimization_metric="sharpe_ratio")
    await optimizer.optimize(data_mixed, opt_config, max_iterations=100)

    print("\n  Top 5 Parameter Sets (by Sharpe Ratio):")
    summary = optimizer.get_summary(top_n=5)
    print(summary.to_string(index=False))

    # Save results
    summary.to_csv(output_dir / "optimization_results.csv", index=False)

    best_params = optimizer.get_best_params()
    print(f"\n  Best Parameters: {best_params}")

    # ==========================================================================
    # Walk-Forward Analysis
    # ==========================================================================
    print("\n[4] Running Walk-Forward Analysis...")

    wf_analyzer = WalkForwardAnalyzer(backtester, train_ratio=0.7, n_folds=3)
    wf_results = await wf_analyzer.run(data_mixed, opt_config)

    # Save walk-forward results
    with open(output_dir / "walk_forward_results.json", "w") as f:
        json.dump(wf_results, f, indent=2, default=str)

    # ==========================================================================
    # Cross-Regime Testing
    # ==========================================================================
    print("\n[5] Cross-Regime Testing with Best Parameters...")

    if best_params:
        best_strategy = BollingerRSIConfluenceStrategy(name="optimized_bb_rsi", **best_params)

        print("\n  Testing on Mean-Reverting Regime:")
        result_mr = await backtester.run(best_strategy, data_mean_rev)
        print(f"    Sharpe: {result_mr.sharpe_ratio:.3f}, Return: {result_mr.total_return:.2f}%")

        print("\n  Testing on Trending Regime:")
        result_tr = await backtester.run(best_strategy, data_trending)
        print(f"    Sharpe: {result_tr.sharpe_ratio:.3f}, Return: {result_tr.total_return:.2f}%")

        print("\n  Testing on Mixed Regime:")
        result_mx = await backtester.run(best_strategy, data_mixed)
        print(f"    Sharpe: {result_mx.sharpe_ratio:.3f}, Return: {result_mx.total_return:.2f}%")

    # ==========================================================================
    # Final Report
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "default_strategy_result": result.to_dict(),
        "best_params": best_params,
        "optimization_summary": summary.to_dict() if len(summary) > 0 else {},
        "regime_tests": {
            "mean_reverting": result_mr.to_dict() if best_params else {},
            "trending": result_tr.to_dict() if best_params else {},
            "mixed": result_mx.to_dict() if best_params else {},
        },
    }

    with open(output_dir / "final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
