"""
Multi-Strategy Optimizer.

Tests 3 different strategies:
1. RSI Mean Reversion (with strict trend confirmation)
2. Momentum Breakout (buys breakouts, not dips)
3. Trend Following (rides trends using SMA crossover)

Also tests:
- Multiple stop-loss levels: 5%, 10%, 20%, None
- Different trading modes: long-only, short-only, both

This should identify which strategy works best for the Nov-Dec 2024 data.
"""

import asyncio
import logging
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.backtesting.runner import BacktestConfig, BacktestRunner
from ordinis.engines.proofbench.core.config import ExecutionConfig, ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.momentum_breakout import MomentumBreakoutModel
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel
from ordinis.engines.signalcore.models.trend_following import TrendFollowingModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy in ["yfinance", "urllib3", "httpx", "ordinis.engines", "ordinis.audit", "root"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)


class StrategyBacktestRunner(BacktestRunner):
    """Custom backtest runner that accepts any model."""

    def __init__(self, config: BacktestConfig, model):
        super().__init__(config)
        self.custom_model = model

    async def initialize(self):
        """Initialize engines and runners with custom model."""
        signal_config = SignalCoreEngineConfig(
            min_probability=0.0,
            min_score=-999,
            enable_governance=False,
        )
        self.signal_engine = SignalCoreEngine(signal_config)
        await self.signal_engine.initialize()

        self.signal_engine.register_model(self.custom_model)

        execution_config = ExecutionConfig(
            commission_pct=self.config.commission_pct,
            estimated_spread=self.config.slippage_bps / 10000,
        )
        backtest_config = ProofBenchEngineConfig(
            initial_capital=self.config.initial_capital,
            execution_config=execution_config,
            enable_governance=False,
        )
        self.backtest_engine = ProofBenchEngine(backtest_config)
        await self.backtest_engine.initialize()

        from ordinis.backtesting.data_adapter import HistoricalDataLoader
        from ordinis.backtesting.signal_runner import HistoricalSignalRunner, SignalRunnerConfig

        self.data_loader = HistoricalDataLoader()
        self.signal_runner = HistoricalSignalRunner(
            self.signal_engine,
            SignalRunnerConfig(
                resampling_freq=self.config.rebalance_freq,
                ensemble_enabled=False,
            ),
        )


def download_daily_data(
    symbols: list[str], start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    """Download daily market data for multiple symbols."""
    print(f"\n📊 Downloading data for {len(symbols)} symbols ({start_date} to {end_date})...")
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, interval="1d")
            if df.empty:
                print(f"  ⚠ No data for {symbol}")
                continue
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            if len(df) > 30:
                data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} bars")
            else:
                print(f"  ⚠ {symbol}: Only {len(df)} bars, skipping")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
    return data


async def run_single_backtest(
    strategy_name: str,
    model,
    symbol: str,
    data: pd.DataFrame,
    stop_loss: float | None,
) -> dict:
    """Run a single backtest."""
    try:
        # Reset model state
        model.reset_state()

        # Create config
        config = BacktestConfig(
            name=f"{strategy_name}_{symbol}",
            symbols=[symbol],
            start_date=data.index[0].strftime("%Y-%m-%d"),
            end_date=data.index[-1].strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            commission_pct=0.001,
            slippage_bps=5,
            rebalance_freq="1d",
        )

        # Run backtest
        runner = StrategyBacktestRunner(config, model)
        results = await runner.run(data={symbol: data})

        return {
            "total_return": results.total_return,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "num_trades": results.num_trades,
            "win_rate": getattr(results, "win_rate", 0),
        }
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return {
            "total_return": -999,
            "sharpe_ratio": 0,
            "max_drawdown": -100,
            "num_trades": 0,
            "win_rate": 0,
        }


def create_rsi_model(params: dict) -> RSIVolumeReversionModel:
    """Create RSI Mean Reversion model with given params."""
    config = ModelConfig(
        model_id="rsi_reversion",
        model_type="rsi_volume_reversion",
        version="1.0.0",
        parameters={
            "rsi_period": params.get("rsi_period", 14),
            "oversold_threshold": params.get("oversold_threshold", 30),
            "overbought_threshold": params.get("overbought_threshold", 70),
            "volume_mult": params.get("volume_mult", 1.5),
            "trend_filter_period": params.get("trend_filter_period", 50),
            # FORCE trend confirmation to avoid catching falling knives
            "require_trend_for_longs": True,
            "require_trend_for_shorts": True,
            "enable_longs": params.get("enable_longs", True),
            "enable_shorts": params.get("enable_shorts", True),
        },
        min_data_points=60,
    )
    return RSIVolumeReversionModel(config)


def create_momentum_model(params: dict) -> MomentumBreakoutModel:
    """Create Momentum Breakout model with given params."""
    config = ModelConfig(
        model_id="momentum_breakout",
        model_type="momentum_breakout",
        version="1.0.0",
        parameters={
            "breakout_period": params.get("breakout_period", 20),
            "volume_period": params.get("volume_period", 20),
            "volume_mult": params.get("volume_mult", 1.5),
            "trend_filter_period": params.get("trend_filter_period", 50),
            "require_trend": params.get("require_trend", True),
            "enable_longs": params.get("enable_longs", True),
            "enable_shorts": params.get("enable_shorts", True),
        },
        min_data_points=60,
    )
    return MomentumBreakoutModel(config)


def create_trend_model(params: dict) -> TrendFollowingModel:
    """Create Trend Following model with given params."""
    config = ModelConfig(
        model_id="trend_following",
        model_type="trend_following",
        version="1.0.0",
        parameters={
            "fast_period": params.get("fast_period", 10),
            "slow_period": params.get("slow_period", 30),
            "require_momentum": params.get("require_momentum", True),
            "volume_filter": params.get("volume_filter", False),
            "enable_longs": params.get("enable_longs", True),
            "enable_shorts": params.get("enable_shorts", True),
        },
        min_data_points=40,
    )
    return TrendFollowingModel(config)


async def main():
    """Run multi-strategy optimization."""
    print("=" * 70)
    print("🚀 Multi-Strategy Optimizer")
    print("=" * 70)

    # Test symbols - mid-cap tech
    symbols = ["AMD", "COIN", "CRWD", "DKNG", "NET"]

    # Two time periods to compare
    periods = {
        "nov_dec_2024": ("2024-11-01", "2024-12-31"),
        "aug_oct_2024": ("2024-08-01", "2024-10-31"),
    }

    # Stop loss variations
    stop_losses = [0.05, 0.10, 0.20, None]

    # Trading modes
    trading_modes = ["long", "short", "both"]

    all_results = []

    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{'='*70}")
        print(f"📅 Period: {period_name} ({start_date} to {end_date})")
        print("=" * 70)

        # Download data for this period
        market_data = download_daily_data(symbols, start_date, end_date)
        if not market_data:
            print(f"⚠ No data for {period_name}, skipping")
            continue

        # Test each strategy
        strategies = [
            (
                "RSI_TrendFilter",
                create_rsi_model,
                {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
            ),
            (
                "Momentum_Breakout",
                create_momentum_model,
                {"breakout_period": 20, "volume_mult": 1.5},
            ),
            ("Trend_Following", create_trend_model, {"fast_period": 10, "slow_period": 30}),
        ]

        for strategy_name, model_factory, base_params in strategies:
            print(f"\n🔬 Testing {strategy_name}...")

            for symbol, data in market_data.items():
                for stop_loss in stop_losses:
                    for mode in trading_modes:
                        params = base_params.copy()

                        if mode == "long":
                            params["enable_longs"] = True
                            params["enable_shorts"] = False
                        elif mode == "short":
                            params["enable_longs"] = False
                            params["enable_shorts"] = True
                        else:
                            params["enable_longs"] = True
                            params["enable_shorts"] = True

                        model = model_factory(params)
                        result = await run_single_backtest(
                            strategy_name, model, symbol, data, stop_loss
                        )

                        if result["total_return"] > -100:  # Valid result
                            all_results.append(
                                {
                                    "strategy": strategy_name,
                                    "period": period_name,
                                    "symbol": symbol,
                                    "stop_loss": stop_loss if stop_loss else "None",
                                    "mode": mode,
                                    "total_return": result["total_return"],
                                    "num_trades": result["num_trades"],
                                    "win_rate": result["win_rate"],
                                    "max_drawdown": result["max_drawdown"],
                                    "sharpe": result["sharpe_ratio"],
                                }
                            )

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)

    if df.empty:
        print("\n❌ No results to analyze")
        return

    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)

    # Top 20 best overall
    print("\n🏆 TOP 20 BEST PERFORMING CONFIGURATIONS:")
    df_sorted = df.sort_values("total_return", ascending=False)
    top_20 = df_sorted.head(20)
    for _, row in top_20.iterrows():
        ret = row["total_return"] * 100
        color = "🟢" if ret > 0 else "🔴"
        print(
            f"  {color} {row['strategy']:25s} | {row['symbol']:5s} | {row['mode']:5s} | "
            f"SL={row['stop_loss']!s:5s} | {row['period']:15s} | "
            f"Return={ret:+.1f}% | Trades={row['num_trades']}"
        )

    # Summary by strategy
    print("\n📈 AVERAGE RETURN BY STRATEGY:")
    by_strategy = df.groupby("strategy")["total_return"].agg(["mean", "std", "min", "max", "count"])
    for strategy, row in by_strategy.iterrows():
        print(
            f"  {strategy:30s}: Mean={row['mean']*100:+.2f}%, "
            f"Std={row['std']*100:.2f}%, Min={row['min']*100:.1f}%, Max={row['max']*100:.1f}% (n={row['count']:.0f})"
        )

    # Summary by period
    print("\n📅 AVERAGE RETURN BY PERIOD:")
    by_period = df.groupby("period")["total_return"].agg(["mean", "std", "min", "max", "count"])
    for period, row in by_period.iterrows():
        print(
            f"  {period:20s}: Mean={row['mean']*100:+.2f}%, "
            f"Std={row['std']*100:.2f}%, Min={row['min']*100:.1f}%, Max={row['max']*100:.1f}% (n={row['count']:.0f})"
        )

    # Summary by stop loss
    print("\n🛑 AVERAGE RETURN BY STOP-LOSS:")
    by_sl = df.groupby("stop_loss")["total_return"].agg(["mean", "std", "min", "max", "count"])
    for sl, row in by_sl.iterrows():
        print(
            f"  SL={sl!s:6s}: Mean={row['mean']*100:+.2f}%, "
            f"Std={row['std']*100:.2f}%, Min={row['min']*100:.1f}%, Max={row['max']*100:.1f}% (n={row['count']:.0f})"
        )

    # Summary by mode
    print("\n🎯 AVERAGE RETURN BY TRADING MODE:")
    by_mode = df.groupby("mode")["total_return"].agg(["mean", "std", "min", "max", "count"])
    for mode, row in by_mode.iterrows():
        print(
            f"  {mode:6s}: Mean={row['mean']*100:+.2f}%, "
            f"Std={row['std']*100:.2f}%, Min={row['min']*100:.1f}%, Max={row['max']*100:.1f}% (n={row['count']:.0f})"
        )

    # Cross-tab: Strategy x Period
    print("\n📊 STRATEGY vs PERIOD (Mean Return %):")
    pivot = df.pivot_table(
        values="total_return", index="strategy", columns="period", aggfunc="mean"
    )
    for strategy in pivot.index:
        row_str = f"  {strategy:25s}: "
        for period in pivot.columns:
            val = pivot.loc[strategy, period] * 100
            row_str += f"{period}={val:+.2f}%  "
        print(row_str)

    # Find the BEST strategy for each period
    print("\n🥇 BEST CONFIGURATION FOR EACH PERIOD:")
    for period in df["period"].unique():
        period_df = df[df["period"] == period]
        best = period_df.loc[period_df["total_return"].idxmax()]
        print(
            f"  {period}: {best['strategy']} | {best['symbol']} | {best['mode']} | "
            f"SL={best['stop_loss']} | Return={best['total_return']*100:+.1f}%"
        )

    # Positive return count
    positive_count = (df["total_return"] > 0).sum()
    total_count = len(df)
    print(
        f"\n✅ POSITIVE RETURNS: {positive_count} / {total_count} "
        f"({positive_count/total_count*100:.1f}%)"
    )

    # Save full results
    output_path = Path("data/results/multi_strategy_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Full results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
