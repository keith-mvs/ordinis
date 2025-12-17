"""
Multi-Strategy Intraday Optimizer.

Tests 3 strategies with intraday data (1min, 5min, 15min):
1. RSI Mean Reversion - with RELAXED filters to generate trades
2. Momentum Breakout - with RELAXED filters to generate trades
3. Trend Following - with tuned fast/slow periods

Uses existing intraday data from massive_downloader.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data.download_massive import load_massive_data

from ordinis.backtesting.data_adapter import HistoricalDataLoader
from ordinis.backtesting.runner import BacktestConfig, BacktestRunner
from ordinis.backtesting.signal_runner import HistoricalSignalRunner, SignalRunnerConfig
from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.execution import ExecutionConfig
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.models.momentum_breakout import MomentumBreakoutModel
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel
from ordinis.engines.signalcore.models.trend_following import TrendFollowingModel
from ordinis.engines.signalcore.regime_detector import RegimeDetector, regime_filter

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("strategy_optimizer")
logger.setLevel(logging.INFO)

for noisy in ["ordinis", "massive_downloader", "root"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)


MID_CAP_STOCKS = ["AMD", "COIN", "CRWD", "DKNG", "NET"]


class StrategyBacktestRunner(BacktestRunner):
    """Custom backtest runner that accepts any model."""

    def __init__(self, config: BacktestConfig, model: Model):
        super().__init__(config)
        self.custom_model = model

    async def initialize(self):
        """Initialize engines with custom model."""
        signal_config = SignalCoreEngineConfig(
            min_probability=0.0,  # Accept all signals
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

        self.data_loader = HistoricalDataLoader()
        self.signal_runner = HistoricalSignalRunner(
            self.signal_engine,
            SignalRunnerConfig(
                resampling_freq=self.config.rebalance_freq,
                ensemble_enabled=False,
            ),
        )


async def load_intraday_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load intraday data from massive_downloader cache."""
    end_date = datetime.now()
    all_data = []

    for i in range(days):
        date = end_date - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        file_path = f"data/massive/{date_str}.csv.gz"

        if os.path.exists(file_path):
            try:
                df = load_massive_data(file_path, symbol)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception:
                pass

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1min data to desired timeframe."""
    if df.empty:
        return df

    rule_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h"}
    rule = rule_map.get(timeframe, "5min")

    if rule == "1min":
        return df

    resampled = (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    return resampled


def create_rsi_model(params: dict) -> RSIVolumeReversionModel:
    """Create RSI model with given params."""
    config = ModelConfig(
        model_id=f"rsi_{params.get('name', 'default')}",
        model_type="rsi_volume_reversion",
        parameters={
            "rsi_period": params.get("rsi_period", 14),
            "oversold_threshold": params.get("oversold", 30),
            "overbought_threshold": params.get("overbought", 70),
            "volume_mult": params.get("volume_mult", 1.0),  # Relaxed - no volume filter
            "trend_filter_period": params.get("trend_filter", 0),  # Relaxed - no trend
            "require_trend_for_longs": params.get("require_trend", False),
            "require_trend_for_shorts": params.get("require_trend", False),
            "enable_longs": params.get("enable_longs", True),
            "enable_shorts": params.get("enable_shorts", True),
        },
        min_data_points=30,
    )
    return RSIVolumeReversionModel(config)


def create_momentum_model(params: dict) -> MomentumBreakoutModel:
    """Create Momentum model with given params."""
    config = ModelConfig(
        model_id=f"momentum_{params.get('name', 'default')}",
        model_type="momentum_breakout",
        parameters={
            "breakout_period": params.get("breakout_period", 10),  # Shorter lookback
            "volume_period": params.get("volume_period", 20),
            "volume_mult": params.get("volume_mult", 1.0),  # Relaxed
            "trend_filter_period": params.get("trend_filter", 0),  # Relaxed
            "require_trend": params.get("require_trend", False),
            "enable_longs": params.get("enable_longs", True),
            "enable_shorts": params.get("enable_shorts", True),
        },
        min_data_points=30,
    )
    return MomentumBreakoutModel(config)


def create_trend_model(params: dict) -> TrendFollowingModel:
    """Create Trend Following model with given params."""
    config = ModelConfig(
        model_id=f"trend_{params.get('name', 'default')}",
        model_type="trend_following",
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


async def run_single_backtest(
    model: Model, symbol: str, data: pd.DataFrame, timeframe: str
) -> dict:
    """Run a single backtest and return results."""
    try:
        model.reset_state()

        config = BacktestConfig(
            name=f"{model.config.model_id}_{symbol}_{timeframe}",
            symbols=[symbol],
            start_date=data.index[0].strftime("%Y-%m-%d"),
            end_date=data.index[-1].strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            commission_pct=0.001,
            slippage_bps=5,
            rebalance_freq=timeframe,
        )

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
        return {
            "total_return": -999,
            "sharpe_ratio": 0,
            "max_drawdown": -100,
            "num_trades": 0,
            "win_rate": 0,
            "error": str(e),
        }


def generate_strategy_configs():
    """Generate parameter configurations for all strategies."""

    configs = []

    # ===== RSI RELAXED (to generate trades) =====
    rsi_configs = [
        # Very relaxed - should trigger often
        {
            "name": "rsi_relaxed",
            "strategy": "rsi",
            "rsi_period": 7,
            "oversold": 35,
            "overbought": 65,
            "volume_mult": 1.0,
            "trend_filter": 0,
            "require_trend": False,
        },
        # Standard RSI, no filters
        {
            "name": "rsi_standard",
            "strategy": "rsi",
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "volume_mult": 1.0,
            "trend_filter": 0,
            "require_trend": False,
        },
        # With weak trend filter
        {
            "name": "rsi_trend",
            "strategy": "rsi",
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "volume_mult": 1.0,
            "trend_filter": 20,
            "require_trend": True,
        },
    ]

    # ===== MOMENTUM RELAXED =====
    momentum_configs = [
        # Short lookback - more breakouts
        {
            "name": "mom_fast",
            "strategy": "momentum",
            "breakout_period": 5,
            "volume_mult": 1.0,
            "trend_filter": 0,
            "require_trend": False,
        },
        # Medium lookback
        {
            "name": "mom_medium",
            "strategy": "momentum",
            "breakout_period": 10,
            "volume_mult": 1.0,
            "trend_filter": 0,
            "require_trend": False,
        },
        # With weak trend filter
        {
            "name": "mom_trend",
            "strategy": "momentum",
            "breakout_period": 10,
            "volume_mult": 1.2,
            "trend_filter": 20,
            "require_trend": True,
        },
    ]

    # ===== TREND FOLLOWING - TUNE PERIODS =====
    trend_configs = [
        # Very fast - more signals
        {
            "name": "trend_fast",
            "strategy": "trend",
            "fast_period": 5,
            "slow_period": 15,
            "require_momentum": False,
        },
        # Fast
        {
            "name": "trend_quick",
            "strategy": "trend",
            "fast_period": 10,
            "slow_period": 30,
            "require_momentum": True,
        },
        # Standard
        {
            "name": "trend_std",
            "strategy": "trend",
            "fast_period": 20,
            "slow_period": 50,
            "require_momentum": True,
        },
        # Slow - fewer signals, stronger trends
        {
            "name": "trend_slow",
            "strategy": "trend",
            "fast_period": 30,
            "slow_period": 100,
            "require_momentum": True,
        },
    ]

    configs.extend(rsi_configs)
    configs.extend(momentum_configs)
    configs.extend(trend_configs)

    return configs


async def run_optimization(use_regime_filter: bool = False):
    """Run comprehensive multi-strategy optimization.

    Args:
        use_regime_filter: If True, skip incompatible strategy/stock combinations
    """

    logger.info("=" * 80)
    logger.info("🚀 MULTI-STRATEGY INTRADAY OPTIMIZER")
    if use_regime_filter:
        logger.info("🔍 REGIME FILTER: ENABLED")
    logger.info("=" * 80)

    # Get strategy configs
    strategy_configs = generate_strategy_configs()
    timeframes = ["5min", "15min"]  # Skip 1min for speed
    modes = ["long", "short", "both"]

    total_tests = len(strategy_configs) * len(MID_CAP_STOCKS) * len(timeframes) * len(modes)
    logger.info(f"📊 Strategies: {len(strategy_configs)}")
    logger.info(f"📈 Stocks: {MID_CAP_STOCKS}")
    logger.info(f"⏱️  Timeframes: {timeframes}")
    logger.info(f"🎯 Modes: {modes}")
    logger.info(f"📋 Total tests: {total_tests}")

    # Load data for all stocks
    logger.info("\n📥 Loading intraday data (30 days)...")
    stock_data = {}
    for symbol in MID_CAP_STOCKS:
        data = await load_intraday_data(symbol, days=30)
        if not data.empty:
            stock_data[symbol] = data
            logger.info(f"   ✓ {symbol}: {len(data)} bars")
        else:
            logger.info(f"   ⚠ {symbol}: No data")

    if not stock_data:
        logger.error("❌ No data loaded!")
        return

    # Analyze regimes if filtering is enabled
    regime_cache = {}
    if use_regime_filter:
        logger.info("\n🔍 Analyzing market regimes...")
        detector = RegimeDetector()
        for symbol, raw_data in stock_data.items():
            tf_data = resample_data(raw_data.copy(), "5min")
            if len(tf_data) >= 50:
                analysis = detector.analyze(tf_data, symbol, "5min")
                regime_cache[symbol] = analysis
                logger.info(
                    f"   {symbol}: {analysis.regime.value} ({analysis.trade_recommendation})"
                )

    # Run all tests
    all_results = []
    skipped_count = 0
    test_num = 0

    for symbol, raw_data in stock_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 Testing {symbol}")
        logger.info("=" * 60)

        for timeframe in timeframes:
            # Resample data
            tf_data = resample_data(raw_data.copy(), timeframe)
            if len(tf_data) < 50:
                logger.info(f"   ⚠ {timeframe}: Insufficient data ({len(tf_data)} bars)")
                continue

            logger.info(f"\n   ⏱️  Timeframe: {timeframe} ({len(tf_data)} bars)")

            for config in strategy_configs:
                for mode in modes:
                    test_num += 1

                    # Set mode
                    params = config.copy()
                    params["enable_longs"] = mode in ["long", "both"]
                    params["enable_shorts"] = mode in ["short", "both"]

                    strategy = params["strategy"]

                    # Check regime filter
                    if use_regime_filter:
                        should_trade, reason = regime_filter(tf_data, strategy, symbol)
                        if not should_trade:
                            skipped_count += 1
                            all_results.append(
                                {
                                    "strategy": params["name"],
                                    "strategy_type": strategy,
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "mode": mode,
                                    "total_return": 0.0,
                                    "num_trades": 0,
                                    "win_rate": 0.0,
                                    "max_drawdown": 0.0,
                                    "sharpe": 0.0,
                                    "skipped": True,
                                    "skip_reason": reason,
                                }
                            )
                            continue

                    # Create model
                    if strategy == "rsi":
                        model = create_rsi_model(params)
                    elif strategy == "momentum":
                        model = create_momentum_model(params)
                    else:
                        model = create_trend_model(params)

                    # Run backtest
                    result = await run_single_backtest(model, symbol, tf_data, timeframe)

                    if result["total_return"] > -100:
                        all_results.append(
                            {
                                "strategy": params["name"],
                                "strategy_type": strategy,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "mode": mode,
                                "total_return": result["total_return"],
                                "num_trades": result["num_trades"],
                                "win_rate": result["win_rate"],
                                "max_drawdown": result["max_drawdown"],
                                "sharpe": result["sharpe_ratio"],
                                "skipped": False,
                                "skip_reason": "",
                            }
                        )

                    if test_num % 50 == 0:
                        logger.info(
                            f"      Progress: {test_num}/{total_tests} ({test_num/total_tests*100:.1f}%)"
                        )

    if use_regime_filter:
        logger.info(f"\n🔍 Regime filter skipped {skipped_count} incompatible tests")

    # Analyze results
    df = pd.DataFrame(all_results)

    if df.empty:
        logger.error("\n❌ No valid results!")
        return

    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTS SUMMARY")
    logger.info("=" * 80)

    # Filter to results with trades
    df_with_trades = df[df["num_trades"] > 0]

    logger.info(f"\n📋 Total tests: {len(df)}")
    logger.info(f"📈 Tests with trades: {len(df_with_trades)}")

    # Top 20 best
    logger.info("\n🏆 TOP 20 BEST PERFORMING (with trades):")
    if not df_with_trades.empty:
        top_20 = df_with_trades.sort_values("total_return", ascending=False).head(20)
        for _, row in top_20.iterrows():
            ret = row["total_return"] * 100
            color = "🟢" if ret > 0 else "🔴"
            logger.info(
                f"  {color} {row['strategy']:15s} | {row['symbol']:5s} | {row['timeframe']:5s} | "
                f"{row['mode']:5s} | Return={ret:+7.1f}% | Trades={row['num_trades']:3.0f} | "
                f"WR={row['win_rate']*100:.0f}%"
            )

    # By strategy type
    logger.info("\n📈 AVERAGE RETURN BY STRATEGY TYPE:")
    if not df_with_trades.empty:
        by_type = df_with_trades.groupby("strategy_type").agg(
            {"total_return": ["mean", "std", "min", "max", "count"], "num_trades": "mean"}
        )
        for stype in by_type.index:
            mean_ret = by_type.loc[stype, ("total_return", "mean")] * 100
            std_ret = by_type.loc[stype, ("total_return", "std")] * 100
            avg_trades = by_type.loc[stype, ("num_trades", "mean")]
            logger.info(
                f"  {stype:12s}: Mean={mean_ret:+.2f}%, Std={std_ret:.2f}%, AvgTrades={avg_trades:.1f}"
            )

    # By timeframe
    logger.info("\n⏱️  AVERAGE RETURN BY TIMEFRAME:")
    if not df_with_trades.empty:
        by_tf = df_with_trades.groupby("timeframe")["total_return"].agg(["mean", "count"])
        for tf, row in by_tf.iterrows():
            logger.info(f"  {tf:8s}: Mean={row['mean']*100:+.2f}% (n={row['count']:.0f})")

    # By mode
    logger.info("\n🎯 AVERAGE RETURN BY MODE:")
    if not df_with_trades.empty:
        by_mode = df_with_trades.groupby("mode")["total_return"].agg(["mean", "count"])
        for mode, row in by_mode.iterrows():
            logger.info(f"  {mode:8s}: Mean={row['mean']*100:+.2f}% (n={row['count']:.0f})")

    # Positive returns
    if not df_with_trades.empty:
        positive = (df_with_trades["total_return"] > 0).sum()
        total = len(df_with_trades)
        logger.info(f"\n✅ POSITIVE RETURNS: {positive} / {total} ({positive/total*100:.1f}%)")

    # Save results
    suffix = "_regime_filtered" if use_regime_filter else ""
    output_path = Path(f"data/results/multi_strategy_intraday_results{suffix}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Full results saved to: {output_path}")

    # Best per symbol
    logger.info("\n🥇 BEST STRATEGY PER SYMBOL:")
    if not df_with_trades.empty:
        for symbol in df_with_trades["symbol"].unique():
            sym_df = df_with_trades[df_with_trades["symbol"] == symbol]
            if not sym_df.empty:
                best = sym_df.loc[sym_df["total_return"].idxmax()]
                logger.info(
                    f"  {symbol}: {best['strategy']} | {best['timeframe']} | {best['mode']} | "
                    f"Return={best['total_return']*100:+.1f}%"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-strategy optimizer with regime detection")
    parser.add_argument(
        "--regime-filter",
        action="store_true",
        help="Enable regime filtering to skip incompatible strategy/stock combos",
    )
    args = parser.parse_args()

    asyncio.run(run_optimization(use_regime_filter=args.regime_filter))
