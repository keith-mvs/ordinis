"""
RSI Volume Reversion Strategy Optimizer.

Grid search over RSI, Volume, and Risk parameters to find optimal configuration.
This is the "Learning Engine" for the mean reversion strategy.
"""

import asyncio
from datetime import datetime, timedelta
from itertools import product
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data.download_massive import download_day_data, load_massive_data

from ordinis.backtesting.data_adapter import HistoricalDataLoader
from ordinis.backtesting.runner import BacktestConfig, BacktestRunner
from ordinis.backtesting.signal_runner import HistoricalSignalRunner, SignalRunnerConfig
from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.execution import ExecutionConfig
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel

# Configure logging - suppress verbose audit logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rsi_optimizer")
logger.setLevel(logging.INFO)

# Suppress noisy loggers
for noisy_logger in [
    "ordinis.audit.SignalCoreEngine",
    "ordinis.engines.SignalCoreEngine",
    "ordinis.engines.ProofBenchEngine",
    "ordinis.audit",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class RSIBacktestRunner(BacktestRunner):
    """Custom backtest runner for RSI Volume Reversion strategy."""

    def __init__(self, config: BacktestConfig, model_config: ModelConfig):
        super().__init__(config)
        self.custom_model_config = model_config

    async def initialize(self):
        """Initialize engines and runners with custom model."""
        signal_config = SignalCoreEngineConfig(
            min_probability=0.5,
            min_score=0.1,
            enable_governance=False,
        )
        self.signal_engine = SignalCoreEngine(signal_config)
        await self.signal_engine.initialize()

        model = RSIVolumeReversionModel(self.custom_model_config)
        self.signal_engine.register_model(model)

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
                ensemble_enabled=True,
            ),
        )


async def load_data(symbol: str, days: int = 5) -> pd.DataFrame:
    """Load historical data for backtesting."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data_frames = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join("data/massive", f"{date_str}.csv.gz")

        if not os.path.exists(file_path):
            logger.info(f"Downloading data for {date_str}...")
            file_path = download_day_data(date_str)

        if file_path and os.path.exists(file_path):
            try:
                df = load_massive_data(file_path, symbol)
                if not df.empty:
                    data_frames.append(df)
            except Exception:
                pass

        current_date += timedelta(days=1)

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames).sort_index()


async def run_single_backtest(data: pd.DataFrame, symbol: str, params: dict) -> dict:
    """Run a single backtest with given parameters."""
    try:
        config = BacktestConfig(
            name=f"RSI_Opt_{params['rsi_period']}_{params['oversold']}",
            symbols=[symbol],
            start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            commission_pct=0.0,
            slippage_bps=1,
            rebalance_freq=params["timeframe"],
        )

        model_config = ModelConfig(
            model_id="RSI_Volume_Reversion",
            model_type="rsi_volume_reversion",
            parameters={
                "rsi_period": params["rsi_period"],
                "oversold_threshold": params["oversold"],
                "overbought_threshold": 100 - params["oversold"],  # Symmetric
                "exit_rsi": params["exit_rsi"],
                "volume_period": 20,
                "volume_mult": params["volume_mult"],
                "trend_filter_period": params["trend_filter"],
                "stop_loss_pct": params["stop_loss"],
                "take_profit_pct": params["take_profit"],
                "enable_shorts": False,
                "symbol": symbol,
            },
        )

        runner = RSIBacktestRunner(config, model_config)
        results = await runner.run(data={symbol: data})

        return {
            "params": params,
            "total_return": results.total_return,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "num_trades": results.num_trades,
        }
    except Exception as e:
        return {
            "params": params,
            "total_return": -999,
            "sharpe_ratio": 0,
            "max_drawdown": 100,
            "num_trades": 0,
            "error": str(e),
        }


async def run_optimization():
    """Run grid search optimization."""
    symbol = "NVDA"

    logger.info("=" * 70)
    logger.info("🔬 RSI VOLUME REVERSION STRATEGY OPTIMIZER")
    logger.info("=" * 70)

    # Load data once
    logger.info(f"\n📥 Loading data for {symbol}...")
    data = await load_data(symbol, days=5)
    if data.empty:
        logger.error("No data available!")
        return
    logger.info(f"   Loaded {len(data)} rows")

    # ========== PARAMETER GRID ==========
    # Reduced grid for faster iteration (expand once strategy is profitable)
    param_grid = {
        "timeframe": ["1min", "5min"],  # Timeframe
        "rsi_period": [7, 14],  # RSI periods
        "oversold": [30, 35],  # Oversold threshold (relaxed)
        "exit_rsi": [45, 50],  # Exit RSI level
        "volume_mult": [1.0, 1.3],  # Volume multiplier (1.0 = disabled)
        "trend_filter": [0],  # Disabled for simplicity
        "stop_loss": [0.03, 0.05],  # Stop loss %
        "take_profit": [0.0, 0.02],  # Take profit %
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    total = len(combinations)

    logger.info(f"\n🎯 Testing {total} parameter combinations...\n")

    results = []
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo, strict=False))

        # Progress indicator
        if i % 10 == 0 or i == 1:
            logger.info(
                f"   [{i}/{total}] Testing: RSI={params['rsi_period']}, "
                f"Oversold={params['oversold']}, Vol={params['volume_mult']}x, "
                f"SL={params['stop_loss']*100}%"
            )

        result = await run_single_backtest(data.copy(), symbol, params)
        results.append(result)

    # ========== ANALYZE RESULTS ==========
    # Filter valid results
    valid_results = [r for r in results if r["total_return"] > -900]

    # Sort by total return
    sorted_results = sorted(valid_results, key=lambda x: x["total_return"], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("📊 TOP 10 CONFIGURATIONS")
    logger.info("=" * 70)

    for i, r in enumerate(sorted_results[:10], 1):
        p = r["params"]
        logger.info(
            f"\n#{i}: Return: {r['total_return']:.2f}% | "
            f"Sharpe: {r['sharpe_ratio']:.2f} | "
            f"Trades: {r['num_trades']}"
        )
        logger.info(
            f"    TF={p['timeframe']} RSI={p['rsi_period']} "
            f"Oversold={p['oversold']} Exit={p['exit_rsi']} "
            f"Vol={p['volume_mult']}x Trend={p['trend_filter']} "
            f"SL={p['stop_loss']*100}% TP={p['take_profit']*100}%"
        )

    # ========== BEST CONFIGURATION ==========
    if sorted_results:
        best = sorted_results[0]
        bp = best["params"]

        logger.info("\n" + "=" * 70)
        logger.info("🏆 BEST CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Total Return: {best['total_return']:.2f}%")
        logger.info(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {best['max_drawdown']:.2f}%")
        logger.info(f"Total Trades: {best['num_trades']}")
        logger.info("\nParameters:")
        logger.info(f"  Timeframe: {bp['timeframe']}")
        logger.info(f"  RSI Period: {bp['rsi_period']}")
        logger.info(f"  Oversold: < {bp['oversold']}")
        logger.info(f"  Exit RSI: {bp['exit_rsi']}")
        logger.info(f"  Volume Mult: {bp['volume_mult']}x")
        logger.info(f"  Trend Filter: {bp['trend_filter']} SMA")
        logger.info(f"  Stop Loss: {bp['stop_loss']*100}%")
        logger.info(f"  Take Profit: {bp['take_profit']*100}%")

    # ========== SAVE REPORT ==========
    report_path = os.path.join("artifacts", "reports", "rsi_optimization_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# RSI Volume Reversion Optimization Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Symbol:** {symbol}\n")
        f.write(f"**Combinations Tested:** {total}\n\n")

        f.write("## Top 10 Configurations\n\n")
        f.write(
            "| Rank | Return | Sharpe | Trades | TF | RSI | Oversold | Exit | Vol | Trend | SL | TP |\n"
        )
        f.write(
            "|------|--------|--------|--------|-----|-----|----------|------|-----|-------|-------|-------|\n"
        )

        for i, r in enumerate(sorted_results[:10], 1):
            p = r["params"]
            f.write(
                f"| {i} | {r['total_return']:.2f}% | {r['sharpe_ratio']:.2f} | "
                f"{r['num_trades']} | {p['timeframe']} | {p['rsi_period']} | "
                f"{p['oversold']} | {p['exit_rsi']} | {p['volume_mult']}x | "
                f"{p['trend_filter']} | {p['stop_loss']*100}% | {p['take_profit']*100}% |\n"
            )

        if sorted_results:
            best = sorted_results[0]
            bp = best["params"]
            f.write("\n## Best Configuration Details\n\n")
            f.write("```python\n")
            f.write("model_config = ModelConfig(\n")
            f.write("    model_id='RSI_Volume_Reversion',\n")
            f.write("    parameters={\n")
            f.write(f"        'rsi_period': {bp['rsi_period']},\n")
            f.write(f"        'oversold_threshold': {bp['oversold']},\n")
            f.write(f"        'exit_rsi': {bp['exit_rsi']},\n")
            f.write(f"        'volume_mult': {bp['volume_mult']},\n")
            f.write(f"        'trend_filter_period': {bp['trend_filter']},\n")
            f.write(f"        'stop_loss_pct': {bp['stop_loss']},\n")
            f.write(f"        'take_profit_pct': {bp['take_profit']},\n")
            f.write("    }\n")
            f.write(")\n")
            f.write(f"# Rebalance Freq: {bp['timeframe']}\n")
            f.write("```\n")

    logger.info(f"\n📄 Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_optimization())
