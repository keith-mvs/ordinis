"""
Strategy Optimization Script.

Runs a grid search over SMA Crossover parameters to find the best configuration.
"""

import asyncio
from datetime import datetime, timedelta
import itertools
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Add scripts dir to path to import data loader
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
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.sma_crossover import SMACrossoverModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimizer")
logging.getLogger("ordinis").setLevel(logging.WARNING)


class CustomBacktestRunner(BacktestRunner):
    def __init__(self, config: BacktestConfig, model_config: ModelConfig):
        super().__init__(config)
        self.custom_model_config = model_config

    async def initialize(self):
        """Initialize engines and runners with custom model."""
        # Signal engine
        signal_config = SignalCoreEngineConfig(
            min_probability=0.5,
            min_score=0.1,
            enable_governance=False,
        )
        self.signal_engine = SignalCoreEngine(signal_config)
        await self.signal_engine.initialize()

        # Register CUSTOM model
        model = SMACrossoverModel(self.custom_model_config)
        self.signal_engine.register_model(model)

        # Backtest engine
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

        # Data loader and signal runner
        self.data_loader = HistoricalDataLoader()
        self.signal_runner = HistoricalSignalRunner(
            self.signal_engine,
            SignalRunnerConfig(
                resampling_freq=self.config.rebalance_freq,
                ensemble_enabled=True,
            ),
        )


async def run_optimization():
    # 1. Data Acquisition (Load once)
    symbol = "NVDA"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    logger.info(f"Loading data for {symbol} from {start_date.date()} to {end_date.date()}")

    data_frames = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join("data/massive", f"{date_str}.csv.gz")
        if os.path.exists(file_path):
            try:
                df = load_massive_data(file_path, symbol)
                if not df.empty:
                    data_frames.append(df)
            except Exception:
                pass
        current_date += timedelta(days=1)

    if not data_frames:
        logger.error("No data available.")
        return

    full_df = pd.concat(data_frames).sort_index()
    logger.info(f"Loaded {len(full_df)} rows.")

    # 2. Define Parameter Grid
    param_grid = {
        "timeframe": ["5min"],  # Stick to 5min for now as 1min is too noisy
        "fast_period": [10, 20],
        "slow_period": [30, 50],
        "trend_filter": [0, 200],
        "stop_loss": [0.05, 0.10],
        "take_profit": [0.0, 0.03],  # 0 means disabled
    }

    keys, values = zip(*param_grid.items(), strict=False)
    combinations = [dict(zip(keys, v, strict=False)) for v in itertools.product(*values)]

    logger.info(f"Starting optimization with {len(combinations)} combinations...")

    results = []

    for i, params in enumerate(combinations):
        # Skip invalid combinations
        if params["fast_period"] >= params["slow_period"]:
            continue

        logger.info(f"Testing [{i+1}/{len(combinations)}]: {params}")

        # Setup Config
        config = BacktestConfig(
            name=f"Opt_{i}",
            symbols=[symbol],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            commission_pct=0.0,
            slippage_bps=1,
            rebalance_freq=params["timeframe"],
            stop_loss_pct=params["stop_loss"],
            take_profit_pct=params["take_profit"],
        )

        model_config = ModelConfig(
            model_id="SMA_Crossover",
            model_type="sma_crossover",
            parameters={
                "fast_period": params["fast_period"],
                "slow_period": params["slow_period"],
                "trend_filter_period": params["trend_filter"],
                "symbol": symbol,
                "exit_on_cross": False,
                "stop_loss_pct": params["stop_loss"],
                "take_profit_pct": params["take_profit"],
            },
        )

        try:
            runner = CustomBacktestRunner(config, model_config)
            # Suppress output for speed
            res = await runner.run(data={symbol: full_df})

            results.append(
                {
                    "params": params,
                    "return": res.total_return,
                    "sharpe": res.sharpe_ratio,
                    "trades": res.num_trades,
                    "drawdown": res.max_drawdown,
                }
            )

            logger.info(f"  -> Return: {res.total_return:.2f}%, Trades: {res.num_trades}")

        except Exception as e:
            logger.error(f"  -> Failed: {e}")

    # 3. Analyze Results
    if not results:
        logger.error("No results generated.")
        return

    # Sort by Return
    sorted_results = sorted(results, key=lambda x: x["return"], reverse=True)
    best = sorted_results[0]

    logger.info("\n" + "=" * 50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Best Return: {best['return']:.2f}%")
    logger.info(f"Parameters: {best['params']}")
    logger.info(f"Trades: {best['trades']}")
    logger.info(f"Sharpe: {best['sharpe']:.2f}")
    logger.info(f"Drawdown: {best['drawdown']:.2f}%")

    # Save to Report
    report_path = os.path.join("artifacts", "reports", "optimization_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Strategy Optimization Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Best Configuration\n")
        f.write(f"- **Return:** {best['return']:.2f}%\n")
        f.write(f"- **Parameters:** `{best['params']}`\n")
        f.write(f"- **Trades:** {best['trades']}\n")
        f.write(f"- **Sharpe:** {best['sharpe']:.2f}\n\n")

        f.write("## Top 5 Configurations\n")
        f.write("| Rank | Return | Trades | Sharpe | Params |\n")
        f.write("|------|--------|--------|--------|--------|\n")
        for i, res in enumerate(sorted_results[:5]):
            f.write(
                f"| {i+1} | {res['return']:.2f}% | {res['trades']} | {res['sharpe']:.2f} | `{res['params']}` |\n"
            )

    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_optimization())
