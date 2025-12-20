"""
Intraday Backtest Script using Massive Data (S3).

This script runs the SMA Crossover strategy on 1-minute aggregate data
downloaded from Massive's S3 bucket.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Add scripts dir to path to import data loader
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
from ordinis.engines.signalcore.models.sma_crossover import SMACrossoverModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intraday_backtest")
logging.getLogger("ordinis.audit").setLevel(logging.WARNING)
logging.getLogger("ordinis.engines.SignalCoreEngine").setLevel(logging.WARNING)
logging.getLogger("ordinis.engines.ProofBenchEngine").setLevel(logging.WARNING)


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
        logger.info(f"Registered custom model: {self.custom_model_config.model_id}")

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


async def run_backtest():
    # 1. Configuration
    symbol = "NVDA"
    # Backtest over the last 5 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    logger.info(f"Starting backtest for {symbol} from {start_date.date()} to {end_date.date()}")

    # 2. Data Acquisition
    # Download data for each day in the range
    data_frames = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        # Check if file exists, if not download
        file_path = os.path.join("data/massive", f"{date_str}.csv.gz")
        if not os.path.exists(file_path):
            logger.info(f"Downloading data for {date_str}...")
            # Note: This requires MASSIVE_FF_ACC_KEY_ID and MASSIVE_FF_SECRET_KEY env vars
            file_path = download_day_data(date_str)

        if file_path and os.path.exists(file_path):
            try:
                df = load_massive_data(file_path, symbol)
                if not df.empty:
                    data_frames.append(df)
            except Exception as e:
                logger.error(f"Failed to load data for {date_str}: {e}")

        current_date += timedelta(days=1)

    if not data_frames:
        logger.error("No data available for backtest.")
        return

    # Combine all days
    full_df = pd.concat(data_frames).sort_index()
    logger.info(f"Loaded {len(full_df)} rows of data.")
    logger.info(f"Data Head:\n{full_df.head()}")
    logger.info(f"Data Tail:\n{full_df.tail()}")
    logger.info(f"Data Describe:\n{full_df.describe()}")

    # 3. Setup Backtest
    config = BacktestConfig(
        name="Massive_Intraday_SMA_Optimized",
        symbols=[symbol],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=100000.0,
        commission_pct=0.0,  # Zero commission for simplicity
        slippage_bps=1,  # 1 bps slippage
        rebalance_freq="5min",  # Tuned: 5min timeframe
    )

    # Configure Strategy
    model_config = ModelConfig(
        model_id="SMA_Crossover",
        model_type="sma_crossover",
        parameters={
            "fast_period": 20,  # Tuned: 20 periods (5min * 20 = 100min)
            "slow_period": 50,  # Tuned: 50 periods (5min * 50 = 250min)
            "trend_filter_period": 200,  # Tuned: 200 periods (5min * 200 = 1000min)
            "symbol": symbol,
            "exit_on_cross": False,  # Force SHORT signal for exit to trigger trade logic
            "stop_loss_pct": 0.10,  # 10% Stop Loss
        },
    )

    # 4. Run Backtest
    runner = CustomBacktestRunner(config, model_config)

    logger.info("Running backtest...")
    # Pass data directly to run()
    results = await runner.run(data={symbol: full_df})

    # 5. Report
    logger.info("Backtest Complete.")
    logger.info(f"Total Return: {results.total_return:.2f}%")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2f}%")
    logger.info(f"Trades: {results.num_trades}")

    # Generate Markdown Report
    report_path = os.path.join("artifacts", "reports", "intraday_backtest_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"# Intraday Backtest Report: {config.name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Symbol:** {symbol}\n")
        f.write(f"**Period:** {start_date.date()} to {end_date.date()}\n\n")

        f.write("## Strategy Configuration\n")
        f.write(f"- **Model:** {model_config.model_id}\n")
        f.write(f"- **Timeframe:** {config.rebalance_freq}\n")
        f.write(f"- **Fast SMA:** {model_config.parameters['fast_period']}\n")
        f.write(f"- **Slow SMA:** {model_config.parameters['slow_period']}\n")
        f.write(
            f"- **Trend Filter:** {model_config.parameters.get('trend_filter_period', 'Disabled')}\n"
        )
        f.write(f"- **Stop Loss:** {model_config.parameters.get('stop_loss_pct', 'N/A')}\n\n")

        f.write("## Performance Metrics\n")
        f.write(f"- **Total Return:** {results.total_return:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {results.sharpe_ratio:.2f}\n")
        f.write(f"- **Max Drawdown:** {results.max_drawdown:.2f}%\n")
        f.write(f"- **Total Trades:** {results.num_trades}\n")

    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_backtest())
