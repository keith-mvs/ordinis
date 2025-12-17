"""
RSI Volume Reversion Backtest Script.

This script runs the RSI Mean Reversion strategy with Volume Confirmation
on intraday data. Designed to address the 0% win rate from SMA crossover
by using mean reversion logic which works better in range-bound markets.

Strategy Logic:
- LONG: RSI < 30 AND Volume > 150% avg AND Price > 50 SMA (uptrend)
- EXIT: RSI > 50 (mean reversion target)
- Risk: Stop Loss at configurable %
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scripts.data.download_massive import download_day_data, load_massive_data

from ordinis.backtesting.data_adapter import HistoricalDataLoader
from ordinis.backtesting.runner import BacktestConfig, BacktestRunner  # type: ignore
from ordinis.backtesting.signal_runner import HistoricalSignalRunner, SignalRunnerConfig
from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.execution import ExecutionConfig
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rsi_volume_backtest")
logging.getLogger("ordinis.audit").setLevel(logging.WARNING)
logging.getLogger("ordinis.engines.SignalCoreEngine").setLevel(logging.WARNING)
logging.getLogger("ordinis.engines.ProofBenchEngine").setLevel(logging.WARNING)


class RSIBacktestRunner(BacktestRunner):
    """Custom backtest runner for RSI Volume Reversion strategy."""

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

        # Register RSI Volume Reversion model
        model = RSIVolumeReversionModel(self.custom_model_config)
        self.signal_engine.register_model(model)
        logger.info(f"Registered model: {self.custom_model_config.model_id}")

        # Backtest engine with risk management
        stop_loss_pct = self.custom_model_config.parameters.get("stop_loss_pct", 0.05)
        take_profit_pct = self.custom_model_config.parameters.get("take_profit_pct", 0.0)

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
    """Run the RSI Volume Reversion backtest."""

    # ========== CONFIGURATION ==========
    symbol = "NVDA"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    logger.info("=" * 60)
    logger.info("RSI VOLUME REVERSION BACKTEST")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 60)

    # ========== DATA ACQUISITION ==========
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
            except Exception as e:
                logger.error(f"Failed to load data for {date_str}: {e}")

        current_date += timedelta(days=1)

    if not data_frames:
        logger.error("No data available for backtest.")
        return

    # Combine all days
    full_df = pd.concat(data_frames).sort_index()
    logger.info(f"Loaded {len(full_df)} rows of data.")

    # ========== STRATEGY PARAMETERS ==========
    # These are the tuned parameters for RSI Volume Reversion
    config = BacktestConfig(
        name="RSI_Volume_Reversion_NVDA",
        symbols=[symbol],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=100000.0,
        commission_pct=0.0,
        slippage_bps=1,
        rebalance_freq="1min",  # Using 1-min for more responsive mean reversion
    )

    model_config = ModelConfig(
        model_id="RSI_Volume_Reversion",
        model_type="rsi_volume_reversion",
        parameters={
            # RSI Parameters
            "rsi_period": 14,
            "oversold_threshold": 30,  # Buy when RSI < 30
            "overbought_threshold": 70,
            "exit_rsi": 50,  # Exit when RSI reverts to 50
            # Volume Confirmation
            "volume_period": 20,
            "volume_mult": 1.5,  # Volume must be 150% of average
            # Trend Filter
            "trend_filter_period": 50,  # 50-period SMA trend filter
            # Risk Management
            "stop_loss_pct": 0.03,  # 3% stop loss (tighter for mean reversion)
            "take_profit_pct": 0.02,  # 2% take profit (quick scalps)
            # Other
            "enable_shorts": False,  # Long only for safety
            "symbol": symbol,
        },
    )

    logger.info("\n📊 Strategy Configuration:")
    logger.info(f"  RSI Period: {model_config.parameters['rsi_period']}")
    logger.info(f"  Oversold Threshold: {model_config.parameters['oversold_threshold']}")
    logger.info(f"  Exit RSI: {model_config.parameters['exit_rsi']}")
    logger.info(f"  Volume Multiplier: {model_config.parameters['volume_mult']}x")
    logger.info(f"  Trend Filter: {model_config.parameters['trend_filter_period']} SMA")
    logger.info(f"  Stop Loss: {model_config.parameters['stop_loss_pct']*100}%")
    logger.info(f"  Take Profit: {model_config.parameters['take_profit_pct']*100}%")

    # ========== RUN BACKTEST ==========
    runner = RSIBacktestRunner(config, model_config)

    logger.info("\n🚀 Running backtest...")
    results = await runner.run(data={symbol: full_df})

    # ========== RESULTS ==========
    logger.info("\n" + "=" * 60)
    logger.info("📈 BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {results.total_return:.2f}%")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2f}%")
    logger.info(f"Total Trades: {results.num_trades}")

    # Calculate win rate if we have trade details
    if hasattr(results, "trades") and results.trades:
        winners = sum(1 for t in results.trades if t.get("pnl", 0) > 0)
        win_rate = (winners / len(results.trades)) * 100 if results.trades else 0
        logger.info(f"Win Rate: {win_rate:.1f}%")

    # ========== GENERATE REPORT ==========
    report_path = os.path.join("artifacts", "reports", "rsi_volume_backtest_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# RSI Volume Reversion Backtest Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Symbol:** {symbol}\n")
        f.write(f"**Period:** {start_date.date()} to {end_date.date()}\n\n")

        f.write("## Strategy Logic\n\n")
        f.write("This strategy uses **mean reversion** instead of trend following:\n\n")
        f.write("**Entry Conditions (LONG):**\n")
        f.write("1. RSI < 30 (oversold)\n")
        f.write("2. Volume > 150% of 20-period average (volume spike)\n")
        f.write("3. Price > 50 SMA (uptrend confirmation)\n\n")
        f.write("**Exit Conditions:**\n")
        f.write("1. RSI > 50 (mean reversion target)\n")
        f.write("2. Stop Loss: 3%\n")
        f.write("3. Take Profit: 2%\n\n")

        f.write("## Strategy Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Model | {model_config.model_id} |\n")
        f.write(f"| Timeframe | {config.rebalance_freq} |\n")
        f.write(f"| RSI Period | {model_config.parameters['rsi_period']} |\n")
        f.write(f"| Oversold | < {model_config.parameters['oversold_threshold']} |\n")
        f.write(f"| Exit RSI | {model_config.parameters['exit_rsi']} |\n")
        f.write(f"| Volume Mult | {model_config.parameters['volume_mult']}x |\n")
        f.write(f"| Trend Filter | {model_config.parameters['trend_filter_period']} SMA |\n")
        f.write(f"| Stop Loss | {model_config.parameters['stop_loss_pct']*100}% |\n")
        f.write(f"| Take Profit | {model_config.parameters['take_profit_pct']*100}% |\n\n")

        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Return | {results.total_return:.2f}% |\n")
        f.write(f"| Sharpe Ratio | {results.sharpe_ratio:.2f} |\n")
        f.write(f"| Max Drawdown | {results.max_drawdown:.2f}% |\n")
        f.write(f"| Total Trades | {results.num_trades} |\n")

    logger.info(f"\n📄 Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_backtest())
