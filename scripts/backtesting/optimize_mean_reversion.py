"""
Strategy Optimization Script.

Runs a grid search over Mean Reversion parameters to find the best configuration.
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
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)  # Ensure src is in path
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
from ordinis.engines.signalcore.models.mean_reversion import MeanReversionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimizer")
# logging.getLogger("ordinis").setLevel(logging.WARNING)
logging.getLogger("ordinis.engines.signalcore.models.mean_reversion").setLevel(logging.DEBUG)


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
        model = MeanReversionModel(self.custom_model_config)
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
        logger.error("No data found.")
        return

    full_data = pd.concat(data_frames).sort_index()
    # Ensure unique index
    full_data = full_data[~full_data.index.duplicated(keep="first")]

    logger.info(f"Total data points: {len(full_data)}")

    # 2. Define Parameter Grid
    param_grid = {
        "timeframe": ["1min", "5min"],
        "rsi_period": [14],
        "rsi_oversold": [30, 25],
        "volume_factor": [1.5, 2.0],
        "trend_filter": [200],
        "stop_loss": [0.02, 0.05],
        "take_profit": [0.0, 0.05],  # 0.0 means disabled
    }

    keys, values = zip(*param_grid.items(), strict=False)
    combinations = [dict(zip(keys, v, strict=False)) for v in itertools.product(*values)]

    logger.info(f"Starting optimization with {len(combinations)} combinations...")

    results = []

    for i, params in enumerate(combinations):
        logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")

        # Resample data if needed
        if params["timeframe"] == "1min":
            test_data = full_data
        else:
            test_data = (
                full_data.resample(params["timeframe"])
                .agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
                .dropna()
            )

        if len(test_data) < 300:  # Need enough data for 200 SMA
            logger.warning("Not enough data for this timeframe.")
            continue

        # Configure Model
        model_config = ModelConfig(
            model_id="mean_reversion",
            model_type="mean_reversion",
            parameters={
                "rsi_period": params["rsi_period"],
                "rsi_oversold": params["rsi_oversold"],
                "rsi_overbought": 100 - params["rsi_oversold"],  # Symmetric
                "volume_factor": params["volume_factor"],
                "trend_filter_period": params["trend_filter"],
                "bb_period": 20,
                "bb_std": 2.0,
            },
        )

        # Configure Backtest
        backtest_config = BacktestConfig(
            name=f"opt_{i}",
            symbols=[symbol],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            rebalance_freq=params["timeframe"],
            stop_loss_pct=params["stop_loss"],
            take_profit_pct=params["take_profit"],
        )

        runner = CustomBacktestRunner(backtest_config, model_config)

        try:
            # Run Backtest
            # We need to inject the data directly or mock the loader,
            # but the runner uses HistoricalDataLoader which loads from disk.
            # To avoid reloading/resampling every time, we can hack the runner
            # or just let it run (it might be slow).
            # For this script, let's just use the runner's run() method which expects data on disk.
            # BUT, we already loaded data.
            # Let's override the data_loader.load_data method on the instance.

            await runner.initialize()

            # Pass data directly
            report = await runner.run(data={symbol: test_data})

            metrics = report
            results.append(
                {
                    **params,
                    "total_return": metrics.total_return,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "trades": metrics.num_trades,
                    "win_rate": metrics.win_rate,
                }
            )

            logger.info(
                f"Result: Return={metrics.total_return:.2%}, Trades={metrics.num_trades}, WinRate={metrics.win_rate:.2%}"
            )

        except Exception as e:
            logger.error(f"Error running backtest: {e}")

    # 3. Analyze Results
    if not results:
        logger.error("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_return", ascending=False)

    print("\nOptimization Results (Top 10):")
    print(results_df.head(10).to_string())

    # Save to file
    os.makedirs("artifacts/reports", exist_ok=True)
    results_df.to_csv("artifacts/reports/optimization_results_mean_reversion.csv", index=False)

    best_result = results_df.iloc[0]

    summary = f"""
# Optimization Report (Mean Reversion)

**Best Configuration:**
- Timeframe: {best_result['timeframe']}
- RSI Period: {best_result['rsi_period']}
- RSI Oversold: {best_result['rsi_oversold']}
- Volume Factor: {best_result['volume_factor']}
- Trend Filter: {best_result['trend_filter']}
- Stop Loss: {best_result['stop_loss']}
- Take Profit: {best_result['take_profit']}

**Performance:**
- Total Return: {best_result['total_return']:.2%}
- Sharpe Ratio: {best_result['sharpe_ratio']:.2f}
- Max Drawdown: {best_result['max_drawdown']:.2%}
- Total Trades: {best_result['trades']}
- Win Rate: {best_result['win_rate']:.2%}
"""
    with open("artifacts/reports/optimization_report_mean_reversion.md", "w") as f:
        f.write(summary)

    print("\nReport saved to artifacts/reports/optimization_report_mean_reversion.md")


if __name__ == "__main__":
    asyncio.run(run_optimization())
