"""
Multi-Stock RSI Strategy Optimizer with Edge Case Testing.

Tests RSI overbought shorts and oversold longs across multiple mid-cap stocks
with edge-case parameters from strict to relaxed.

Stocks: 5 mid-cap stocks with different market behaviors
Timeline: 30 days (or available data)
Aggregates: 1min, 5min, 15min

Edge Cases (in-to-out, strict to relaxed):
- STRICT: RSI 20/80, Volume 2.0x, Trend required
- MODERATE: RSI 30/70, Volume 1.5x, No trend filter
- RELAXED: RSI 40/60, Volume 1.0x (disabled), No trend filter
"""

import asyncio
from datetime import datetime, timedelta
from itertools import product
import json
import logging
import os
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
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel

# Configure logging - suppress verbose loggers
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("multi_stock_optimizer")
logger.setLevel(logging.INFO)

for noisy_logger in [
    "ordinis.audit.SignalCoreEngine",
    "ordinis.engines.SignalCoreEngine",
    "ordinis.engines.ProofBenchEngine",
    "ordinis.audit",
    "root",
    "massive_downloader",
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


# ========== MID-CAP STOCKS ==========
# Selected for diversity: Tech, Finance, Healthcare, Consumer, Industrial
MID_CAP_STOCKS = [
    "AMD",  # Tech - volatile, high volume
    "COIN",  # Finance/Crypto - very volatile
    "CRWD",  # Cybersecurity - growth stock
    "DKNG",  # Consumer/Gaming - momentum plays
    "NET",  # Tech/Cloud - growth stock
]

# Alternative if some not available:
BACKUP_STOCKS = ["SQ", "ROKU", "SNAP", "UBER", "LYFT"]


class MultiStockBacktestRunner(BacktestRunner):
    """Custom backtest runner for multi-stock optimization."""

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


async def load_multi_day_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load historical data for backtesting across multiple days.

    Only loads data that already exists locally - does not attempt downloads.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data_frames = []
    current_date = start_date
    loaded_days = 0

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join("data/massive", f"{date_str}.csv.gz")

        # Only load if file exists locally - skip downloads to avoid errors
        if os.path.exists(file_path):
            try:
                df = load_massive_data(file_path, symbol)
                if df is not None and not df.empty:
                    data_frames.append(df)
                    loaded_days += 1
            except Exception:
                pass

        current_date += timedelta(days=1)

    if not data_frames:
        return pd.DataFrame()

    combined = pd.concat(data_frames).sort_index()
    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample minute data to different timeframes."""
    if timeframe == "1min":
        return data

    tf_map = {
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1h": "1h",
    }

    if timeframe not in tf_map:
        return data

    resampled = (
        data.resample(tf_map[timeframe])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    return resampled


async def run_single_backtest(
    data: pd.DataFrame, symbol: str, params: dict, timeframe: str
) -> dict:
    """Run a single backtest with given parameters."""
    try:
        # Resample data to desired timeframe
        tf_data = resample_data(data.copy(), timeframe)

        if len(tf_data) < 50:
            return {
                "params": params,
                "symbol": symbol,
                "timeframe": timeframe,
                "total_return": -999,
                "error": "Insufficient data after resampling",
            }

        config = BacktestConfig(
            name=f"RSI_{symbol}_{timeframe}_{params['mode']}",
            symbols=[symbol],
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            initial_capital=100000.0,
            commission_pct=0.001,  # 0.1% commission
            slippage_bps=5,  # 5 bps slippage
            rebalance_freq=timeframe,
        )

        # Determine mode-based settings
        enable_longs = params["mode"] in ["long", "both"]
        enable_shorts = params["mode"] in ["short", "both"]

        model_config = ModelConfig(
            model_id=f"RSI_{params['edge_case']}_{params['mode']}",
            model_type="rsi_volume_reversion",
            parameters={
                "rsi_period": params["rsi_period"],
                "oversold_threshold": params["oversold"],
                "overbought_threshold": params["overbought"],
                "exit_rsi": params["exit_rsi"],
                "volume_period": 20,
                "volume_mult": params["volume_mult"],
                "trend_filter_period": params["trend_filter"],
                "enable_longs": enable_longs,
                "enable_shorts": enable_shorts,
                "require_trend_for_longs": params["require_trend"],
                "require_trend_for_shorts": params["require_trend"],
                "stop_loss_pct": params["stop_loss"],
                "take_profit_pct": params["take_profit"],
                "symbol": symbol,
            },
        )

        runner = MultiStockBacktestRunner(config, model_config)
        results = await runner.run(data={symbol: tf_data})

        return {
            "params": params,
            "symbol": symbol,
            "timeframe": timeframe,
            "total_return": results.total_return,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "num_trades": results.num_trades,
            "win_rate": getattr(results, "win_rate", 0),
        }
    except Exception as e:
        return {
            "params": params,
            "symbol": symbol,
            "timeframe": timeframe,
            "total_return": -999,
            "sharpe_ratio": 0,
            "max_drawdown": 100,
            "num_trades": 0,
            "error": str(e),
        }


def generate_edge_case_grid():
    """
    Generate parameter grid from strict (edge) to relaxed (out).

    Edge cases test extremes:
    - STRICT: Very selective, high confidence entries
    - MODERATE: Balanced approach
    - RELAXED: More trades, lower selectivity
    """

    edge_cases = {
        # ===== STRICT (Edge) =====
        "strict": {
            "rsi_period": [7],
            "oversold": [20],  # Very oversold
            "overbought": [80],  # Very overbought
            "exit_rsi": [50],
            "volume_mult": [2.0],  # High volume required
            "trend_filter": [50],  # Trend filter enabled
            "require_trend": [True],
            "stop_loss": [0.02],  # Tight stop
            "take_profit": [0.02],  # Quick profit
        },
        # ===== MODERATE =====
        "moderate": {
            "rsi_period": [7, 14],
            "oversold": [30],  # Standard oversold
            "overbought": [70],  # Standard overbought
            "exit_rsi": [45, 50, 55],
            "volume_mult": [1.5],  # Moderate volume
            "trend_filter": [0],  # No trend filter
            "require_trend": [False],
            "stop_loss": [0.03],
            "take_profit": [0.0, 0.03],  # No TP or 3%
        },
        # ===== RELAXED (Out) =====
        "relaxed": {
            "rsi_period": [7, 14],
            "oversold": [35, 40],  # More relaxed
            "overbought": [60, 65],  # More relaxed
            "exit_rsi": [50],
            "volume_mult": [1.0],  # Volume disabled
            "trend_filter": [0],  # No trend filter
            "require_trend": [False],
            "stop_loss": [0.05],  # Wider stop
            "take_profit": [0.0],  # Let winners run
        },
    }

    # Trading modes to test
    modes = ["short", "long", "both"]

    # Timeframes to test
    timeframes = ["1min", "5min", "15min"]

    all_combinations = []

    for edge_case, params in edge_cases.items():
        keys = list(params.keys())
        for combo in product(*[params[k] for k in keys]):
            param_dict = dict(zip(keys, combo, strict=False))
            param_dict["edge_case"] = edge_case

            for mode in modes:
                param_dict_copy = param_dict.copy()
                param_dict_copy["mode"] = mode
                all_combinations.append(param_dict_copy)

    return all_combinations, timeframes


async def run_optimization():
    """Run comprehensive multi-stock optimization."""

    logger.info("=" * 80)
    logger.info("🔬 MULTI-STOCK RSI STRATEGY OPTIMIZER - EDGE CASE TESTING")
    logger.info("=" * 80)
    logger.info("")

    # Generate parameter grid
    param_grid, timeframes = generate_edge_case_grid()

    logger.info(f"📊 Parameter combinations per stock/timeframe: {len(param_grid)}")
    logger.info(f"📈 Stocks to test: {MID_CAP_STOCKS}")
    logger.info(f"⏱️  Timeframes: {timeframes}")
    logger.info("")

    # Load data for all stocks
    logger.info("📥 Loading historical data (30 days)...")
    stock_data = {}

    for symbol in MID_CAP_STOCKS:
        logger.info(f"   Loading {symbol}...")
        data = await load_multi_day_data(symbol, days=30)
        if not data.empty:
            stock_data[symbol] = data
            logger.info(f"   ✓ {symbol}: {len(data)} rows")
        else:
            logger.info(f"   ✗ {symbol}: No data available")

    # Use backup stocks if needed
    if len(stock_data) < 3:
        logger.info("   Loading backup stocks...")
        for symbol in BACKUP_STOCKS:
            if len(stock_data) >= 5:
                break
            if symbol not in stock_data:
                data = await load_multi_day_data(symbol, days=30)
                if not data.empty:
                    stock_data[symbol] = data
                    logger.info(f"   ✓ {symbol}: {len(data)} rows")

    if not stock_data:
        logger.error("No data available for any stocks!")
        return

    logger.info(f"\n✓ Loaded data for {len(stock_data)} stocks")

    # Calculate total combinations
    total_combos = len(param_grid) * len(stock_data) * len(timeframes)
    logger.info(f"\n🎯 Total test combinations: {total_combos}")
    logger.info("   (This may take a while...)\n")

    # Run backtests
    all_results = []
    completed = 0

    for symbol, data in stock_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 Testing {symbol}")
        logger.info(f"{'='*60}")

        for timeframe in timeframes:
            logger.info(f"\n   ⏱️  Timeframe: {timeframe}")

            for i, params in enumerate(param_grid):
                result = await run_single_backtest(data, symbol, params, timeframe)
                all_results.append(result)
                completed += 1

                # Progress update every 50 tests
                if completed % 50 == 0:
                    pct = (completed / total_combos) * 100
                    logger.info(f"      Progress: {completed}/{total_combos} ({pct:.1f}%)")

    # ========== ANALYZE RESULTS ==========
    logger.info("\n" + "=" * 80)
    logger.info("📊 ANALYSIS RESULTS")
    logger.info("=" * 80)

    # Filter valid results with trades
    valid_results = [r for r in all_results if r.get("total_return", -999) > -900]
    traded_results = [r for r in valid_results if r.get("num_trades", 0) > 0]

    logger.info(f"\nTotal tests: {len(all_results)}")
    logger.info(f"Valid results: {len(valid_results)}")
    logger.info(f"Results with trades: {len(traded_results)}")

    # ===== RESULTS BY EDGE CASE =====
    logger.info("\n" + "-" * 60)
    logger.info("📊 RESULTS BY EDGE CASE (Strict → Relaxed)")
    logger.info("-" * 60)

    for edge_case in ["strict", "moderate", "relaxed"]:
        ec_results = [
            r for r in traded_results if r.get("params", {}).get("edge_case") == edge_case
        ]
        if ec_results:
            avg_return = sum(r["total_return"] for r in ec_results) / len(ec_results)
            avg_trades = sum(r["num_trades"] for r in ec_results) / len(ec_results)
            best = max(ec_results, key=lambda x: x["total_return"])
            logger.info(f"\n{edge_case.upper()}:")
            logger.info(f"   Tests with trades: {len(ec_results)}")
            logger.info(f"   Avg Return: {avg_return:.2f}%")
            logger.info(f"   Avg Trades: {avg_trades:.1f}")
            logger.info(
                f"   Best: {best['total_return']:.2f}% ({best['symbol']}, {best['timeframe']})"
            )

    # ===== RESULTS BY MODE =====
    logger.info("\n" + "-" * 60)
    logger.info("📊 RESULTS BY TRADING MODE")
    logger.info("-" * 60)

    for mode in ["short", "long", "both"]:
        mode_results = [r for r in traded_results if r.get("params", {}).get("mode") == mode]
        if mode_results:
            avg_return = sum(r["total_return"] for r in mode_results) / len(mode_results)
            best = max(mode_results, key=lambda x: x["total_return"])
            logger.info(f"\n{mode.upper()} MODE:")
            logger.info(f"   Tests with trades: {len(mode_results)}")
            logger.info(f"   Avg Return: {avg_return:.2f}%")
            logger.info(
                f"   Best: {best['total_return']:.2f}% ({best['symbol']}, {best['timeframe']})"
            )

    # ===== RESULTS BY STOCK =====
    logger.info("\n" + "-" * 60)
    logger.info("📊 RESULTS BY STOCK")
    logger.info("-" * 60)

    for symbol in stock_data:
        stock_results = [r for r in traded_results if r.get("symbol") == symbol]
        if stock_results:
            avg_return = sum(r["total_return"] for r in stock_results) / len(stock_results)
            best = max(stock_results, key=lambda x: x["total_return"])
            logger.info(f"\n{symbol}:")
            logger.info(f"   Tests with trades: {len(stock_results)}")
            logger.info(f"   Avg Return: {avg_return:.2f}%")
            logger.info(
                f"   Best: {best['total_return']:.2f}% ({best['params'].get('edge_case')}, {best['timeframe']})"
            )

    # ===== TOP 20 CONFIGURATIONS =====
    sorted_results = sorted(traded_results, key=lambda x: x["total_return"], reverse=True)

    logger.info("\n" + "=" * 80)
    logger.info("🏆 TOP 20 CONFIGURATIONS")
    logger.info("=" * 80)

    for i, r in enumerate(sorted_results[:20], 1):
        p = r["params"]
        logger.info(
            f"\n#{i}: {r['symbol']} | {r['timeframe']} | Return: {r['total_return']:.2f}% | "
            f"Sharpe: {r.get('sharpe_ratio', 0):.2f} | Trades: {r['num_trades']}"
        )
        logger.info(
            f"    Edge: {p['edge_case']} | Mode: {p['mode']} | "
            f"RSI: {p['oversold']}/{p['overbought']} | Exit: {p['exit_rsi']} | "
            f"Vol: {p['volume_mult']}x | SL: {p['stop_loss']*100}%"
        )

    # ===== BEST OVERALL =====
    if sorted_results:
        best = sorted_results[0]
        bp = best["params"]

        logger.info("\n" + "=" * 80)
        logger.info("🥇 BEST CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Symbol: {best['symbol']}")
        logger.info(f"Timeframe: {best['timeframe']}")
        logger.info(f"Total Return: {best['total_return']:.2f}%")
        logger.info(f"Sharpe Ratio: {best.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {best.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total Trades: {best['num_trades']}")
        logger.info(f"\nEdge Case: {bp['edge_case']}")
        logger.info(f"Trading Mode: {bp['mode']}")
        logger.info("\nParameters:")
        logger.info(f"  RSI Period: {bp['rsi_period']}")
        logger.info(f"  Oversold: < {bp['oversold']}")
        logger.info(f"  Overbought: > {bp['overbought']}")
        logger.info(f"  Exit RSI: {bp['exit_rsi']}")
        logger.info(f"  Volume Mult: {bp['volume_mult']}x")
        logger.info(f"  Trend Filter: {bp['trend_filter']} SMA")
        logger.info(f"  Require Trend: {bp['require_trend']}")
        logger.info(f"  Stop Loss: {bp['stop_loss']*100}%")
        logger.info(f"  Take Profit: {bp['take_profit']*100}%")

    # ===== SAVE DETAILED REPORT =====
    report_path = os.path.join("artifacts", "reports", "multi_stock_optimization.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Multi-Stock RSI Optimization Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Stocks Tested:** {list(stock_data.keys())}\n")
        f.write(f"**Timeframes:** {timeframes}\n")
        f.write(f"**Total Combinations:** {total_combos}\n")
        f.write(f"**Results with Trades:** {len(traded_results)}\n\n")

        f.write("## Edge Case Summary\n\n")
        f.write("| Edge Case | Tests w/ Trades | Avg Return | Avg Trades | Best Return |\n")
        f.write("|-----------|-----------------|------------|------------|-------------|\n")

        for edge_case in ["strict", "moderate", "relaxed"]:
            ec_results = [
                r for r in traded_results if r.get("params", {}).get("edge_case") == edge_case
            ]
            if ec_results:
                avg_return = sum(r["total_return"] for r in ec_results) / len(ec_results)
                avg_trades = sum(r["num_trades"] for r in ec_results) / len(ec_results)
                best_return = max(r["total_return"] for r in ec_results)
                f.write(
                    f"| {edge_case} | {len(ec_results)} | {avg_return:.2f}% | {avg_trades:.1f} | {best_return:.2f}% |\n"
                )

        f.write("\n## Top 20 Configurations\n\n")
        f.write("| Rank | Symbol | TF | Return | Sharpe | Trades | Edge | Mode | RSI | Vol |\n")
        f.write("|------|--------|-----|--------|--------|--------|------|------|-----|-----|\n")

        for i, r in enumerate(sorted_results[:20], 1):
            p = r["params"]
            f.write(
                f"| {i} | {r['symbol']} | {r['timeframe']} | {r['total_return']:.2f}% | "
                f"{r.get('sharpe_ratio', 0):.2f} | {r['num_trades']} | {p['edge_case']} | "
                f"{p['mode']} | {p['oversold']}/{p['overbought']} | {p['volume_mult']}x |\n"
            )

        if sorted_results:
            best = sorted_results[0]
            bp = best["params"]
            f.write("\n## Best Configuration Code\n\n")
            f.write("```python\n")
            f.write("model_config = ModelConfig(\n")
            f.write(f"    model_id='RSI_{bp['edge_case']}_{bp['mode']}',\n")
            f.write("    parameters={\n")
            f.write(f"        'rsi_period': {bp['rsi_period']},\n")
            f.write(f"        'oversold_threshold': {bp['oversold']},\n")
            f.write(f"        'overbought_threshold': {bp['overbought']},\n")
            f.write(f"        'exit_rsi': {bp['exit_rsi']},\n")
            f.write(f"        'volume_mult': {bp['volume_mult']},\n")
            f.write(f"        'trend_filter_period': {bp['trend_filter']},\n")
            f.write(f"        'require_trend_for_longs': {bp['require_trend']},\n")
            f.write(f"        'require_trend_for_shorts': {bp['require_trend']},\n")
            f.write(f"        'enable_longs': {bp['mode'] in ['long', 'both']},\n")
            f.write(f"        'enable_shorts': {bp['mode'] in ['short', 'both']},\n")
            f.write(f"        'stop_loss_pct': {bp['stop_loss']},\n")
            f.write(f"        'take_profit_pct': {bp['take_profit']},\n")
            f.write("    }\n")
            f.write(")\n")
            f.write(f"# Best on: {best['symbol']} with {best['timeframe']} timeframe\n")
            f.write("```\n")

    # Save JSON results for further analysis
    json_path = os.path.join("artifacts", "reports", "multi_stock_results.json")
    with open(json_path, "w") as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in sorted_results[:100]:
            json_results.append(
                {
                    "symbol": r["symbol"],
                    "timeframe": r["timeframe"],
                    "total_return": r["total_return"],
                    "sharpe_ratio": r.get("sharpe_ratio", 0),
                    "max_drawdown": r.get("max_drawdown", 0),
                    "num_trades": r["num_trades"],
                    "params": r["params"],
                }
            )
        json.dump(json_results, f, indent=2)

    logger.info("\n📄 Reports saved to:")
    logger.info(f"   - {report_path}")
    logger.info(f"   - {json_path}")


if __name__ == "__main__":
    asyncio.run(run_optimization())
