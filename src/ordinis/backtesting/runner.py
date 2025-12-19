"""High-level backtest runner integrating all components."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.execution import (
    ExecutionConfig,
    Order,
    OrderSide,
    OrderType,
)
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.sma_crossover import SMACrossoverModel

from .data_adapter import DataAdapter, HistoricalDataLoader
from .metrics import BacktestMetrics
from .signal_runner import HistoricalSignalRunner, SignalRunnerConfig


@dataclass
class BacktestConfig:
    """Configuration for a complete backtest run.

    Attributes:
        name: Backtest name for reporting
        symbols: Symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        commission_pct: Commission as % of trade value
        slippage_bps: Slippage in basis points
        max_position_size: Max notional per symbol
        max_portfolio_exposure: Max total exposure
        rebalance_freq: Rebalance frequency (daily, weekly, etc.)
        stop_loss_pct: Stop loss percentage (e.g. 0.05 for 5%)
        take_profit_pct: Take profit percentage (e.g. 0.10 for 10%)
    """

    name: str
    symbols: list[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_bps: int = 5
    max_position_size: float = 0.1
    max_portfolio_exposure: float = 1.0
    rebalance_freq: str = "1d"
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0


class BacktestRunner:
    """End-to-end backtest runner."""

    def __init__(self, config: BacktestConfig, output_dir: Path | None = None):
        """Initialize runner.

        Args:
            config: Backtest configuration
            output_dir: Directory for artifacts (signals, trades, report)
        """
        self.config = config
        self.output_dir = output_dir or Path("backtest_results") / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Engines
        self.signal_engine: SignalCoreEngine | None = None
        self.backtest_engine: ProofBenchEngine | None = None

        # Data and runners
        self.data_loader: HistoricalDataLoader | None = None
        self.signal_runner: HistoricalSignalRunner | None = None
        self.adapter = DataAdapter()

        # Results
        self.signals_df: pd.DataFrame | None = None
        self.trades_df: pd.DataFrame | None = None
        self.metrics: BacktestMetrics | None = None
        self._success_prints = 0

    async def initialize(self):
        """Initialize engines and runners."""
        # Signal engine
        signal_config = SignalCoreEngineConfig(
            min_probability=0.5,
            min_score=0.1,
            enable_governance=False,
        )
        self.signal_engine = SignalCoreEngine(signal_config)
        await self.signal_engine.initialize()

        # Register default model
        model_config = ModelConfig(
            model_id="sma_crossover_v1",
            model_type="technical",
            parameters={"short_period": 20, "long_period": 50},
        )
        model = SMACrossoverModel(model_config)
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

    async def shutdown(self):
        """Shutdown engines."""
        if self.signal_engine:
            await self.signal_engine.shutdown()
        if self.backtest_engine:
            await self.backtest_engine.shutdown()

    async def run(self, data: dict[str, pd.DataFrame] | None = None) -> BacktestMetrics:
        """Run complete backtest pipeline.

        Args:
            data: Optional pre-loaded data. If None, loads from disk.

        Returns:
            BacktestMetrics with all performance data
        """
        try:
            await self.initialize()

            # Load data if not provided
            if data is None:
                print(f"[backtest] Loading data for {self.config.symbols}...")
                data = self.data_loader.load_batch(
                    self.config.symbols,
                    self.config.start_date,
                    self.config.end_date,
                )

            if not data:
                raise ValueError(f"No data loaded for {self.config.symbols}")

            # Generate signals
            print("[backtest] Generating historical signals...")
            signal_batches = await self.signal_runner.generate_batch_signals(data)
            print(f"[backtest] Generated {len(signal_batches)} signal batches.")
            if len(signal_batches) > 0:
                first_ts = next(iter(signal_batches.keys()))
                print(f"[backtest] Sample timestamp: {first_ts} (type: {type(first_ts)})")

            # Load data into backtest engine
            print("[backtest] Loading data into backtest engine...")
            for symbol, df in data.items():
                self.backtest_engine.load_data(symbol, df)

            # Strategy: react to signals
            def on_signal_bar(engine, symbol, bar):
                """Strategy callback on each bar."""
                ts = bar.timestamp

                # Check for stop loss
                pos = engine.get_position(symbol)
                if pos and pos.quantity > 0:
                    # Long position stop loss
                    if self.config.stop_loss_pct > 0 and bar.close < pos.avg_entry_price * (
                        1 - self.config.stop_loss_pct
                    ):
                        if self._success_prints < 10:
                            print(
                                f"[DEBUG] Stop Loss triggered for {symbol} at {ts} (Price: {bar.close:.2f}, Entry: {pos.avg_entry_price:.2f})"
                            )
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=pos.quantity,
                            order_type=OrderType.MARKET,
                        )
                        engine.submit_order(order)
                        return

                    # Long position take profit
                    if self.config.take_profit_pct > 0 and bar.close > pos.avg_entry_price * (
                        1 + self.config.take_profit_pct
                    ):
                        if self._success_prints < 10:
                            print(
                                f"[DEBUG] Take Profit triggered for {symbol} at {ts} (Price: {bar.close:.2f}, Entry: {pos.avg_entry_price:.2f})"
                            )
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=pos.quantity,
                            order_type=OrderType.MARKET,
                        )
                        engine.submit_order(order)
                        return

                # Check if we have a signal for this timestamp
                import logging

                batch = self.signal_runner.get_cached_batch(ts)
                if self._success_prints < 5:
                    logging.warning(
                        f"[DEBUG] Checking signals for {symbol} at {ts} (type: {type(ts)})"
                    )
                    if batch:
                        logging.warning(f"[DEBUG] Found batch for {ts}")
                    elif (
                        hasattr(self.signal_runner, "_batch_cache")
                        and len(self.signal_runner._batch_cache) > 0
                    ):
                        first_key = next(iter(self.signal_runner._batch_cache.keys()))
                        logging.warning(
                            f"[DEBUG] No batch. First cached key: {first_key} (type: {type(first_key)})"
                        )

                if not batch:
                    return

                signal = batch.get_by_symbol(symbol)
                if signal:
                    # Simple position sizing
                    cash = engine.get_cash()
                    equity = engine.get_equity()
                    max_qty = int((equity * self.config.max_position_size) / bar.close)

                    if max_qty <= 0:
                        return

                    # Execute on signal
                    is_valid = signal.is_actionable(
                        min_probability=self.signal_engine.config.min_probability,
                        min_score=self.signal_engine.config.min_score,
                    )

                    if signal.direction.value == "long" and is_valid:
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=max_qty,
                            order_type=OrderType.MARKET,
                        )
                        engine.submit_order(order)

                    elif signal.direction.value == "short" and is_valid:
                        pos = engine.get_position(symbol)
                        if pos:
                            # Close long position
                            qty = min(pos.quantity, max_qty)
                            if qty > 0:
                                order = Order(
                                    symbol=symbol,
                                    side=OrderSide.SELL,
                                    quantity=qty,
                                    order_type=OrderType.MARKET,
                                )
                                engine.submit_order(order)

            self.backtest_engine.set_strategy(on_signal_bar)

            # Run backtest
            print("[backtest] Running backtest simulation...")
            results = await self.backtest_engine.run_backtest()

            # Compute extended metrics
            print("[backtest] Computing metrics...")
            self.metrics = BacktestMetrics(
                total_return=results.metrics.total_return,
                annualized_return=results.metrics.annualized_return,
                volatility=results.metrics.volatility,
                downside_deviation=results.metrics.downside_deviation,
                sharpe_ratio=results.metrics.sharpe_ratio,
                sortino_ratio=results.metrics.sortino_ratio,
                calmar_ratio=results.metrics.calmar_ratio,
                max_drawdown=results.metrics.max_drawdown,
                avg_drawdown=results.metrics.avg_drawdown,
                max_drawdown_duration=results.metrics.max_drawdown_duration,
                num_trades=results.metrics.num_trades,
                win_rate=results.metrics.win_rate,
                profit_factor=results.metrics.profit_factor,
                avg_win=results.metrics.avg_win,
                avg_loss=results.metrics.avg_loss,
                largest_win=results.metrics.largest_win,
                largest_loss=results.metrics.largest_loss,
                avg_trade_duration=results.metrics.avg_trade_duration,
                expectancy=results.metrics.expectancy,
                recovery_factor=results.metrics.recovery_factor,
                equity_final=results.metrics.equity_final,
            )

            # Store results
            self.trades_df = results.trades
            self.signals_df = pd.DataFrame(
                [
                    s.to_dict()
                    for s in [sig for batch in signal_batches.values() for sig in batch.signals]
                ]
            )

            # Write artifacts
            self._write_artifacts(results)

            print("[backtest] ✓ Backtest complete")
            print(f"  Total Return: {self.metrics.total_return:.2f}%")
            print(f"  Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {self.metrics.max_drawdown:.2f}%")
            print(f"  Win Rate: {self.metrics.win_rate:.2f}%")
            print(f"  Artifacts: {self.output_dir}")

            return self.metrics

        finally:
            await self.shutdown()

    def _write_artifacts(self, results):
        """Write signals, trades, equity curve, and report."""
        # Signals
        if self.signals_df is not None and len(self.signals_df) > 0:
            self.signals_df.to_csv(self.output_dir / "signals.csv", index=False)

        # Trades
        if self.trades_df is not None and len(self.trades_df) > 0:
            self.trades_df.to_csv(self.output_dir / "trades.csv", index=False)

        # Equity curve
        if results.equity_curve is not None and len(results.equity_curve) > 0:
            results.equity_curve.to_csv(self.output_dir / "equity_curve.csv")

        # Report
        report = {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics) if self.metrics else {},
            "timestamp": datetime.now().isoformat(),
            "symbols_traded": self.config.symbols,
            "trades_count": int(self.metrics.num_trades) if self.metrics else 0,
        }

        (self.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))


async def run_backtest_async(config: BacktestConfig) -> BacktestMetrics:
    """Convenience function to run backtest asynchronously.

    Args:
        config: Backtest configuration

    Returns:
        BacktestMetrics with results
    """
    runner = BacktestRunner(config)
    return await runner.run()


def run_backtest(config: BacktestConfig) -> BacktestMetrics:
    """Convenience function to run backtest (sync wrapper).

    Args:
        config: Backtest configuration

    Returns:
        BacktestMetrics with results
    """
    return asyncio.run(run_backtest_async(config))
