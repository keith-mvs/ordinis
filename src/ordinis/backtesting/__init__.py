"""Backtesting module for SignalCore strategies."""

from .data_adapter import DataAdapter, HistoricalDataLoader
from .metrics import BacktestMetrics, MetricsEngine
from .runner import BacktestConfig, BacktestRunner
from .signal_runner import HistoricalSignalRunner, SignalRunnerConfig

__all__ = [
    "BacktestRunner",
    "BacktestConfig",
    "DataAdapter",
    "HistoricalDataLoader",
    "MetricsEngine",
    "BacktestMetrics",
    "HistoricalSignalRunner",
    "SignalRunnerConfig",
]
