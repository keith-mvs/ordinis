"""
Network Parity Portfolio Optimization System

A comprehensive quantitative trading framework that optimizes equity portfolios
through iterative refinement using NVIDIA Nemo models.

Modules:
    - equity_universe: Universe construction and sector management
    - data_pipeline: Historical data loading with 1min/1D aggregation
    - backtesting: Strategy backtesting with transaction cost modeling
    - nemo_integration: NVIDIA Nemo model integration for optimization
    - optimization: Iterative optimization controller
    - reporting: JSON traceability and audit logging
"""

__version__ = "1.0.0"
__author__ = "Ordinis Quantitative Research"

import sys
from pathlib import Path
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import NetworkParityParams, OptimizationConfig
from equity_universe import EquityUniverse, StockInfo
from data_pipeline import DataPipeline, MarketData
from backtesting import BacktestEngine, BacktestResult, Trade
from nemo_integration import NemoOptimizer, NemoSuggestion
from optimization import OptimizationController, OptimizationState
from reporting import ReportGenerator

__all__ = [
    "NetworkParityParams",
    "OptimizationConfig",
    "EquityUniverse",
    "StockInfo",
    "DataPipeline",
    "MarketData",
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "NemoOptimizer",
    "NemoSuggestion",
    "OptimizationController",
    "OptimizationState",
    "ReportGenerator",
]
