"""
Trade execution optimization models.

- RL-based execution: TensorTrade integration
- Classical algorithms: VWAP, TWAP, Almgren-Chriss
"""

from .classical_algorithms import (
    AlmgrenChrissExecutor,
    TWAPExecutor,
    VWAPExecutor,
)
from .tensortrade_executor import (
    ExecutionAction,
    ExecutionResult,
    RLExecutionOptimizer,
    TensorTradeExecutor,
)

__all__ = [
    # RL-based
    "TensorTradeExecutor",
    "RLExecutionOptimizer",
    "ExecutionAction",
    "ExecutionResult",
    # Classical
    "VWAPExecutor",
    "TWAPExecutor",
    "AlmgrenChrissExecutor",
]
