"""
ProofBench - Backtesting Engine

Event-driven simulation engine for strategy validation.
Provides realistic execution modeling with comprehensive performance analytics.
"""

from .analytics.performance import PerformanceAnalyzer, PerformanceMetrics
from .core.events import Event, EventQueue, EventType
from .core.execution import (
    Bar,
    ExecutionConfig,
    ExecutionSimulator,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from .core.portfolio import Portfolio, Position, PositionSide, Trade
from .core.simulator import SimulationConfig, SimulationEngine, SimulationResults

__all__ = [
    # Events
    "Event",
    "EventType",
    "EventQueue",
    # Simulation
    "SimulationEngine",
    "SimulationConfig",
    "SimulationResults",
    # Portfolio
    "Portfolio",
    "Position",
    "PositionSide",
    "Trade",
    # Execution
    "ExecutionSimulator",
    "ExecutionConfig",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Fill",
    "Bar",
    # Analytics
    "PerformanceAnalyzer",
    "PerformanceMetrics",
]

__version__ = "0.1.0"
