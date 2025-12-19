"""
ProofBench - Backtesting Engine

Event-driven simulation engine for strategy validation.
Provides realistic execution modeling with comprehensive performance analytics.

The engine follows the standard Ordinis engine template with:
- core/ - Engine, config, and domain models
- hooks/ - Governance hooks for preflight/audit
- analytics/ - Performance analysis and reporting
"""

# Core engine components
# Analytics
from ordinis.engines.proofbench.analytics.llm_enhanced import LLMPerformanceNarrator
from ordinis.engines.proofbench.analytics.performance import (
    PerformanceAnalyzer,
    PerformanceMetrics,
)
from ordinis.engines.proofbench.core import (
    Bar,
    Event,
    EventQueue,
    EventType,
    ExecutionConfig,
    ExecutionSimulator,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionSide,
    ProofBenchEngine,
    ProofBenchEngineConfig,
    SimulationConfig,
    SimulationEngine,
    SimulationResults,
    Trade,
)

# Governance hooks
from ordinis.engines.proofbench.hooks import (
    CapitalLimitRule,
    DataValidationRule,
    ProofBenchGovernanceHook,
    SymbolLimitRule,
)

__all__ = [
    # Execution
    "Bar",
    # Governance Hooks
    "CapitalLimitRule",
    "DataValidationRule",
    # Events
    "Event",
    "EventQueue",
    "EventType",
    "ExecutionConfig",
    "ExecutionSimulator",
    "Fill",
    # Analytics
    "LLMPerformanceNarrator",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    # Portfolio
    "Portfolio",
    "Position",
    "PositionSide",
    # Core Engine
    "ProofBenchEngine",
    "ProofBenchEngineConfig",
    "ProofBenchGovernanceHook",
    # Simulation
    "SimulationConfig",
    "SimulationEngine",
    "SimulationResults",
    "SymbolLimitRule",
    "Trade",
]

__version__ = "0.1.0"
