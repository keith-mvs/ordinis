"""
Portfolio rebalancing engine for Ordinis trading system.

Provides multiple rebalancing strategies:
- Target Allocation: Maintain fixed % weights
- Risk Parity: Equal risk contribution per asset
- Signal-Driven: Rebalance based on indicator signals
- Threshold-Based: Rebalance when drift exceeds tolerance

Plus a unified engine to orchestrate multiple strategies, with event hooks
and adapters for integration with other Ordinis systems.

The engine follows the standard Ordinis engine template with:
- core/ - Engine, config, and domain models
- hooks/ - Governance hooks for preflight/audit
- strategies/ - Rebalancing strategy implementations
"""

# Core engine components
# Adapters
from ordinis.engines.portfolio.adapters import (
    FlowRouteAdapter,
    FlowRouteOrderRequest,
    ProofBenchAdapter,
    SignalCoreAdapter,
)
from ordinis.engines.portfolio.core import (
    ExecutionResult,
    PortfolioEngine,
    PortfolioEngineConfig,
    RebalancingHistory,
    StrategyType,
)

# Events
from ordinis.engines.portfolio.events import EventHooks, RebalanceEvent, RebalanceEventType

# Governance hooks
from ordinis.engines.portfolio.hooks import (
    PortfolioGovernanceHook,
    PositionLimitRule,
    RebalanceFrequencyRule,
    TradeValueRule,
)

# Strategies
from ordinis.engines.portfolio.strategies import (
    RebalanceDecision,
    RiskParityDecision,
    RiskParityRebalancer,
    RiskParityWeights,
    SignalDrivenDecision,
    SignalDrivenRebalancer,
    SignalDrivenWeights,
    SignalInput,
    SignalMethod,
    TargetAllocation,
    TargetAllocationRebalancer,
    ThresholdBasedRebalancer,
    ThresholdConfig,
    ThresholdDecision,
    ThresholdStatus,
)

# Backward compatibility aliases
RebalancingEngine = PortfolioEngine

__all__ = [
    # Events
    "EventHooks",
    "ExecutionResult",
    "FlowRouteAdapter",
    "FlowRouteOrderRequest",
    # Core Engine
    "PortfolioEngine",
    "PortfolioEngineConfig",
    # Governance Hooks
    "PortfolioGovernanceHook",
    "PositionLimitRule",
    "ProofBenchAdapter",
    "RebalanceDecision",
    "RebalanceEvent",
    "RebalanceEventType",
    "RebalanceFrequencyRule",
    "RebalancingEngine",  # Backward compatibility alias
    "RebalancingHistory",
    "RiskParityDecision",
    # Risk Parity
    "RiskParityRebalancer",
    "RiskParityWeights",
    # Adapters
    "SignalCoreAdapter",
    "SignalDrivenDecision",
    # Signal-Driven
    "SignalDrivenRebalancer",
    "SignalDrivenWeights",
    "SignalInput",
    "SignalMethod",
    "StrategyType",
    "TargetAllocation",
    # Target Allocation
    "TargetAllocationRebalancer",
    # Threshold-Based
    "ThresholdBasedRebalancer",
    "ThresholdConfig",
    "ThresholdDecision",
    "ThresholdStatus",
    "TradeValueRule",
]
