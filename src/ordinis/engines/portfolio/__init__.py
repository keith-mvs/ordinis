"""
Portfolio rebalancing engine for Ordinis trading system.

Provides multiple rebalancing strategies:
- Target Allocation: Maintain fixed % weights
- Risk Parity: Equal risk contribution per asset
- Signal-Driven: Rebalance based on indicator signals
- Threshold-Based: Rebalance when drift exceeds tolerance

Plus a unified engine to orchestrate multiple strategies.
"""

from .engine import ExecutionResult, RebalancingEngine, RebalancingHistory, StrategyType
from .risk_parity import RiskParityDecision, RiskParityRebalancer, RiskParityWeights
from .signal_driven import (
    SignalDrivenDecision,
    SignalDrivenRebalancer,
    SignalDrivenWeights,
    SignalInput,
    SignalMethod,
)
from .target_allocation import RebalanceDecision, TargetAllocation, TargetAllocationRebalancer
from .threshold_based import (
    ThresholdBasedRebalancer,
    ThresholdConfig,
    ThresholdDecision,
    ThresholdStatus,
)

__all__ = [
    # Target Allocation
    "TargetAllocationRebalancer",
    "TargetAllocation",
    "RebalanceDecision",
    # Risk Parity
    "RiskParityRebalancer",
    "RiskParityWeights",
    "RiskParityDecision",
    # Signal-Driven
    "SignalDrivenRebalancer",
    "SignalDrivenWeights",
    "SignalDrivenDecision",
    "SignalInput",
    "SignalMethod",
    # Threshold-Based
    "ThresholdBasedRebalancer",
    "ThresholdConfig",
    "ThresholdDecision",
    "ThresholdStatus",
    # Unified Engine
    "RebalancingEngine",
    "StrategyType",
    "RebalancingHistory",
    "ExecutionResult",
]
