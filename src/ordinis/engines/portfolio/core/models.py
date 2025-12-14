"""
Portfolio engine domain models.

Defines core data structures for portfolio rebalancing operations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class StrategyType(Enum):
    """Types of rebalancing strategies."""

    TARGET_ALLOCATION = "target_allocation"
    RISK_PARITY = "risk_parity"
    SIGNAL_DRIVEN = "signal_driven"
    THRESHOLD_BASED = "threshold_based"


@dataclass
class RebalancingHistory:
    """Historical record of a rebalancing event.

    Attributes:
        timestamp: When the rebalancing occurred
        strategy_type: Which strategy was used
        decisions_count: Number of rebalancing decisions generated
        total_adjustment_value: Total dollar value of adjustments
        execution_status: Status of execution (planned, executed, failed)
        metadata: Additional strategy-specific information
    """

    timestamp: datetime
    strategy_type: StrategyType
    decisions_count: int
    total_adjustment_value: float
    execution_status: str
    metadata: dict[str, Any]


@dataclass
class ExecutionResult:
    """Result of executing rebalancing decisions.

    Attributes:
        timestamp: When execution completed
        decisions_executed: Number of decisions successfully executed
        decisions_failed: Number of decisions that failed
        total_value_traded: Total dollar value traded
        success: True if all decisions executed successfully
        errors: List of error messages if any
    """

    timestamp: datetime
    decisions_executed: int
    decisions_failed: int
    total_value_traded: float
    success: bool
    errors: list[str]
