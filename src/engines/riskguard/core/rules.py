"""
Risk rule definitions and evaluation framework.

All rules are deterministic and auditable.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class RuleCategory(Enum):
    """Categories of risk rules."""

    PRE_TRADE = "pre_trade"  # Before order generation
    ORDER_VALIDATION = "order_val"  # Before order submission
    POSITION_LIMIT = "position"  # Position-level constraints
    PORTFOLIO_LIMIT = "portfolio"  # Portfolio-level constraints
    KILL_SWITCH = "kill_switch"  # Emergency halt conditions
    SANITY_CHECK = "sanity"  # Data/connectivity validation


@dataclass
class RiskRule:
    """
    Definition of a single risk rule.

    Rules are deterministic expressions evaluated against current state.
    """

    rule_id: str
    category: RuleCategory
    name: str
    description: str

    # Rule specification
    condition: str  # Human-readable condition
    threshold: float
    comparison: Literal["<", "<=", ">", ">=", "==", "!="]

    # Actions
    action_on_breach: Literal["reject", "resize", "warn", "halt"]
    severity: Literal["low", "medium", "high", "critical"]

    # Audit
    enabled: bool = True
    last_modified: datetime | None = None
    modified_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, current_value: float) -> bool:  # noqa: PLR0911
        """
        Evaluate rule against current value.

        Args:
            current_value: Current value to check

        Returns:
            True if rule passes, False if breached
        """
        if not self.enabled:
            return True

        if self.comparison == "<":
            return current_value < self.threshold
        if self.comparison == "<=":
            return current_value <= self.threshold
        if self.comparison == ">":
            return current_value > self.threshold
        if self.comparison == ">=":
            return current_value >= self.threshold
        if self.comparison == "==":
            return abs(current_value - self.threshold) < 1e-9
        if self.comparison == "!=":
            return abs(current_value - self.threshold) >= 1e-9
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "action_on_breach": self.action_on_breach,
            "severity": self.severity,
            "enabled": self.enabled,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "modified_by": self.modified_by,
            "metadata": self.metadata,
        }


@dataclass
class RiskCheckResult:
    """
    Result of risk rule evaluation.

    Provides complete audit trail of rule checks.
    """

    rule_id: str
    rule_name: str
    passed: bool
    current_value: float
    threshold: float
    comparison: str
    message: str
    action_taken: str
    severity: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "message": self.message,
            "action_taken": self.action_taken,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
        }
