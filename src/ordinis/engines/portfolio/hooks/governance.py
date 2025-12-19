"""
Portfolio governance hooks.

Implements governance rules for portfolio rebalancing operations including
position limits, trade value constraints, and rebalance frequency controls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

from ordinis.engines.base import (
    AuditRecord,
    Decision,
    GovernanceHook,
    PreflightContext,
    PreflightResult,
)

_logger = logging.getLogger(__name__)


class PortfolioRule(ABC):
    """Abstract base class for portfolio governance rules."""

    @abstractmethod
    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check if rule passes.

        Args:
            context: Preflight context with operation details

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        ...


@dataclass
class PositionLimitRule(PortfolioRule):
    """Rule enforcing maximum position size as percentage of portfolio.

    Attributes:
        max_position_pct: Maximum single position as fraction of portfolio (0-1)
    """

    max_position_pct: float = 0.25

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check position limits against generated decisions.

        Args:
            context: Preflight context with positions and prices

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        positions = params.get("positions", {})
        prices = params.get("prices", {})

        if not positions or not prices:
            return True, "No positions to check"

        # Calculate total portfolio value
        total_value = sum(
            positions.get(sym, 0) * prices.get(sym, 0) for sym in set(positions) | set(prices)
        )

        if total_value <= 0:
            return True, "Portfolio value is zero"

        # Check each position
        for symbol, shares in positions.items():
            if symbol in prices:
                position_value = shares * prices[symbol]
                position_pct = position_value / total_value
                if position_pct > self.max_position_pct:
                    return False, (
                        f"Position {symbol} at {position_pct:.1%} exceeds "
                        f"limit of {self.max_position_pct:.1%}"
                    )

        return True, "All positions within limits"


@dataclass
class TradeValueRule(PortfolioRule):
    """Rule enforcing minimum and maximum trade values.

    Attributes:
        min_trade_value: Minimum trade value in dollars
        max_trade_value: Maximum single trade value in dollars (0 = unlimited)
    """

    min_trade_value: float = 10.0
    max_trade_value: float = 0.0

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check trade values against thresholds.

        Note: This rule validates configuration, not actual decisions.
        Decision-level validation occurs during execution.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        # This rule is primarily a configuration validation
        # Actual trade value checks happen during execution
        if self.min_trade_value < 0:
            return False, "Minimum trade value cannot be negative"

        if self.max_trade_value > 0 and self.max_trade_value < self.min_trade_value:
            return False, "Maximum trade value cannot be less than minimum"

        return True, "Trade value configuration valid"


@dataclass
class RebalanceFrequencyRule(PortfolioRule):
    """Rule enforcing minimum time between rebalances.

    Attributes:
        min_interval: Minimum time between rebalances
        last_rebalance: Timestamp of last rebalance (set by engine)
    """

    min_interval: timedelta = field(default_factory=lambda: timedelta(hours=24))
    last_rebalance: datetime | None = None

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check if sufficient time has passed since last rebalance.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        if self.last_rebalance is None:
            return True, "No previous rebalance recorded"

        now = context.timestamp or datetime.now(tz=UTC)
        elapsed = now - self.last_rebalance

        if elapsed < self.min_interval:
            remaining = self.min_interval - elapsed
            return False, (
                f"Rebalance too soon. {remaining.total_seconds() / 3600:.1f} hours "
                f"remaining until next allowed rebalance"
            )

        return True, f"Sufficient time elapsed ({elapsed.total_seconds() / 3600:.1f} hours)"

    def update_last_rebalance(self, timestamp: datetime | None = None) -> None:
        """Update the last rebalance timestamp.

        Args:
            timestamp: Rebalance timestamp (default: now)
        """
        self.last_rebalance = timestamp or datetime.now(tz=UTC)


class PortfolioGovernanceHook(GovernanceHook):
    """Governance hook for portfolio rebalancing operations.

    Implements preflight checks and audit logging for all portfolio operations.
    Supports configurable rules for position limits, trade values, and frequency.

    Example:
        >>> from ordinis.engines.portfolio.hooks import (
        ...     PortfolioGovernanceHook,
        ...     PositionLimitRule,
        ...     RebalanceFrequencyRule,
        ... )
        >>> from datetime import timedelta
        >>> hook = PortfolioGovernanceHook(
        ...     rules=[
        ...         PositionLimitRule(max_position_pct=0.20),
        ...         RebalanceFrequencyRule(min_interval=timedelta(hours=4)),
        ...     ]
        ... )
        >>> engine = PortfolioEngine(config, governance_hook=hook)
    """

    def __init__(
        self,
        rules: list[PortfolioRule] | None = None,
        audit_all_operations: bool = True,
    ) -> None:
        """Initialize portfolio governance hook.

        Args:
            rules: List of governance rules to enforce
            audit_all_operations: Whether to audit all operations (not just failures)
        """
        self._rules = rules or [
            PositionLimitRule(),
            TradeValueRule(),
        ]
        self._audit_all = audit_all_operations
        self._audit_log: list[AuditRecord] = []

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Check all governance rules before operation.

        Args:
            context: Operation context with parameters

        Returns:
            PreflightResult indicating allow/deny and reason
        """
        _logger.debug(
            "Portfolio governance preflight: %s",
            context.operation,
        )

        failed_rules: list[str] = []
        modifications: dict[str, Any] = {}

        for rule in self._rules:
            try:
                passed, reason = rule.check(context)
                if not passed:
                    failed_rules.append(f"{rule.__class__.__name__}: {reason}")
            except Exception as e:
                _logger.error("Rule %s failed with exception: %s", rule.__class__.__name__, e)
                failed_rules.append(f"{rule.__class__.__name__}: Error - {e}")

        if failed_rules:
            return PreflightResult(
                decision=Decision.DENY,
                allowed=False,
                reason="; ".join(failed_rules),
            )

        return PreflightResult(
            decision=Decision.ALLOW,
            allowed=True,
            reason="All governance rules passed",
            modifications=modifications if modifications else None,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Record audit entry for operation.

        Args:
            record: Audit record to log
        """
        if self._audit_all or record.status in ("blocked", "error", "partial"):
            self._audit_log.append(record)
            _logger.info(
                "Portfolio audit: %s - %s (%s)",
                record.operation,
                record.status,
                record.details,
            )

    def get_audit_log(self, limit: int | None = None) -> list[AuditRecord]:
        """Get audit log entries.

        Args:
            limit: Maximum entries to return (None = all)

        Returns:
            List of audit records
        """
        if limit:
            return self._audit_log[-limit:]
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def add_rule(self, rule: PortfolioRule) -> None:
        """Add a governance rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, rule_type: type[PortfolioRule]) -> bool:
        """Remove rules of a specific type.

        Args:
            rule_type: Type of rule to remove

        Returns:
            True if any rules were removed
        """
        original_count = len(self._rules)
        self._rules = [r for r in self._rules if not isinstance(r, rule_type)]
        return len(self._rules) < original_count

    def get_rules(self) -> list[PortfolioRule]:
        """Get all registered rules.

        Returns:
            List of governance rules
        """
        return self._rules.copy()
