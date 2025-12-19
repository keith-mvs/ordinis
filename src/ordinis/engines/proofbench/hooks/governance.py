"""
ProofBench governance hooks.

Implements governance rules for backtesting operations including
capital limits, symbol restrictions, and data validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


class ProofBenchRule(ABC):
    """Abstract base class for ProofBench governance rules."""

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
class CapitalLimitRule(ProofBenchRule):
    """Rule enforcing capital limits for backtests.

    Attributes:
        min_capital: Minimum initial capital allowed
        max_capital: Maximum initial capital allowed (0 = unlimited)
    """

    min_capital: float = 1000.0
    max_capital: float = 0.0

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check capital against limits.

        Args:
            context: Preflight context with initial_capital parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        capital = params.get("initial_capital", 0)

        if capital < self.min_capital:
            return False, (
                f"Initial capital ${capital:,.2f} below minimum ${self.min_capital:,.2f}"
            )

        if self.max_capital > 0 and capital > self.max_capital:
            return False, (
                f"Initial capital ${capital:,.2f} exceeds maximum ${self.max_capital:,.2f}"
            )

        return True, f"Capital ${capital:,.2f} within limits"


@dataclass
class SymbolLimitRule(ProofBenchRule):
    """Rule enforcing symbol limits for backtests.

    Attributes:
        max_symbols: Maximum number of symbols allowed per backtest
        allowed_symbols: List of allowed symbols (empty = all allowed)
        blocked_symbols: List of blocked symbols
    """

    max_symbols: int = 50
    allowed_symbols: list[str] | None = None
    blocked_symbols: list[str] | None = None

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check symbols against limits.

        Args:
            context: Preflight context with symbols parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        symbols = params.get("symbols", [])

        if len(symbols) > self.max_symbols:
            return False, (f"Symbol count {len(symbols)} exceeds maximum {self.max_symbols}")

        if self.allowed_symbols:
            invalid = [s for s in symbols if s not in self.allowed_symbols]
            if invalid:
                return False, f"Symbols not in allowlist: {invalid}"

        if self.blocked_symbols:
            blocked = [s for s in symbols if s in self.blocked_symbols]
            if blocked:
                return False, f"Blocked symbols: {blocked}"

        return True, f"All {len(symbols)} symbols valid"


@dataclass
class DataValidationRule(ProofBenchRule):
    """Rule enforcing data requirements for backtests.

    Attributes:
        require_data: Require at least one symbol loaded
        min_bars_per_symbol: Minimum bars required per symbol
    """

    require_data: bool = True
    min_bars_per_symbol: int = 100

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check data requirements.

        Args:
            context: Preflight context with symbols parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        symbols = params.get("symbols", [])

        if self.require_data and not symbols:
            return False, "No data loaded for backtest"

        return True, "Data validation passed"


class ProofBenchGovernanceHook(GovernanceHook):
    """Governance hook for ProofBench backtesting operations.

    Implements preflight checks and audit logging for all backtest operations.
    Supports configurable rules for capital, symbols, and data validation.

    Example:
        >>> from ordinis.engines.proofbench.hooks import (
        ...     ProofBenchGovernanceHook,
        ...     CapitalLimitRule,
        ...     SymbolLimitRule,
        ... )
        >>> hook = ProofBenchGovernanceHook(
        ...     rules=[
        ...         CapitalLimitRule(min_capital=10000, max_capital=1000000),
        ...         SymbolLimitRule(max_symbols=20),
        ...     ]
        ... )
        >>> engine = ProofBenchEngine(config, governance_hook=hook)
    """

    def __init__(
        self,
        rules: list[ProofBenchRule] | None = None,
        audit_all_operations: bool = True,
    ) -> None:
        """Initialize ProofBench governance hook.

        Args:
            rules: List of governance rules to enforce
            audit_all_operations: Whether to audit all operations (not just failures)
        """
        self._rules = rules or [
            CapitalLimitRule(),
            SymbolLimitRule(),
            DataValidationRule(),
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
            "ProofBench governance preflight: %s",
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
        if self._audit_all or record.status in ("blocked", "error"):
            self._audit_log.append(record)
            _logger.info(
                "ProofBench audit: %s - %s (%s)",
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

    def add_rule(self, rule: ProofBenchRule) -> None:
        """Add a governance rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, rule_type: type[ProofBenchRule]) -> bool:
        """Remove rules of a specific type.

        Args:
            rule_type: Type of rule to remove

        Returns:
            True if any rules were removed
        """
        original_count = len(self._rules)
        self._rules = [r for r in self._rules if not isinstance(r, rule_type)]
        return len(self._rules) < original_count

    def get_rules(self) -> list[ProofBenchRule]:
        """Get all registered rules.

        Returns:
            List of governance rules
        """
        return self._rules.copy()
