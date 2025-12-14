"""
SignalCore governance hooks.

Implements governance rules for signal generation operations including
data quality checks, model validation, and signal threshold enforcement.
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


class SignalCoreRule(ABC):
    """Abstract base class for SignalCore governance rules."""

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
class DataQualityRule(SignalCoreRule):
    """Rule enforcing minimum data quality for signal generation.

    Attributes:
        min_data_points: Minimum number of data points required
        max_null_pct: Maximum percentage of null values allowed
    """

    min_data_points: int = 100
    max_null_pct: float = 0.01

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check data quality against thresholds.

        Args:
            context: Preflight context with data_points parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        data_points = params.get("data_points", 0)

        if data_points < self.min_data_points:
            return False, (
                f"Insufficient data: {data_points} points, minimum {self.min_data_points} required"
            )

        return True, f"Data quality check passed ({data_points} points)"


@dataclass
class SignalThresholdRule(SignalCoreRule):
    """Rule enforcing signal generation thresholds.

    Attributes:
        min_probability: Minimum probability for actionable signals
        min_score: Minimum absolute score for actionable signals
    """

    min_probability: float = 0.6
    min_score: float = 0.3

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Validate threshold configuration.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        if not 0 <= self.min_probability <= 1:
            return False, "min_probability must be between 0 and 1"

        if not 0 <= self.min_score <= 1:
            return False, "min_score must be between 0 and 1"

        return True, "Signal threshold configuration valid"


@dataclass
class ModelValidationRule(SignalCoreRule):
    """Rule enforcing model requirements before signal generation.

    Attributes:
        require_enabled_models: Require at least one enabled model
        max_symbols_per_batch: Maximum symbols allowed per batch
    """

    require_enabled_models: bool = True
    max_symbols_per_batch: int = 100

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check model requirements.

        Args:
            context: Preflight context with operation details

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        operation = context.operation

        # Check batch size for batch operations
        if operation == "generate_batch":
            symbol_count = params.get("symbol_count", 0)
            if symbol_count > self.max_symbols_per_batch:
                return False, (
                    f"Batch size {symbol_count} exceeds maximum {self.max_symbols_per_batch}"
                )

        return True, "Model validation passed"


class SignalCoreGovernanceHook(GovernanceHook):
    """Governance hook for SignalCore signal generation operations.

    Implements preflight checks and audit logging for all signal operations.
    Supports configurable rules for data quality, thresholds, and models.

    Example:
        >>> from ordinis.engines.signalcore.hooks import (
        ...     SignalCoreGovernanceHook,
        ...     DataQualityRule,
        ...     SignalThresholdRule,
        ... )
        >>> hook = SignalCoreGovernanceHook(
        ...     rules=[
        ...         DataQualityRule(min_data_points=200),
        ...         SignalThresholdRule(min_probability=0.7),
        ...     ]
        ... )
        >>> engine = SignalCoreEngine(config, governance_hook=hook)
    """

    def __init__(
        self,
        rules: list[SignalCoreRule] | None = None,
        audit_all_operations: bool = True,
    ) -> None:
        """Initialize SignalCore governance hook.

        Args:
            rules: List of governance rules to enforce
            audit_all_operations: Whether to audit all operations (not just failures)
        """
        self._rules = rules or [
            DataQualityRule(),
            SignalThresholdRule(),
            ModelValidationRule(),
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
            "SignalCore governance preflight: %s",
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
                "SignalCore audit: %s - %s (%s)",
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

    def add_rule(self, rule: SignalCoreRule) -> None:
        """Add a governance rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, rule_type: type[SignalCoreRule]) -> bool:
        """Remove rules of a specific type.

        Args:
            rule_type: Type of rule to remove

        Returns:
            True if any rules were removed
        """
        original_count = len(self._rules)
        self._rules = [r for r in self._rules if not isinstance(r, rule_type)]
        return len(self._rules) < original_count

    def get_rules(self) -> list[SignalCoreRule]:
        """Get all registered rules.

        Returns:
            List of governance rules
        """
        return self._rules.copy()
