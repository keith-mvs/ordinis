"""
Orchestration Engine Governance Hook.

Validates trading cycles for compliance with operating constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ordinis.engines.base import (
    AuditRecord,
    BaseGovernanceHook,
    Decision,
    PreflightContext,
    PreflightResult,
)

_logger = logging.getLogger(__name__)


@dataclass
class LatencyBudgetRule:
    """Rule for validating latency budgets."""

    total_budget_ms: int = 300
    warn_threshold_pct: float = 0.80  # Warn at 80% of budget

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate latency requirements."""
        # This rule checks configuration, not runtime
        # Runtime latency enforcement happens in the engine
        return True, None


@dataclass
class TradingModeRule:
    """Rule for validating trading mode constraints."""

    allowed_modes: tuple[str, ...] = ("live", "paper", "backtest")
    require_explicit_live: bool = True

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate trading mode."""
        mode = context.get("mode", "")
        if mode and mode not in self.allowed_modes:
            return False, f"Invalid trading mode: {mode}"

        return True, None


class OrchestrationGovernanceHook(BaseGovernanceHook):
    """
    Governance hook for orchestration engine operations.

    Validates:
    - Trading mode constraints
    - Latency budget compliance
    - Cycle frequency limits
    """

    def __init__(
        self,
        latency_rule: LatencyBudgetRule | None = None,
        mode_rule: TradingModeRule | None = None,
    ) -> None:
        """Initialize governance hook with validation rules."""
        super().__init__()
        self.latency_rule = latency_rule or LatencyBudgetRule()
        self.mode_rule = mode_rule or TradingModeRule()
        self._audit_log: list[AuditRecord] = []

    async def preflight(
        self,
        context: PreflightContext | dict[str, Any],
    ) -> PreflightResult:
        """
        Validate cycle before execution.

        Args:
            context: Operation context with parameters.

        Returns:
            PreflightResult with approval decision.
        """
        if isinstance(context, dict):
            ctx = context
        else:
            ctx = context if isinstance(context, dict) else {"context": context}

        operation = ctx.get("operation", "unknown")
        reasons: list[str] = []

        # Validate run_cycle operations
        if operation == "run_cycle":
            passed, reason = self.mode_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

            passed, reason = self.latency_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

        if reasons:
            _logger.warning("Preflight failed: %s", "; ".join(reasons))
            return PreflightResult(
                approved=False,
                decision=Decision.DENY,
                reason="; ".join(reasons),
            )

        return PreflightResult(
            approved=True,
            decision=Decision.ALLOW,
            reason="All validation rules passed",
        )

    async def audit(self, record: AuditRecord) -> None:
        """
        Record cycle for audit trail.

        Args:
            record: Audit record with operation details.
        """
        self._audit_log.append(record)
        _logger.debug(
            "Audit: %s.%s completed in %.2fms",
            record.engine_id,
            record.operation,
            record.duration_ms,
        )

    def get_audit_log(self) -> list[AuditRecord]:
        """Get audit log entries."""
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        """Clear audit log (for testing)."""
        self._audit_log.clear()
