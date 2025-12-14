"""
PortfolioOpt Governance Hook - Risk and Compliance Validation.

Implements governance preflight and audit for portfolio optimization operations.
Validates risk limits, concentration constraints, and optimization parameters.
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
class RiskLimitRule:
    """Rule for validating risk limits."""

    max_target_return: float = 0.05  # 5% max target return
    max_weight_per_asset: float = 0.30  # 30% max single asset
    min_assets: int = 3  # Minimum assets for diversification

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate risk limits."""
        target_return = context.get("target_return", 0)
        if target_return > self.max_target_return:
            return (
                False,
                f"Target return {target_return:.2%} exceeds limit {self.max_target_return:.2%}",
            )

        max_weight = context.get("max_weight", 0)
        if max_weight > self.max_weight_per_asset:
            return (
                False,
                f"Max weight {max_weight:.2%} exceeds limit {self.max_weight_per_asset:.2%}",
            )

        n_assets = context.get("n_assets", 0)
        if n_assets < self.min_assets:
            return False, f"Only {n_assets} assets provided, minimum is {self.min_assets}"

        return True, None


@dataclass
class DataQualityRule:
    """Rule for validating input data quality."""

    min_periods: int = 20  # Minimum historical periods
    max_periods: int = 10000  # Maximum to prevent memory issues

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate data quality requirements."""
        n_periods = context.get("n_periods", 0)

        if n_periods < self.min_periods:
            return False, f"Insufficient data: {n_periods} periods, minimum is {self.min_periods}"

        if n_periods > self.max_periods:
            return False, f"Too much data: {n_periods} periods, maximum is {self.max_periods}"

        return True, None


@dataclass
class SolverValidationRule:
    """Rule for validating solver configuration."""

    allowed_apis: tuple[str, ...] = ("cvxpy", "cuopt")

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate solver configuration."""
        api = context.get("api", "cvxpy")
        if api not in self.allowed_apis:
            return False, f"Invalid solver API: {api}. Allowed: {self.allowed_apis}"

        return True, None


class PortfolioOptGovernanceHook(BaseGovernanceHook):
    """
    Governance hook for portfolio optimization operations.

    Validates:
    - Risk limits (target return, concentration)
    - Data quality (sufficient history)
    - Solver configuration
    """

    def __init__(
        self,
        risk_rule: RiskLimitRule | None = None,
        data_rule: DataQualityRule | None = None,
        solver_rule: SolverValidationRule | None = None,
    ) -> None:
        """Initialize governance hook with validation rules."""
        super().__init__()
        self.risk_rule = risk_rule or RiskLimitRule()
        self.data_rule = data_rule or DataQualityRule()
        self.solver_rule = solver_rule or SolverValidationRule()
        self._audit_log: list[AuditRecord] = []

    async def preflight(
        self,
        context: PreflightContext | dict[str, Any],
    ) -> PreflightResult:
        """
        Validate optimization request before execution.

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

        # Skip non-optimization operations
        if operation not in ("optimize", "generate_scenarios"):
            return PreflightResult(decision=Decision.ALLOW, reason="Non-optimization operation")

        # Validate optimization requests
        if operation == "optimize":
            # Risk limits
            passed, reason = self.risk_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

            # Data quality
            passed, reason = self.data_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

            # Solver validation
            passed, reason = self.solver_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

        # Validate scenario generation
        elif operation == "generate_scenarios":
            n_paths = ctx.get("n_paths", 0)
            if n_paths < 100:
                reasons.append(f"Too few paths: {n_paths}, minimum is 100")
            if n_paths > 100000:
                reasons.append(f"Too many paths: {n_paths}, maximum is 100000")

        if reasons:
            _logger.warning("Preflight failed: %s", "; ".join(reasons))
            return PreflightResult(
                decision=Decision.DENY,
                reason="; ".join(reasons),
            )

        return PreflightResult(
            decision=Decision.ALLOW,
            reason="All validation rules passed",
        )

    async def audit(self, record: AuditRecord) -> None:
        """
        Record optimization for audit trail.

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
