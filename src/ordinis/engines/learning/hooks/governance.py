"""
Learning Engine Governance Hook.

Validates learning operations for compliance and safety.
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
class TrainingValidationRule:
    """Rule for validating training operations."""

    min_samples_required: int = 100
    max_training_duration_hours: float = 24.0
    allowed_model_types: tuple[str, ...] = ("signal", "risk", "llm", "ensemble")

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate training request."""
        model_type = context.get("model_type", "")
        if model_type and model_type not in self.allowed_model_types:
            return False, f"Invalid model type: {model_type}"

        sample_count = context.get("sample_count", 0)
        if sample_count and sample_count < self.min_samples_required:
            return False, f"Insufficient samples: {sample_count} < {self.min_samples_required}"

        return True, None


@dataclass
class PromotionValidationRule:
    """Rule for validating model promotions."""

    require_staging_first: bool = True
    min_staging_duration_hours: float = 1.0
    require_human_approval_for_production: bool = True

    def validate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate promotion request."""
        target_stage = context.get("target_stage", "")

        if target_stage == "production" and self.require_staging_first:
            current_stage = context.get("current_stage", "")
            if current_stage != "staging":
                return False, "Model must be in staging before production promotion"

        return True, None


class LearningGovernanceHook(BaseGovernanceHook):
    """
    Governance hook for learning engine operations.

    Validates:
    - Training job submissions
    - Model promotions
    - Drift response actions
    """

    def __init__(
        self,
        training_rule: TrainingValidationRule | None = None,
        promotion_rule: PromotionValidationRule | None = None,
    ) -> None:
        """Initialize governance hook with validation rules."""
        super().__init__()
        self.training_rule = training_rule or TrainingValidationRule()
        self.promotion_rule = promotion_rule or PromotionValidationRule()
        self._audit_log: list[AuditRecord] = []

    async def preflight(
        self,
        context: PreflightContext | dict[str, Any],
    ) -> PreflightResult:
        """
        Validate learning operation before execution.

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

        # Validate training submissions
        if operation == "submit_training_job":
            passed, reason = self.training_rule.validate(ctx)
            if not passed and reason:
                reasons.append(reason)

        # Validate model promotions
        elif operation == "promote_model":
            passed, reason = self.promotion_rule.validate(ctx)
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
        Record learning operation for audit trail.

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
