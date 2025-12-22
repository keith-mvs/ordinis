"""Governance hooks for engine operations.

This module defines the standard hook interfaces that all engines
must implement for consistent governance, audit, and policy enforcement.

Hooks are called at engine boundaries:
- preflight(): Before an operation (can block)
- audit(): After an operation (records outcome)
- on_error(): When an error occurs
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Protocol

from ordinis.engines.base.models import AuditRecord, EngineError

_logger = logging.getLogger(__name__)


class Decision(Enum):
    """Governance decision outcomes."""

    ALLOW = "allow"  # Proceed with operation
    DENY = "deny"  # Block operation
    WARN = "warn"  # Proceed with warning logged
    DEFER = "defer"  # Escalate for human review


@dataclass
class PreflightContext:
    """Context passed to preflight checks.

    Attributes:
        engine: Engine name.
        action: Operation being attempted.
        inputs: Operation inputs (sanitized).
        trace_id: Distributed trace ID.
        user_id: User initiating the action (if applicable).
        metadata: Additional context.
    """

    engine: str
    action: str
    inputs: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, Any]:
        """Alias for inputs (backward compatibility)."""
        return self.inputs

    @property
    def operation(self) -> str:
        """Alias for action (backward compatibility)."""
        return self.action

    @property
    def timestamp(self) -> Any:
        """Get timestamp from metadata."""
        return self.metadata.get("timestamp")


@dataclass
class PreflightResult:
    """Result from a preflight check.

    Attributes:
        decision: Allow/deny/warn/defer.
        reason: Explanation for the decision.
        policy_id: ID of the policy that made the decision.
        policy_version: Version of the policy.
        adjustments: Suggested modifications to inputs.
        warnings: List of warnings to log.
        expires_at: When this decision expires (for caching).
    """

    decision: Decision
    reason: str = ""
    policy_id: str | None = None
    policy_version: str | None = None
    adjustments: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    expires_at: datetime | None = None

    @property
    def allowed(self) -> bool:
        """Check if operation is allowed to proceed."""
        return self.decision in (Decision.ALLOW, Decision.WARN)

    @property
    def blocked(self) -> bool:
        """Check if operation is blocked."""
        return self.decision == Decision.DENY


class GovernanceHook(Protocol):
    """Protocol for governance hooks.

    All engines must implement this protocol to enable
    consistent governance across the system.
    """

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Check if an operation should proceed.

        Called before every significant engine operation.
        Can block operations that violate policies.

        Args:
            context: Operation context including inputs.

        Returns:
            PreflightResult with decision and metadata.
        """
        ...

    async def audit(self, record: AuditRecord) -> None:
        """Record an audit event.

        Called after every significant engine operation.
        Should never block or raise exceptions.

        Args:
            record: Audit record to persist.
        """
        ...

    async def on_error(self, error: EngineError) -> None:
        """Handle an engine error.

        Called when an error occurs during operation.
        Used for alerting and error tracking.

        Args:
            error: Structured error information.
        """
        ...


class BaseGovernanceHook:
    """Base class for governance hooks.

    Provides default implementations that can be overridden.
    Engines should extend this class for their hooks.
    """

    def __init__(self, engine_name: str) -> None:
        """Initialize the governance hook.

        Args:
            engine_name: Name of the engine using this hook.
        """
        self.engine_name = engine_name
        self._policy_version = "1.0.0"

    @property
    def policy_version(self) -> str:
        """Get the current policy version."""
        return self._policy_version

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Default preflight: allow all operations.

        Override this method to implement custom policies.

        Args:
            context: Operation context.

        Returns:
            PreflightResult allowing the operation.
        """
        return PreflightResult(
            decision=Decision.ALLOW,
            reason="Default policy: allow all",
            policy_version=self._policy_version,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Default audit: log to standard logger.

        Override this method to persist to database, send to
        external audit system, etc.

        Args:
            record: Audit record to persist.
        """
        import logging

        logger = logging.getLogger(f"ordinis.audit.{self.engine_name}")
        logger.info(
            "AUDIT: engine=%s action=%s trace_id=%s decision=%s",
            record.engine,
            record.action,
            record.trace_id,
            record.decision,
        )

    async def on_error(self, error: EngineError) -> None:
        """Default error handler: log the error.

        Override this method to send alerts, update dashboards, etc.

        Args:
            error: Structured error information.
        """
        import logging

        logger = logging.getLogger(f"ordinis.errors.{self.engine_name}")
        log_level = logging.WARNING if error.recoverable else logging.ERROR
        logger.log(
            log_level,
            "ENGINE_ERROR: code=%s engine=%s recoverable=%s message=%s",
            error.code,
            error.engine,
            error.recoverable,
            error.message,
        )


class CompositeGovernanceHook(BaseGovernanceHook):
    """Combines multiple governance hooks.

    Executes all hooks in order. For preflight, returns the most
    restrictive decision. For audit/error, calls all hooks.
    """

    def __init__(self, engine_name: str, hooks: list[GovernanceHook]) -> None:
        """Initialize with a list of hooks.

        Args:
            engine_name: Name of the engine.
            hooks: List of hooks to compose.
        """
        super().__init__(engine_name)
        self._hooks = hooks

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Execute all preflight hooks, return most restrictive.

        Priority: DENY > DEFER > WARN > ALLOW

        Args:
            context: Operation context.

        Returns:
            Most restrictive PreflightResult from all hooks.
        """
        results: list[PreflightResult] = []
        for hook in self._hooks:
            result = await hook.preflight(context)
            results.append(result)
            # Short-circuit on DENY
            if result.decision == Decision.DENY:
                return result

        # Return most restrictive non-DENY result
        priority = {Decision.DEFER: 0, Decision.WARN: 1, Decision.ALLOW: 2}
        return min(results, key=lambda r: priority.get(r.decision, 2))

    async def audit(self, record: AuditRecord) -> None:
        """Execute all audit hooks.

        Args:
            record: Audit record to persist.
        """
        for hook in self._hooks:
            try:
                await hook.audit(record)
            except Exception:
                # Audit should never fail the operation - log and continue
                _logger.debug("Audit hook failed", exc_info=True)

    async def on_error(self, error: EngineError) -> None:
        """Execute all error hooks.

        Args:
            error: Structured error information.
        """
        for hook in self._hooks:
            try:
                await hook.on_error(error)
            except Exception:
                # Error handling should not raise - log and continue
                _logger.debug("Error hook failed", exc_info=True)
