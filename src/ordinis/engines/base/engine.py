"""Base engine protocol and abstract class.

This module defines the standard interface that all Ordinis engines
must implement for consistent lifecycle management, health monitoring,
and governance integration.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from typing import Any, Generic, TypeVar
import uuid

from ordinis.engines.base.config import BaseEngineConfig
from ordinis.engines.base.hooks import (
    BaseGovernanceHook,
    GovernanceHook,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.base.models import (
    AuditRecord,
    EngineError,
    EngineMetrics,
    EngineState,
    HealthLevel,
    HealthStatus,
)
from ordinis.engines.base.requirements import RequirementRegistry

# Type variable for config
ConfigT = TypeVar("ConfigT", bound=BaseEngineConfig)


class BaseEngine(ABC, Generic[ConfigT]):
    """Abstract base class for all Ordinis engines.

    Provides standard lifecycle management, health monitoring,
    metrics collection, and governance integration.

    All engines must inherit from this class and implement
    the abstract methods.

    Type Parameters:
        ConfigT: The configuration type for this engine.

    Attributes:
        config: Engine configuration.
        state: Current engine state.
        requirements: Requirement registry for this engine.
    """

    def __init__(
        self,
        config: ConfigT,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Engine configuration.
            governance_hook: Optional governance hook (uses default if None).
        """
        self._config = config
        self._state = EngineState.UNINITIALIZED
        self._started_at: datetime | None = None
        self._logger = logging.getLogger(f"ordinis.engines.{self.name}")

        # Governance
        self._governance = governance_hook or BaseGovernanceHook(self.name)

        # Metrics
        self._requests_total = 0
        self._requests_failed = 0
        self._latencies: list[float] = []

        # Requirements registry
        self._requirements = RequirementRegistry(self.name.upper())

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Engine name for logging and metrics."""
        return self._config.name or self.__class__.__name__

    @property
    def config(self) -> ConfigT:
        """Engine configuration."""
        return self._config

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if engine is operational."""
        return self._state in (EngineState.READY, EngineState.RUNNING)

    @property
    def requirements(self) -> RequirementRegistry:
        """Requirement registry for this engine."""
        return self._requirements

    @property
    def governance(self) -> GovernanceHook:
        """Governance hook for this engine."""
        return self._governance

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle Methods (must be implemented by subclasses)
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Engine-specific initialization logic.

        Subclasses implement this to set up resources, connections,
        load models, etc.

        Raises:
            Exception: If initialization fails.
        """
        ...

    @abstractmethod
    async def _do_shutdown(self) -> None:
        """Engine-specific shutdown logic.

        Subclasses implement this to clean up resources, close
        connections, save state, etc.
        """
        ...

    @abstractmethod
    async def _do_health_check(self) -> HealthStatus:
        """Engine-specific health check logic.

        Subclasses implement this to verify internal health,
        check dependencies, etc.

        Returns:
            HealthStatus with engine-specific diagnostics.
        """
        ...

    # ─────────────────────────────────────────────────────────────────
    # Public Lifecycle API
    # ─────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the engine.

        Sets up resources and transitions to READY state.

        Raises:
            RuntimeError: If engine is already initialized.
            Exception: If initialization fails.
        """
        if self._state != EngineState.UNINITIALIZED:
            raise RuntimeError(f"Engine {self.name} is already initialized (state={self._state})")

        self._state = EngineState.INITIALIZING
        self._logger.info("Initializing engine: %s", self.name)

        try:
            await self._do_initialize()
            self._state = EngineState.READY
            self._started_at = datetime.utcnow()
            self._logger.info("Engine initialized: %s", self.name)
        except Exception as e:
            self._state = EngineState.ERROR
            self._logger.error("Engine initialization failed: %s - %s", self.name, e)
            await self._governance.on_error(
                EngineError(
                    code="INIT_FAILED",
                    message=str(e),
                    engine=self.name,
                    recoverable=False,
                )
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the engine.

        Cleans up resources and transitions to STOPPED state.
        """
        if self._state == EngineState.STOPPED:
            return

        self._state = EngineState.STOPPING
        self._logger.info("Shutting down engine: %s", self.name)

        try:
            await self._do_shutdown()
        except Exception as e:
            self._logger.error("Engine shutdown error: %s - %s", self.name, e)
        finally:
            self._state = EngineState.STOPPED
            self._logger.info("Engine stopped: %s", self.name)

    async def health_check(self) -> HealthStatus:
        """Check engine health.

        Returns:
            HealthStatus with current health information.
        """
        if not self.is_running:
            return HealthStatus(
                level=HealthLevel.UNHEALTHY,
                message=f"Engine not running (state={self._state})",
            )

        start = datetime.utcnow()
        try:
            status = await self._do_health_check()
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return HealthStatus(
                level=status.level,
                message=status.message,
                details=status.details,
                latency_ms=latency,
            )
        except Exception as e:
            return HealthStatus(
                level=HealthLevel.UNHEALTHY,
                message=f"Health check failed: {e}",
            )

    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics.

        Returns:
            EngineMetrics with performance statistics.
        """
        uptime = 0.0
        if self._started_at:
            uptime = (datetime.utcnow() - self._started_at).total_seconds()

        # Calculate percentiles
        sorted_latencies = sorted(self._latencies) if self._latencies else [0.0]
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            if n == 0:
                return 0.0
            idx = int(p * n)
            return sorted_latencies[min(idx, n - 1)]

        return EngineMetrics(
            requests_total=self._requests_total,
            requests_failed=self._requests_failed,
            latency_p50_ms=percentile(0.50),
            latency_p95_ms=percentile(0.95),
            latency_p99_ms=percentile(0.99),
            uptime_seconds=uptime,
        )

    # ─────────────────────────────────────────────────────────────────
    # Governance Integration
    # ─────────────────────────────────────────────────────────────────

    async def preflight(
        self,
        action: str,
        inputs: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> PreflightResult:
        """Run governance preflight check.

        Args:
            action: The action being attempted.
            inputs: Operation inputs (sanitized).
            **metadata: Additional context.

        Returns:
            PreflightResult with decision.
        """
        if not self._config.governance_enabled:
            from ordinis.engines.base.hooks import Decision

            return PreflightResult(decision=Decision.ALLOW, reason="Governance disabled")

        context = PreflightContext(
            engine=self.name,
            action=action,
            inputs=inputs or {},
            trace_id=str(uuid.uuid4()),
            metadata=metadata,
        )
        return await self._governance.preflight(context)

    async def audit(
        self,
        action: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        model_used: str | None = None,
        latency_ms: float | None = None,
        **metadata: Any,
    ) -> None:
        """Record an audit event.

        Args:
            action: The action performed.
            inputs: Operation inputs (sanitized).
            outputs: Operation outputs (sanitized).
            model_used: AI model used (if applicable).
            latency_ms: Operation latency.
            **metadata: Additional context.
        """
        if not self._config.audit_enabled:
            return

        record = AuditRecord(
            engine=self.name,
            action=action,
            inputs=inputs or {},
            outputs=outputs or {},
            model_used=model_used,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        await self._governance.audit(record)

    # ─────────────────────────────────────────────────────────────────
    # Operation Tracking
    # ─────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def track_operation(
        self,
        action: str,
        inputs: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Context manager for tracking operations.

        Handles preflight, metrics, and audit automatically.

        Args:
            action: The action being performed.
            inputs: Operation inputs.

        Yields:
            Context dict for storing outputs.

        Raises:
            PermissionError: If preflight denies the operation.

        Example:
            async with self.track_operation("generate_signal", {"symbol": "AAPL"}) as ctx:
                result = await self._generate_signal("AAPL")
                ctx["outputs"] = {"signal": result}
        """
        # Preflight check
        preflight_result = await self.preflight(action, inputs)
        if preflight_result.blocked:
            raise PermissionError(
                f"Operation denied: {preflight_result.reason} "
                f"(policy={preflight_result.policy_id})"
            )

        # Track timing
        start = datetime.utcnow()
        context: dict[str, Any] = {"outputs": {}, "model_used": None}
        self._requests_total += 1
        error_occurred = False

        try:
            yield context
        except Exception as e:
            error_occurred = True
            self._requests_failed += 1
            await self._governance.on_error(
                EngineError(
                    code="OPERATION_FAILED",
                    message=str(e),
                    engine=self.name,
                    details={"action": action, "inputs": inputs},
                )
            )
            raise
        finally:
            # Record latency
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            self._latencies.append(latency_ms)
            # Keep only last 1000 latencies
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]

            # Audit (even on error)
            await self.audit(
                action=action,
                inputs=inputs,
                outputs=context.get("outputs"),
                model_used=context.get("model_used"),
                latency_ms=latency_ms,
                error=error_occurred,
            )

    # ─────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """String representation of the engine."""
        return f"<{self.__class__.__name__}(name={self.name}, state={self._state.value})>"
