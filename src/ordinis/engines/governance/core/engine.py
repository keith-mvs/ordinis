"""
Unified Governance Engine.

Standardized engine extending BaseEngine for governance operations.
Wraps GovernanceEngine with lifecycle management and governance hooks.
"""

from datetime import UTC, datetime
from typing import Any

from ordinis.engines.base import (
    BaseEngine,
    EngineMetrics,
    GovernanceHook,
    HealthLevel,
    HealthStatus,
)
from ordinis.engines.governance.core.audit import AuditEngine
from ordinis.engines.governance.core.config import GovernanceEngineConfig
from ordinis.engines.governance.core.ethics import EthicsEngine
from ordinis.engines.governance.core.governance import (
    GovernanceEngine,
    Policy,
    PolicyDecision,
    PolicyType,
)
from ordinis.engines.governance.core.ppi import PPIEngine


class UnifiedGovernanceEngine(BaseEngine[GovernanceEngineConfig]):
    """Unified governance engine extending BaseEngine.

    Wraps GovernanceEngine with standardized lifecycle management,
    health checks, and metrics tracking. Provides preflight/audit
    hooks for integration with other engines.

    Example:
        >>> from ordinis.engines.governance import (
        ...     UnifiedGovernanceEngine,
        ...     GovernanceEngineConfig,
        ... )
        >>> config = GovernanceEngineConfig(
        ...     enable_ethics_checks=True,
        ...     max_position_size_pct=0.05,
        ... )
        >>> engine = UnifiedGovernanceEngine(config)
        >>> await engine.initialize()
        >>> decision = engine.evaluate_trade(
        ...     symbol="AAPL",
        ...     action="buy",
        ...     quantity=100,
        ...     price=150.0,
        ... )
    """

    def __init__(
        self,
        config: GovernanceEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the Unified Governance engine.

        Args:
            config: Engine configuration (uses defaults if None)
            governance_hook: Optional external governance hook
        """
        super().__init__(config or GovernanceEngineConfig(), governance_hook)

        self._governance: GovernanceEngine | None = None
        self._evaluations_count: int = 0
        self._last_evaluation: datetime | None = None
        self._blocked_count: int = 0
        self._approved_count: int = 0

    async def _do_initialize(self) -> None:
        """Initialize governance engine resources."""
        # Create sub-engines based on config
        audit_engine = None
        ppi_engine = None
        ethics_engine = None

        if self.config.enable_audit:
            audit_engine = AuditEngine(session_id=self.config.session_id)

        if self.config.enable_ppi_scanning:
            ppi_engine = PPIEngine()

        if self.config.enable_ethics_checks:
            ethics_engine = EthicsEngine()

        # Create main governance engine
        self._governance = GovernanceEngine(
            audit_engine=audit_engine,
            ppi_engine=ppi_engine,
            ethics_engine=ethics_engine,
            session_id=self.config.session_id,
        )

        # Apply configuration to default policies
        if self.config.enable_default_policies:
            self._apply_config_to_policies()

        # Reset counters
        self._evaluations_count = 0
        self._last_evaluation = None
        self._blocked_count = 0
        self._approved_count = 0

    def _apply_config_to_policies(self) -> None:
        """Apply configuration values to default policies."""
        if self._governance is None:
            return

        # Update position size policy
        pos_policy = self._governance.get_policy("POL-TRADE-001")
        if pos_policy:
            pos_policy.threshold = self.config.max_position_size_pct

        # Update daily trade limit policy
        trade_policy = self._governance.get_policy("POL-TRADE-002")
        if trade_policy:
            trade_policy.threshold = self.config.daily_trade_limit

        # Update drawdown policy
        dd_policy = self._governance.get_policy("POL-RISK-001")
        if dd_policy:
            dd_policy.threshold = self.config.max_drawdown_pct

        # Update human approval threshold policy
        approval_policy = self._governance.get_policy("POL-OPER-001")
        if approval_policy:
            approval_policy.threshold = self.config.require_human_approval_above

        # Update ESG threshold policy
        esg_policy = self._governance.get_policy("POL-ETHICS-001")
        if esg_policy:
            esg_policy.threshold = self.config.ethics_esg_threshold

    async def _do_shutdown(self) -> None:
        """Shutdown governance engine resources."""
        self._governance = None

    async def _do_health_check(self) -> HealthStatus:
        """Check governance engine health.

        Returns:
            Current health status
        """
        issues: list[str] = []

        if self._governance is None:
            issues.append("Governance engine not initialized")
        else:
            # Check sub-engines
            if self.config.enable_audit and self._governance.audit_engine is None:
                issues.append("Audit engine not available")
            if self.config.enable_ppi_scanning and self._governance.ppi_engine is None:
                issues.append("PPI engine not available")
            if self.config.enable_ethics_checks and self._governance.ethics_engine is None:
                issues.append("Ethics engine not available")

        level = HealthLevel.HEALTHY if not issues else HealthLevel.DEGRADED
        return HealthStatus(
            level=level,
            message="Governance engine operational" if not issues else "; ".join(issues),
            details={
                "evaluations_count": self._evaluations_count,
                "approved_count": self._approved_count,
                "blocked_count": self._blocked_count,
                "last_evaluation": (
                    self._last_evaluation.isoformat() if self._last_evaluation else None
                ),
                "sub_engines": {
                    "audit": self.config.enable_audit,
                    "ppi": self.config.enable_ppi_scanning,
                    "ethics": self.config.enable_ethics_checks,
                },
            },
        )

    def evaluate_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        strategy: str | None = None,
        signal_explanation: str | None = None,
        account_value: float | None = None,
        current_position_value: float | None = None,
        **kwargs: Any,
    ) -> PolicyDecision:
        """Evaluate a trade against governance policies.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            quantity: Number of shares
            price: Trade price
            strategy: Strategy name
            signal_explanation: Explanation for audit trail
            account_value: Total account value
            current_position_value: Current position value in symbol
            **kwargs: Additional context

        Returns:
            PolicyDecision object

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._governance is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        decision = self._governance.evaluate_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            strategy=strategy,
            signal_explanation=signal_explanation,
            account_value=account_value,
            current_position_value=current_position_value,
            **kwargs,
        )

        # Update metrics
        self._evaluations_count += 1
        self._last_evaluation = datetime.now(UTC)
        if decision.allowed:
            self._approved_count += 1
        else:
            self._blocked_count += 1

        return decision

    def evaluate_data_transmission(
        self,
        data: dict[str, Any],
        destination: str = "unknown",
    ) -> PolicyDecision:
        """Evaluate data before external transmission.

        Args:
            data: Data to transmit
            destination: Destination identifier

        Returns:
            PolicyDecision object

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._governance is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        decision = self._governance.evaluate_data_transmission(data, destination)

        # Update metrics
        self._evaluations_count += 1
        self._last_evaluation = datetime.now(UTC)
        if decision.allowed:
            self._approved_count += 1
        else:
            self._blocked_count += 1

        return decision

    def add_policy(self, policy: Policy) -> None:
        """Add or update a governance policy.

        Args:
            policy: Policy to add

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._governance is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._governance.add_policy(policy)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a governance policy.

        Args:
            policy_id: ID of policy to remove

        Returns:
            True if policy was removed

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._governance is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        return self._governance.remove_policy(policy_id)

    def get_policy(self, policy_id: str) -> Policy | None:
        """Get a policy by ID.

        Args:
            policy_id: Policy ID

        Returns:
            Policy or None
        """
        if self._governance is None:
            return None
        return self._governance.get_policy(policy_id)

    def list_policies(
        self,
        policy_type: PolicyType | None = None,
        enabled_only: bool = False,
    ) -> list[Policy]:
        """List governance policies.

        Args:
            policy_type: Filter by policy type
            enabled_only: Only return enabled policies

        Returns:
            List of policies
        """
        if self._governance is None:
            return []
        return self._governance.list_policies(policy_type, enabled_only)

    def get_compliance_report(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive compliance report.

        Args:
            start: Report start date
            end: Report end date

        Returns:
            Compliance report dictionary
        """
        if self._governance is None:
            return {"error": "Engine not initialized"}
        return self._governance.get_compliance_report(start, end)

    def get_pending_approvals(self) -> list:
        """Get pending approval requests.

        Returns:
            List of pending approval requests
        """
        if self._governance is None:
            return []
        return self._governance.get_pending_approvals()

    def approve_request(
        self,
        request_id: str,
        approver: str,
        notes: str = "",
    ) -> bool:
        """Approve a pending request.

        Args:
            request_id: Request ID
            approver: Approver identifier
            notes: Approval notes

        Returns:
            True if approved successfully
        """
        if self._governance is None:
            return False
        return self._governance.approve_request(request_id, approver, notes)

    def reject_request(
        self,
        request_id: str,
        approver: str,
        reason: str,
    ) -> bool:
        """Reject a pending request.

        Args:
            request_id: Request ID
            approver: Approver identifier
            reason: Rejection reason

        Returns:
            True if rejected successfully
        """
        if self._governance is None:
            return False
        return self._governance.reject_request(request_id, approver, reason)

    @property
    def audit_engine(self) -> AuditEngine | None:
        """Access the audit engine."""
        if self._governance is None:
            return None
        return self._governance.audit_engine

    @property
    def ppi_engine(self) -> PPIEngine | None:
        """Access the PPI engine."""
        if self._governance is None:
            return None
        return self._governance.ppi_engine

    @property
    def ethics_engine(self) -> EthicsEngine | None:
        """Access the ethics engine."""
        if self._governance is None:
            return None
        return self._governance.ethics_engine

    def get_metrics(self) -> EngineMetrics:
        """Get governance engine metrics.

        Returns:
            Current engine metrics
        """
        metrics = super().get_metrics()
        metrics.custom_metrics.update(
            {
                "evaluations_count": self._evaluations_count,
                "approved_count": self._approved_count,
                "blocked_count": self._blocked_count,
                "approval_rate": (
                    self._approved_count / self._evaluations_count
                    if self._evaluations_count > 0
                    else 1.0
                ),
                "last_evaluation": (
                    self._last_evaluation.isoformat() if self._last_evaluation else None
                ),
                "policies_count": len(self.list_policies()),
                "pending_approvals": len(self.get_pending_approvals()),
            }
        )
        return metrics
