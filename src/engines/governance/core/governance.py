"""
Governance Engine - Policy Enforcement and Compliance Orchestration.

Orchestrates all governance components:
- Audit Engine (traceability)
- PPI Engine (privacy)
- Ethics Engine (OECD principles)

Implements unified policy enforcement and compliance monitoring.

Reference: https://oecd.ai/en/ai-principles
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from .audit import AuditEngine, AuditEventType
from .ethics import EthicsCheckResult, EthicsEngine
from .ppi import PPIEngine


class PolicyType(Enum):
    """Types of governance policies."""

    TRADING = "trading"  # Trading-related policies
    RISK = "risk"  # Risk management policies
    COMPLIANCE = "compliance"  # Regulatory compliance
    ETHICS = "ethics"  # Ethics/OECD policies
    PRIVACY = "privacy"  # Privacy/PPI policies
    OPERATIONAL = "operational"  # System operational policies


class PolicyAction(Enum):
    """Actions to take on policy violation."""

    ALLOW = "allow"  # Allow with logging
    WARN = "warn"  # Allow with warning
    REVIEW = "review"  # Require human review
    BLOCK = "block"  # Block the action
    ESCALATE = "escalate"  # Escalate to supervisor


class ApprovalStatus(Enum):
    """Status of approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Policy:
    """Governance policy definition."""

    policy_id: str
    name: str
    policy_type: PolicyType
    description: str
    enabled: bool = True

    # Conditions
    condition: str = ""  # Rule expression
    threshold: float | None = None

    # Actions
    action_on_violation: PolicyAction = PolicyAction.WARN
    requires_approval: bool = False
    approval_level: int = 1  # 1 = standard, 2 = manager, 3 = executive

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    last_modified: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "policy_type": self.policy_type.value,
            "description": self.description,
            "enabled": self.enabled,
            "condition": self.condition,
            "threshold": self.threshold,
            "action_on_violation": self.action_on_violation.value,
            "requires_approval": self.requires_approval,
            "approval_level": self.approval_level,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
        }


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""

    decision_id: str
    timestamp: datetime
    policy_id: str
    policy_name: str

    # Decision
    allowed: bool
    action_taken: PolicyAction
    requires_approval: bool

    # Context
    context: dict[str, Any]
    violations: list[str]
    warnings: list[str]

    # Audit linkage
    audit_event_id: str | None = None
    correlation_id: str | None = None


@dataclass
class ApprovalRequest:
    """Request for policy override approval."""

    request_id: str
    timestamp: datetime
    policy_id: str
    requester: str
    reason: str

    # Context
    action_context: dict[str, Any]
    decision_id: str

    # Status
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str | None = None
    approval_time: datetime | None = None
    approval_notes: str | None = None

    # Expiration
    expires_at: datetime | None = None


class GovernanceEngine:
    """
    Governance Engine - Unified policy enforcement.

    Orchestrates:
    - Policy definition and enforcement
    - Audit trail integration
    - PPI protection
    - Ethics compliance
    - Approval workflows

    Implements OECD AI Principle 5: Accountability
    - Systematic risk management
    - Clear responsibility assignment
    - Cooperation between stakeholders

    Reference: https://oecd.ai/en/ai-principles
    """

    def __init__(
        self,
        audit_engine: AuditEngine | None = None,
        ppi_engine: PPIEngine | None = None,
        ethics_engine: EthicsEngine | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize governance engine.

        Args:
            audit_engine: Audit trail engine
            ppi_engine: PPI detection engine
            ethics_engine: Ethics compliance engine
            session_id: Session identifier
        """
        self.session_id = session_id or str(uuid.uuid4())

        # Sub-engines
        self._audit = audit_engine or AuditEngine(session_id=self.session_id)
        self._ppi = ppi_engine or PPIEngine()
        self._ethics = ethics_engine or EthicsEngine()

        # Policies
        self._policies: dict[str, Policy] = {}
        self._initialize_default_policies()

        # Decision history
        self._decisions: list[PolicyDecision] = []

        # Approval workflow
        self._approval_requests: dict[str, ApprovalRequest] = {}
        self._approvers: dict[int, list[str]] = {1: [], 2: [], 3: []}

        # Callbacks
        self._decision_callbacks: list[Callable[[PolicyDecision], None]] = []
        self._approval_callbacks: list[Callable[[ApprovalRequest], None]] = []

        # Log initialization
        self._audit.log_event(
            event_type=AuditEventType.SYSTEM_START,
            actor="GovernanceEngine",
            action="Governance engine initialized",
            details={
                "session_id": self.session_id,
                "policies_loaded": len(self._policies),
            },
        )

    def _initialize_default_policies(self) -> None:
        """Initialize default governance policies."""
        default_policies = [
            Policy(
                policy_id="POL-TRADE-001",
                name="Maximum Position Size",
                policy_type=PolicyType.TRADING,
                description="Limit maximum position size per trade",
                condition="position_value_pct <= threshold",
                threshold=0.10,
                action_on_violation=PolicyAction.BLOCK,
            ),
            Policy(
                policy_id="POL-TRADE-002",
                name="Daily Trade Limit",
                policy_type=PolicyType.TRADING,
                description="Limit number of trades per day",
                condition="daily_trades <= threshold",
                threshold=100,
                action_on_violation=PolicyAction.WARN,
            ),
            Policy(
                policy_id="POL-RISK-001",
                name="Maximum Drawdown",
                policy_type=PolicyType.RISK,
                description="Halt trading on excessive drawdown",
                condition="drawdown_pct <= threshold",
                threshold=-0.10,
                action_on_violation=PolicyAction.BLOCK,
            ),
            Policy(
                policy_id="POL-ETHICS-001",
                name="ESG Compliance",
                policy_type=PolicyType.ETHICS,
                description="Ensure ESG score meets minimum",
                condition="esg_score >= threshold",
                threshold=40.0,
                action_on_violation=PolicyAction.REVIEW,
                requires_approval=True,
            ),
            Policy(
                policy_id="POL-PRIVACY-001",
                name="PPI Protection",
                policy_type=PolicyType.PRIVACY,
                description="Block transmission of detected PPI",
                condition="ppi_detected == False",
                action_on_violation=PolicyAction.BLOCK,
            ),
            Policy(
                policy_id="POL-OPER-001",
                name="Human Oversight Threshold",
                policy_type=PolicyType.OPERATIONAL,
                description="Require human approval for large trades",
                condition="trade_value <= threshold",
                threshold=100000,
                action_on_violation=PolicyAction.REVIEW,
                requires_approval=True,
                approval_level=2,
            ),
        ]

        for policy in default_policies:
            self._policies[policy.policy_id] = policy

    # Policy Management

    def add_policy(self, policy: Policy) -> None:
        """Add or update a policy."""
        self._policies[policy.policy_id] = policy
        self._audit.log_event(
            event_type=AuditEventType.CONFIG_CHANGE,
            actor="GovernanceEngine",
            action=f"Policy added/updated: {policy.name}",
            details=policy.to_dict(),
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        if policy_id in self._policies:
            policy = self._policies.pop(policy_id)
            self._audit.log_event(
                event_type=AuditEventType.CONFIG_CHANGE,
                actor="GovernanceEngine",
                action=f"Policy removed: {policy.name}",
                details={"policy_id": policy_id},
            )
            return True
        return False

    def get_policy(self, policy_id: str) -> Policy | None:
        """Get policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        policy_type: PolicyType | None = None,
        enabled_only: bool = False,
    ) -> list[Policy]:
        """List policies with optional filtering."""
        policies = list(self._policies.values())

        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return policies

    # Core Evaluation

    def evaluate_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        portfolio_state: dict[str, Any],
        strategy: str | None = None,
        signal_explanation: str | None = None,
    ) -> tuple[bool, PolicyDecision, list[EthicsCheckResult]]:
        """
        Comprehensive governance evaluation for a trade.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            quantity: Number of shares
            price: Trade price
            portfolio_state: Current portfolio state
            strategy: Strategy name
            signal_explanation: Explanation for audit trail

        Returns:
            Tuple of (allowed, decision, ethics_results)
        """
        correlation_id = str(uuid.uuid4())
        violations = []
        warnings = []
        requires_approval = False
        action_taken = PolicyAction.ALLOW

        trade_value = quantity * price

        # 1. Run ethics checks
        ethics_approved, ethics_results = self._ethics.check_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            strategy=strategy or "unknown",
            signal_explanation=signal_explanation,
        )

        if not ethics_approved:
            violations.append("Ethics check failed")
            action_taken = PolicyAction.BLOCK

        # 2. Evaluate trading policies
        for policy in self.list_policies(PolicyType.TRADING, enabled_only=True):
            passed, message = self._evaluate_policy(policy, trade_value, portfolio_state)
            if not passed:
                if policy.action_on_violation == PolicyAction.BLOCK:
                    violations.append(f"{policy.name}: {message}")
                    action_taken = PolicyAction.BLOCK
                elif policy.action_on_violation == PolicyAction.WARN:
                    warnings.append(f"{policy.name}: {message}")
                if policy.requires_approval:
                    requires_approval = True

        # 3. Evaluate risk policies
        for policy in self.list_policies(PolicyType.RISK, enabled_only=True):
            passed, message = self._evaluate_policy(policy, trade_value, portfolio_state)
            if not passed:
                violations.append(f"{policy.name}: {message}")
                action_taken = PolicyAction.BLOCK

        # 4. Check operational policies (human oversight)
        for policy in self.list_policies(PolicyType.OPERATIONAL, enabled_only=True):
            if policy.threshold and trade_value > policy.threshold:
                if policy.requires_approval:
                    requires_approval = True
                    warnings.append(
                        f"{policy.name}: Trade value ${trade_value:,.0f} requires approval"
                    )

        # 5. Create decision
        allowed = action_taken != PolicyAction.BLOCK and not requires_approval
        if requires_approval:
            action_taken = PolicyAction.REVIEW

        decision = PolicyDecision(
            decision_id=self._generate_decision_id(),
            timestamp=datetime.utcnow(),
            policy_id="COMPOSITE",
            policy_name="Trade Evaluation",
            allowed=allowed,
            action_taken=action_taken,
            requires_approval=requires_approval,
            context={
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "trade_value": trade_value,
                "strategy": strategy,
            },
            violations=violations,
            warnings=warnings,
            correlation_id=correlation_id,
        )

        # 6. Log to audit trail
        audit_event = self._audit.log_event(
            event_type=AuditEventType.POLICY_APPLIED,
            actor="GovernanceEngine",
            action=f"Trade evaluation: {'ALLOWED' if allowed else 'BLOCKED/REVIEW'}",
            details={
                "decision": decision.decision_id,
                "allowed": allowed,
                "violations": violations,
                "warnings": warnings,
                "ethics_passed": ethics_approved,
            },
            affected_symbols=[symbol],
            correlation_id=correlation_id,
        )
        decision.audit_event_id = audit_event.event_id

        # 7. Store decision
        self._decisions.append(decision)

        # 8. Create approval request if needed
        if requires_approval and not violations:
            self._create_approval_request(
                decision=decision,
                requester="system",
                reason="; ".join(warnings),
            )

        # 9. Trigger callbacks
        for callback in self._decision_callbacks:
            try:
                callback(decision)
            except Exception:
                pass

        return allowed, decision, ethics_results

    def evaluate_data_transmission(
        self,
        data: dict[str, Any],
        destination: str,
    ) -> tuple[bool, dict[str, Any], list]:
        """
        Evaluate data before external transmission.

        Scans for PPI and applies masking.

        Args:
            data: Data to transmit
            destination: Destination identifier

        Returns:
            Tuple of (allowed, masked_data, detections)
        """
        # Scan for PPI
        masked_data, detections = self._ppi.scan_dict(data)

        # Check if transmission should be blocked
        should_block, blocking_categories = self._ppi.should_block_transmission(detections)

        if should_block:
            self._audit.log_event(
                event_type=AuditEventType.PPI_DETECTED,
                actor="GovernanceEngine",
                action=f"Blocked transmission to {destination}",
                details={
                    "blocking_categories": [c.value for c in blocking_categories],
                    "detection_count": len(detections),
                },
            )

        return not should_block, masked_data, detections

    def _evaluate_policy(
        self,
        policy: Policy,
        trade_value: float,
        portfolio_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate a single policy."""
        # Simple policy evaluation based on condition
        if "position_value_pct" in policy.condition and policy.threshold:
            equity = portfolio_state.get("equity", 1)
            pct = trade_value / equity
            if pct > policy.threshold:
                return False, f"Position {pct:.1%} exceeds {policy.threshold:.1%} limit"

        if "daily_trades" in policy.condition and policy.threshold:
            daily_trades = portfolio_state.get("daily_trades", 0)
            if daily_trades >= policy.threshold:
                return False, f"Daily trades {daily_trades} at limit {policy.threshold}"

        if "drawdown_pct" in policy.condition and policy.threshold:
            drawdown = portfolio_state.get("drawdown_pct", 0)
            if drawdown < policy.threshold:  # More negative = worse
                return False, f"Drawdown {drawdown:.1%} exceeds {policy.threshold:.1%}"

        return True, "Policy check passed"

    # Approval Workflow

    def _create_approval_request(
        self,
        decision: PolicyDecision,
        requester: str,
        reason: str,
        expires_hours: int = 24,
    ) -> ApprovalRequest:
        """Create an approval request."""
        request = ApprovalRequest(
            request_id=self._generate_request_id(),
            timestamp=datetime.utcnow(),
            policy_id=decision.policy_id,
            requester=requester,
            reason=reason,
            action_context=decision.context,
            decision_id=decision.decision_id,
            expires_at=datetime.utcnow().replace(hour=datetime.utcnow().hour + expires_hours),
        )

        self._approval_requests[request.request_id] = request

        self._audit.log_event(
            event_type=AuditEventType.HUMAN_REVIEW_REQUESTED,
            actor="GovernanceEngine",
            action=f"Approval requested: {request.request_id}",
            details={
                "requester": requester,
                "reason": reason,
                "decision_id": decision.decision_id,
            },
            correlation_id=decision.correlation_id,
        )

        # Trigger callbacks
        for callback in self._approval_callbacks:
            try:
                callback(request)
            except Exception:
                pass

        return request

    def approve_request(
        self,
        request_id: str,
        approver: str,
        notes: str = "",
    ) -> bool:
        """Approve a pending request."""
        if request_id not in self._approval_requests:
            return False

        request = self._approval_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.APPROVED
        request.approver = approver
        request.approval_time = datetime.utcnow()
        request.approval_notes = notes

        self._audit.log_human_review(
            review_type="approval_request",
            decision="approved",
            reviewer=approver,
            reason=notes,
        )

        return True

    def reject_request(
        self,
        request_id: str,
        approver: str,
        reason: str,
    ) -> bool:
        """Reject a pending request."""
        if request_id not in self._approval_requests:
            return False

        request = self._approval_requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.REJECTED
        request.approver = approver
        request.approval_time = datetime.utcnow()
        request.approval_notes = reason

        self._audit.log_human_review(
            review_type="approval_request",
            decision="rejected",
            reviewer=approver,
            reason=reason,
        )

        return True

    def get_pending_approvals(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return [r for r in self._approval_requests.values() if r.status == ApprovalStatus.PENDING]

    def register_approver(self, approver_id: str, level: int) -> None:
        """Register an approver at a specific level."""
        if level in self._approvers:
            self._approvers[level].append(approver_id)

    # Reporting

    def get_compliance_report(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Combines audit, PPI, and ethics data.
        """
        # Audit summary
        audit_summary = self._audit.get_chain_summary()

        # PPI summary
        ppi_summary = self._ppi.get_detection_summary(start, end)

        # Ethics summary
        ethics_summary = self._ethics.get_compliance_summary(start, end)

        # Decision summary
        decisions = self._decisions
        if start:
            decisions = [d for d in decisions if d.timestamp >= start]
        if end:
            decisions = [d for d in decisions if d.timestamp <= end]

        allowed_count = sum(1 for d in decisions if d.allowed)
        blocked_count = sum(1 for d in decisions if not d.allowed)

        return {
            "report_generated": datetime.utcnow().isoformat(),
            "period": {
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
            "audit": {
                "total_events": audit_summary["total_events"],
                "chain_valid": audit_summary["chain_valid"],
            },
            "privacy": {
                "ppi_detections": ppi_summary["total_detections"],
                "high_risk": ppi_summary["high_risk_count"],
            },
            "ethics": {
                "compliance_rate": ethics_summary["compliance_rate"],
                "checks_performed": ethics_summary["total_checks"],
                "pending_reviews": ethics_summary["pending_reviews"],
            },
            "decisions": {
                "total": len(decisions),
                "allowed": allowed_count,
                "blocked": blocked_count,
                "approval_rate": allowed_count / len(decisions) if decisions else 1.0,
            },
            "pending_approvals": len(self.get_pending_approvals()),
            "oecd_principles": self._get_oecd_compliance_summary(),
        }

    def _get_oecd_compliance_summary(self) -> dict[str, Any]:
        """Get compliance summary for OECD principles."""
        ethics_summary = self._ethics.get_compliance_summary()
        by_principle = ethics_summary.get("by_principle", {})

        return {
            "principle_1_wellbeing": self._calculate_principle_score(
                by_principle, ["inclusive_growth", "sustainable_development", "human_wellbeing"]
            ),
            "principle_2_human_rights": self._calculate_principle_score(
                by_principle, ["human_rights", "fairness", "privacy", "non_discrimination"]
            ),
            "principle_3_transparency": self._calculate_principle_score(
                by_principle, ["transparency", "explainability", "disclosure"]
            ),
            "principle_4_robustness": self._calculate_principle_score(
                by_principle, ["robustness", "security", "safety"]
            ),
            "principle_5_accountability": self._calculate_principle_score(
                by_principle,
                ["accountability", "traceability", "risk_management", "human_oversight"],
            ),
        }

    def _calculate_principle_score(
        self,
        by_principle: dict,
        sub_principles: list[str],
    ) -> float:
        """Calculate aggregate score for a principle category."""
        total_passed = 0
        total_checks = 0

        for sub in sub_principles:
            if sub in by_principle:
                total_passed += by_principle[sub].get("passed", 0)
                total_checks += by_principle[sub].get("passed", 0) + by_principle[sub].get(
                    "failed", 0
                )

        return total_passed / total_checks if total_checks > 0 else 1.0

    # Callbacks

    def register_decision_callback(
        self,
        callback: Callable[[PolicyDecision], None],
    ) -> None:
        """Register callback for policy decisions."""
        self._decision_callbacks.append(callback)

    def register_approval_callback(
        self,
        callback: Callable[[ApprovalRequest], None],
    ) -> None:
        """Register callback for approval requests."""
        self._approval_callbacks.append(callback)

    # ID Generation

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        return f"DEC-{uuid.uuid4().hex[:12].upper()}"

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"REQ-{uuid.uuid4().hex[:12].upper()}"

    # Properties for sub-engine access

    @property
    def audit(self) -> AuditEngine:
        """Access audit engine."""
        return self._audit

    @property
    def ppi(self) -> PPIEngine:
        """Access PPI engine."""
        return self._ppi

    @property
    def ethics(self) -> EthicsEngine:
        """Access ethics engine."""
        return self._ethics
