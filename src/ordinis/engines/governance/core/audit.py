"""
Audit Engine - Immutable audit trail with hash chaining.

Implements OECD AI Principle 5: Accountability
- Traceability of datasets, processes, and decisions
- Analysis of AI system outputs and responses to inquiry
- Systematic risk management approach

Reference: https://oecd.ai/en/ai-principles
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any
import uuid


class AuditEventType(Enum):
    """Types of auditable events in the trading system."""

    # Trading lifecycle
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_FILTERED = "signal_filtered"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk management
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_OVERRIDE = "risk_override"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    POSITION_RESIZED = "position_resized"

    # Governance
    POLICY_APPLIED = "policy_applied"
    POLICY_VIOLATION = "policy_violation"
    ETHICS_CHECK = "ethics_check"
    PPI_DETECTED = "ppi_detected"
    PPI_MASKED = "ppi_masked"

    # System
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    ERROR_OCCURRED = "error_occurred"
    MANUAL_INTERVENTION = "manual_intervention"

    # AI/Model
    MODEL_PREDICTION = "model_prediction"
    MODEL_OVERRIDE = "model_override"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"


@dataclass
class AuditEvent:
    """
    Immutable audit event with cryptographic integrity.

    Each event links to the previous via hash chain, ensuring
    tamper-evidence and full traceability.
    """

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    actor: str  # System component or user identifier
    action: str  # Description of what happened

    # Context
    details: dict[str, Any] = field(default_factory=dict)
    affected_symbols: list[str] = field(default_factory=list)

    # Traceability (OECD Principle 5)
    correlation_id: str | None = None  # Links related events
    parent_event_id: str | None = None  # Direct causation

    # Hash chain
    previous_hash: str = ""
    event_hash: str = field(default="", init=False)

    # Metadata
    session_id: str | None = None
    environment: str = "production"
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Calculate event hash after initialization."""
        self.event_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of event content.

        Includes all fields except event_hash itself.
        """
        content = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "affected_symbols": self.affected_symbols,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "previous_hash": self.previous_hash,
            "session_id": self.session_id,
            "environment": self.environment,
            "version": self.version,
        }

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event hash matches content."""
        return self.event_hash == self._calculate_hash()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "affected_symbols": self.affected_symbols,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "session_id": self.session_id,
            "environment": self.environment,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create event from dictionary."""
        event = cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            actor=data["actor"],
            action=data["action"],
            details=data.get("details", {}),
            affected_symbols=data.get("affected_symbols", []),
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id"),
            previous_hash=data.get("previous_hash", ""),
            session_id=data.get("session_id"),
            environment=data.get("environment", "production"),
            version=data.get("version", "1.0"),
        )
        return event


class AuditEngine:
    """
    Immutable audit trail engine with hash chaining.

    Implements OECD AI Principles for accountability:
    - Full traceability of all decisions
    - Tamper-evident log chain
    - Support for inquiry and analysis

    Reference: https://oecd.ai/en/ai-principles
    """

    # Genesis hash for first event in chain
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        session_id: str | None = None,
        environment: str = "production",
        storage_backend: Any = None,
    ) -> None:
        """
        Initialize audit engine.

        Args:
            session_id: Unique session identifier
            environment: Environment name (production, staging, etc.)
            storage_backend: Optional storage backend for persistence
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.environment = environment
        self.storage = storage_backend

        # In-memory chain
        self._events: list[AuditEvent] = []
        self._last_hash: str = self.GENESIS_HASH

        # Index for fast lookup
        self._event_index: dict[str, int] = {}
        self._correlation_index: dict[str, list[str]] = {}
        self._type_index: dict[AuditEventType, list[str]] = {}

        # Log system start
        self.log_event(
            event_type=AuditEventType.SYSTEM_START,
            actor="AuditEngine",
            action="Audit trail initialized",
            details={
                "session_id": self.session_id,
                "environment": self.environment,
            },
        )

    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        details: dict[str, Any] | None = None,
        affected_symbols: list[str] | None = None,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> AuditEvent:
        """
        Log an audit event to the chain.

        Args:
            event_type: Type of event
            actor: Component or user that triggered the event
            action: Description of what happened
            details: Additional event details
            affected_symbols: Ticker symbols involved
            correlation_id: ID linking related events
            parent_event_id: Direct parent event

        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            actor=actor,
            action=action,
            details=details or {},
            affected_symbols=affected_symbols or [],
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            previous_hash=self._last_hash,
            session_id=self.session_id,
            environment=self.environment,
        )

        # Add to chain
        self._events.append(event)
        self._last_hash = event.event_hash

        # Update indexes
        self._event_index[event.event_id] = len(self._events) - 1

        if correlation_id:
            if correlation_id not in self._correlation_index:
                self._correlation_index[correlation_id] = []
            self._correlation_index[correlation_id].append(event.event_id)

        if event_type not in self._type_index:
            self._type_index[event_type] = []
        self._type_index[event_type].append(event.event_id)

        # Persist if storage backend available
        if self.storage:
            self._persist_event(event)

        return event

    def get_event(self, event_id: str) -> AuditEvent | None:
        """Get event by ID."""
        if event_id not in self._event_index:
            return None
        return self._events[self._event_index[event_id]]

    def get_events_by_correlation(self, correlation_id: str) -> list[AuditEvent]:
        """Get all events linked by correlation ID."""
        if correlation_id not in self._correlation_index:
            return []
        return [
            self._events[self._event_index[eid]] for eid in self._correlation_index[correlation_id]
        ]

    def get_events_by_type(
        self,
        event_type: AuditEventType,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Get events by type."""
        if event_type not in self._type_index:
            return []

        event_ids = self._type_index[event_type]
        if limit:
            event_ids = event_ids[-limit:]

        return [self._events[self._event_index[eid]] for eid in event_ids]

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        event_types: list[AuditEventType] | None = None,
    ) -> list[AuditEvent]:
        """Get events within time range."""
        events = [e for e in self._events if start <= e.timestamp <= end]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events

    def get_decision_trace(self, event_id: str) -> list[AuditEvent]:
        """
        Get full decision trace leading to an event.

        Follows parent_event_id chain to reconstruct
        the full decision path (OECD traceability requirement).
        """
        trace = []
        current_id = event_id

        while current_id:
            event = self.get_event(current_id)
            if not event:
                break
            trace.append(event)
            current_id = event.parent_event_id

        return list(reversed(trace))

    def verify_chain_integrity(self) -> tuple[bool, list[str]]:
        """
        Verify integrity of entire audit chain.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if not self._events:
            return True, []

        # Verify first event links to genesis
        if self._events[0].previous_hash != self.GENESIS_HASH:
            errors.append(f"First event does not link to genesis hash: {self._events[0].event_id}")

        # Verify each event
        for i, event in enumerate(self._events):
            # Verify self-integrity
            if not event.verify_integrity():
                errors.append(f"Event hash mismatch: {event.event_id}")

            # Verify chain linkage
            if i > 0:
                expected_prev = self._events[i - 1].event_hash
                if event.previous_hash != expected_prev:
                    errors.append(
                        f"Chain break at event {event.event_id}: "
                        f"expected {expected_prev[:8]}..., "
                        f"got {event.previous_hash[:8]}..."
                    )

        return len(errors) == 0, errors

    def get_chain_summary(self) -> dict[str, Any]:
        """Get summary statistics of audit chain."""
        type_counts = {}
        for event_type in AuditEventType:
            count = len(self._type_index.get(event_type, []))
            if count > 0:
                type_counts[event_type.value] = count

        return {
            "session_id": self.session_id,
            "environment": self.environment,
            "total_events": len(self._events),
            "event_types": type_counts,
            "first_event": self._events[0].timestamp.isoformat() if self._events else None,
            "last_event": self._events[-1].timestamp.isoformat() if self._events else None,
            "chain_valid": self.verify_chain_integrity()[0],
            "last_hash": self._last_hash[:16] + "...",
        }

    def export_chain(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Export audit chain for external analysis."""
        events = self._events

        if start:
            events = [e for e in events if e.timestamp >= start]
        if end:
            events = [e for e in events if e.timestamp <= end]

        return [e.to_dict() for e in events]

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"EVT-{uuid.uuid4().hex[:12].upper()}"

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist event to storage backend."""
        if self.storage and hasattr(self.storage, "write"):
            self.storage.write(event.to_dict())

    # Convenience methods for common events

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        source: str,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> AuditEvent:
        """Log a trading signal generation."""
        return self.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor=source,
            action=f"Generated {signal_type} signal for {symbol}",
            details={
                "signal_type": signal_type,
                "confidence": confidence,
                **(details or {}),
            },
            affected_symbols=[symbol],
            correlation_id=correlation_id,
        )

    def log_order(
        self,
        event_type: AuditEventType,
        symbol: str,
        order_id: str,
        details: dict[str, Any],
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> AuditEvent:
        """Log an order lifecycle event."""
        return self.log_event(
            event_type=event_type,
            actor="FlowRoute",
            action=f"Order {event_type.value}: {order_id}",
            details={"order_id": order_id, **details},
            affected_symbols=[symbol],
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
        )

    def log_risk_check(
        self,
        passed: bool,
        rule_id: str,
        rule_name: str,
        current_value: float,
        threshold: float,
        symbol: str | None = None,
        correlation_id: str | None = None,
    ) -> AuditEvent:
        """Log a risk check result."""
        return self.log_event(
            event_type=(
                AuditEventType.RISK_CHECK_PASSED if passed else AuditEventType.RISK_CHECK_FAILED
            ),
            actor="RiskGuard",
            action=f"Risk check {'passed' if passed else 'failed'}: {rule_name}",
            details={
                "rule_id": rule_id,
                "rule_name": rule_name,
                "current_value": current_value,
                "threshold": threshold,
                "passed": passed,
            },
            affected_symbols=[symbol] if symbol else [],
            correlation_id=correlation_id,
        )

    def log_ethics_check(
        self,
        principle: str,
        passed: bool,
        score: float,
        details: dict[str, Any],
        correlation_id: str | None = None,
    ) -> AuditEvent:
        """Log an ethics/OECD principle check."""
        return self.log_event(
            event_type=AuditEventType.ETHICS_CHECK,
            actor="EthicsEngine",
            action=f"OECD Principle check: {principle}",
            details={
                "principle": principle,
                "passed": passed,
                "score": score,
                **details,
            },
            correlation_id=correlation_id,
        )

    def log_human_review(
        self,
        review_type: str,
        decision: str,
        reviewer: str,
        reason: str,
        correlation_id: str | None = None,
    ) -> AuditEvent:
        """Log human review decision (meaningful human oversight)."""
        return self.log_event(
            event_type=AuditEventType.HUMAN_REVIEW_COMPLETED,
            actor=reviewer,
            action=f"Human review completed: {review_type}",
            details={
                "review_type": review_type,
                "decision": decision,
                "reason": reason,
            },
            correlation_id=correlation_id,
        )
