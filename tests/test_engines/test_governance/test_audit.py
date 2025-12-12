"""Tests for Audit Engine."""

from datetime import datetime

from src.engines.governance.core.audit import (
    AuditEngine,
    AuditEvent,
    AuditEventType,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test basic event creation."""
        event = AuditEvent(
            event_id="EVT-TEST001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="TestActor",
            action="Test action",
            previous_hash="0" * 64,
        )

        assert event.event_id == "EVT-TEST001"
        assert event.event_type == AuditEventType.SIGNAL_GENERATED
        assert event.event_hash != ""
        assert len(event.event_hash) == 64  # SHA-256 hex

    def test_event_hash_changes_with_content(self) -> None:
        """Test that hash changes when content changes."""
        event1 = AuditEvent(
            event_id="EVT-001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ORDER_CREATED,
            actor="Actor",
            action="Action 1",
            previous_hash="0" * 64,
        )

        event2 = AuditEvent(
            event_id="EVT-001",
            timestamp=event1.timestamp,
            event_type=AuditEventType.ORDER_CREATED,
            actor="Actor",
            action="Action 2",  # Different action
            previous_hash="0" * 64,
        )

        assert event1.event_hash != event2.event_hash

    def test_event_integrity_verification(self) -> None:
        """Test event integrity verification."""
        event = AuditEvent(
            event_id="EVT-TEST",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.RISK_CHECK_PASSED,
            actor="RiskGuard",
            action="Check passed",
            previous_hash="abc123",
        )

        assert event.verify_integrity() is True

    def test_event_serialization(self) -> None:
        """Test event to_dict and from_dict."""
        original = AuditEvent(
            event_id="EVT-SERIAL",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ORDER_FILLED,
            actor="FlowRoute",
            action="Order filled",
            details={"order_id": "ORD-123", "price": 150.50},
            affected_symbols=["AAPL"],
            previous_hash="prev_hash",
        )

        data = original.to_dict()
        restored = AuditEvent.from_dict(data)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.details == original.details
        assert restored.affected_symbols == original.affected_symbols


class TestAuditEngine:
    """Tests for AuditEngine."""

    def test_engine_initialization(self) -> None:
        """Test engine initializes correctly."""
        engine = AuditEngine(environment="test")

        assert engine.session_id is not None
        assert engine.environment == "test"
        # Should have logged system start
        summary = engine.get_chain_summary()
        assert summary["total_events"] >= 1

    def test_log_event(self) -> None:
        """Test basic event logging."""
        engine = AuditEngine()

        event = engine.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="SignalCore",
            action="Generated buy signal",
            details={"confidence": 0.85},
            affected_symbols=["MSFT"],
        )

        assert event.event_id is not None
        assert event.event_type == AuditEventType.SIGNAL_GENERATED
        assert event.actor == "SignalCore"
        assert "MSFT" in event.affected_symbols

    def test_chain_linkage(self) -> None:
        """Test that events are properly chained."""
        engine = AuditEngine()

        event1 = engine.log_event(
            event_type=AuditEventType.ORDER_CREATED,
            actor="Test",
            action="First event",
        )

        event2 = engine.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            actor="Test",
            action="Second event",
        )

        # Second event should link to first
        assert event2.previous_hash == event1.event_hash

    def test_chain_integrity_verification(self) -> None:
        """Test chain integrity verification."""
        engine = AuditEngine()

        # Log several events
        for i in range(5):
            engine.log_event(
                event_type=AuditEventType.SIGNAL_GENERATED,
                actor="Test",
                action=f"Event {i}",
            )

        is_valid, errors = engine.verify_chain_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_get_event_by_id(self) -> None:
        """Test retrieving event by ID."""
        engine = AuditEngine()

        event = engine.log_event(
            event_type=AuditEventType.RISK_CHECK_PASSED,
            actor="RiskGuard",
            action="All checks passed",
        )

        retrieved = engine.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id

    def test_correlation_tracking(self) -> None:
        """Test events can be linked by correlation ID."""
        engine = AuditEngine()
        correlation_id = "TRADE-001"

        engine.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="SignalCore",
            action="Signal generated",
            correlation_id=correlation_id,
        )

        engine.log_event(
            event_type=AuditEventType.ORDER_CREATED,
            actor="FlowRoute",
            action="Order created",
            correlation_id=correlation_id,
        )

        engine.log_event(
            event_type=AuditEventType.ORDER_FILLED,
            actor="FlowRoute",
            action="Order filled",
            correlation_id=correlation_id,
        )

        correlated = engine.get_events_by_correlation(correlation_id)
        assert len(correlated) == 3

    def test_decision_trace(self) -> None:
        """Test getting full decision trace."""
        engine = AuditEngine()

        event1 = engine.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="SignalCore",
            action="Root signal",
        )

        event2 = engine.log_event(
            event_type=AuditEventType.RISK_CHECK_PASSED,
            actor="RiskGuard",
            action="Risk approved",
            parent_event_id=event1.event_id,
        )

        event3 = engine.log_event(
            event_type=AuditEventType.ORDER_CREATED,
            actor="FlowRoute",
            action="Order created",
            parent_event_id=event2.event_id,
        )

        trace = engine.get_decision_trace(event3.event_id)
        assert len(trace) == 3
        assert trace[0].event_id == event1.event_id
        assert trace[2].event_id == event3.event_id

    def test_get_events_by_type(self) -> None:
        """Test filtering events by type."""
        engine = AuditEngine()

        engine.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="Test",
            action="Signal 1",
        )
        engine.log_event(
            event_type=AuditEventType.ORDER_CREATED,
            actor="Test",
            action="Order",
        )
        engine.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            actor="Test",
            action="Signal 2",
        )

        signals = engine.get_events_by_type(AuditEventType.SIGNAL_GENERATED)
        assert len(signals) == 2

    def test_convenience_methods(self) -> None:
        """Test convenience logging methods."""
        engine = AuditEngine()

        # Test log_signal
        signal_event = engine.log_signal(
            symbol="AAPL",
            signal_type="buy",
            confidence=0.9,
            source="SMA_Crossover",
        )
        assert signal_event.event_type == AuditEventType.SIGNAL_GENERATED

        # Test log_risk_check
        risk_event = engine.log_risk_check(
            passed=True,
            rule_id="RULE-001",
            rule_name="Position Size",
            current_value=0.05,
            threshold=0.10,
            symbol="AAPL",
        )
        assert risk_event.event_type == AuditEventType.RISK_CHECK_PASSED

    def test_export_chain(self) -> None:
        """Test exporting audit chain."""
        engine = AuditEngine()

        for i in range(3):
            engine.log_event(
                event_type=AuditEventType.SIGNAL_GENERATED,
                actor="Test",
                action=f"Event {i}",
            )

        exported = engine.export_chain()
        assert len(exported) >= 4  # 3 + system start
        assert all("event_id" in e for e in exported)
        assert all("event_hash" in e for e in exported)
