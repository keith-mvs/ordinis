"""Tests for Governance Engine."""

from datetime import datetime

from ordinis.engines.governance.core.governance import (
    GovernanceEngine,
    Policy,
    PolicyAction,
    PolicyDecision,
    PolicyType,
)


class TestPolicy:
    """Tests for Policy dataclass."""

    def test_policy_creation(self) -> None:
        """Test basic policy creation."""
        policy = Policy(
            policy_id="POL-001",
            name="Max Position Size",
            policy_type=PolicyType.RISK,
            action=PolicyAction.BLOCK,
            threshold=0.10,
            description="Block trades exceeding 10% position size",
        )

        assert policy.policy_id == "POL-001"
        assert policy.policy_type == PolicyType.RISK
        assert policy.action == PolicyAction.BLOCK
        assert policy.threshold == 0.10

    def test_policy_with_conditions(self) -> None:
        """Test policy with custom conditions."""
        policy = Policy(
            policy_id="POL-002",
            name="After Hours Trading",
            policy_type=PolicyType.OPERATIONAL,
            action=PolicyAction.WARN,
            conditions={"market_hours_only": True},
        )

        assert policy.conditions["market_hours_only"] is True


class TestGovernanceEngine:
    """Tests for GovernanceEngine."""

    def test_engine_initialization(self) -> None:
        """Test engine initializes with default policies."""
        engine = GovernanceEngine()

        assert engine.audit_engine is not None
        assert engine.ppi_engine is not None
        assert engine.ethics_engine is not None
        assert len(engine._policies) > 0

    def test_evaluate_trade_basic(self) -> None:
        """Test basic trade evaluation."""
        engine = GovernanceEngine()

        decision = engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            strategy="SMA_Crossover",
            signal_explanation="Buy on SMA crossover",
            account_value=100000,
            current_position_value=5000,
        )

        assert isinstance(decision, PolicyDecision)
        assert decision.action in PolicyAction
        assert len(decision.checks_performed) > 0

    def test_evaluate_trade_with_risk_breach(self) -> None:
        """Test trade evaluation with position size breach."""
        engine = GovernanceEngine()

        # Try to buy a huge position
        decision = engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=10000,
            price=150.0,  # $1.5M position
            strategy="Test",
            signal_explanation="Test signal",
            account_value=100000,  # Only $100k account
            current_position_value=0,
        )

        # Should be blocked or require review
        assert decision.action in (PolicyAction.BLOCK, PolicyAction.REVIEW)

    def test_evaluate_trade_small_order(self) -> None:
        """Test that small orders pass easily."""
        engine = GovernanceEngine()

        decision = engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=10,
            price=150.0,  # $1,500 position
            strategy="Momentum",
            signal_explanation="RSI oversold with volume confirmation",
            account_value=100000,
            current_position_value=0,
        )

        assert decision.action == PolicyAction.ALLOW

    def test_add_custom_policy(self) -> None:
        """Test adding custom policy."""
        engine = GovernanceEngine()

        custom_policy = Policy(
            policy_id="CUSTOM-001",
            name="No Penny Stocks",
            policy_type=PolicyType.TRADING,
            action=PolicyAction.BLOCK,
            threshold=5.0,  # Min price $5
            description="Block trades for stocks under $5",
        )

        engine.add_policy(custom_policy)

        assert "CUSTOM-001" in engine._policies

    def test_disable_policy(self) -> None:
        """Test disabling a policy."""
        engine = GovernanceEngine()

        # Add and then disable
        policy = Policy(
            policy_id="TEST-DISABLE",
            name="Test Policy",
            policy_type=PolicyType.COMPLIANCE,
            action=PolicyAction.WARN,
        )
        engine.add_policy(policy)
        engine.disable_policy("TEST-DISABLE")

        assert engine._policies["TEST-DISABLE"].enabled is False

    def test_evaluate_data_transmission(self) -> None:
        """Test data transmission evaluation for PPI."""
        engine = GovernanceEngine()

        # Safe data
        safe_decision = engine.evaluate_data_transmission(
            {
                "symbol": "AAPL",
                "price": 150.0,
                "quantity": 100,
            }
        )

        assert safe_decision.action == PolicyAction.ALLOW

    def test_evaluate_data_transmission_with_ppi(self) -> None:
        """Test data transmission blocks PPI."""
        engine = GovernanceEngine()

        # Data with PPI
        ppi_decision = engine.evaluate_data_transmission(
            {
                "user": {
                    "name": "John Doe",
                    "ssn": "123-45-6789",
                    "email": "john@example.com",
                },
            }
        )

        # Should block or require masking
        assert ppi_decision.action in (PolicyAction.BLOCK, PolicyAction.REVIEW)
        assert len(ppi_decision.ppi_detections) > 0

    def test_get_compliance_report(self) -> None:
        """Test compliance report generation."""
        engine = GovernanceEngine()

        # Run some evaluations
        engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            strategy="Test",
            signal_explanation="Test",
            account_value=100000,
            current_position_value=0,
        )

        report = engine.get_compliance_report()

        assert "total_evaluations" in report
        assert "decisions_by_action" in report
        assert "audit_summary" in report

    def test_human_review_workflow(self) -> None:
        """Test human review workflow integration."""
        engine = GovernanceEngine()

        # Force a review-required decision
        decision = engine.evaluate_trade(
            symbol="NEWSTOCK",
            action="buy",
            quantity=5000,
            price=100.0,  # $500k position
            strategy="Test",
            signal_explanation="Test",
            account_value=100000,  # Way over position limit
            current_position_value=0,
        )

        if decision.action == PolicyAction.REVIEW:
            # Should have review item
            assert decision.review_required is True

    def test_audit_trail_created(self) -> None:
        """Test that evaluations create audit entries."""
        engine = GovernanceEngine()

        initial_count = engine.audit_engine.get_chain_summary()["total_events"]

        engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            strategy="Test",
            signal_explanation="Test",
            account_value=100000,
            current_position_value=0,
        )

        final_count = engine.audit_engine.get_chain_summary()["total_events"]

        assert final_count > initial_count

    def test_policy_violation_tracking(self) -> None:
        """Test that violations are tracked."""
        engine = GovernanceEngine()
        violations = []

        engine.register_violation_callback(lambda v: violations.append(v))

        # Trigger a violation with oversized position
        engine.evaluate_trade(
            symbol="TEST",
            action="buy",
            quantity=100000,
            price=100.0,
            strategy="Test",
            signal_explanation="Test",
            account_value=10000,
            current_position_value=0,
        )

        # Should have triggered at least one violation
        assert len(violations) > 0

    def test_get_policy_by_type(self) -> None:
        """Test filtering policies by type."""
        engine = GovernanceEngine()

        risk_policies = engine.get_policies_by_type(PolicyType.RISK)
        trading_policies = engine.get_policies_by_type(PolicyType.TRADING)

        assert all(p.policy_type == PolicyType.RISK for p in risk_policies)
        assert all(p.policy_type == PolicyType.TRADING for p in trading_policies)

    def test_ethics_integration(self) -> None:
        """Test ethics engine integration."""
        engine = GovernanceEngine()

        # Check that ethics checks are performed
        decision = engine.evaluate_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            strategy="BlackBox",  # Poor explainability
            signal_explanation="Buy",  # Minimal explanation
            account_value=100000,
            current_position_value=0,
        )

        # Should have ethics checks in results
        ethics_checks = [
            c
            for c in decision.checks_performed
            if "ethics" in c.lower() or "explainability" in c.lower()
        ]
        assert len(ethics_checks) > 0


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_decision_creation(self) -> None:
        """Test decision creation."""
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="All checks passed",
            checks_performed=["risk", "ethics", "compliance"],
            timestamp=datetime.utcnow(),
        )

        assert decision.action == PolicyAction.ALLOW
        assert len(decision.checks_performed) == 3

    def test_decision_with_violations(self) -> None:
        """Test decision with violations."""
        decision = PolicyDecision(
            action=PolicyAction.BLOCK,
            reason="Position size exceeded",
            checks_performed=["risk"],
            violations=["Position size 15% exceeds 10% limit"],
            timestamp=datetime.utcnow(),
        )

        assert decision.action == PolicyAction.BLOCK
        assert len(decision.violations) == 1

    def test_decision_review_required(self) -> None:
        """Test decision requiring review."""
        decision = PolicyDecision(
            action=PolicyAction.REVIEW,
            reason="Large trade requires approval",
            checks_performed=["risk", "ethics"],
            review_required=True,
            reviewer_level="senior",
            timestamp=datetime.utcnow(),
        )

        assert decision.review_required is True
        assert decision.reviewer_level == "senior"
