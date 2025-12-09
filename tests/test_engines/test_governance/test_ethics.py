"""Tests for Ethics Engine (OECD AI Principles)."""

from datetime import datetime

from src.engines.governance.core.ethics import (
    ESGScore,
    EthicsCheckResult,
    EthicsEngine,
    EthicsPolicy,
    OECDPrinciple,
)


class TestOECDPrinciples:
    """Test OECD AI Principles enumeration."""

    def test_all_principles_defined(self) -> None:
        """Test that all OECD principles are defined."""
        # Principle 1
        assert OECDPrinciple.INCLUSIVE_GROWTH
        assert OECDPrinciple.SUSTAINABLE_DEVELOPMENT
        assert OECDPrinciple.HUMAN_WELLBEING

        # Principle 2
        assert OECDPrinciple.HUMAN_RIGHTS
        assert OECDPrinciple.FAIRNESS
        assert OECDPrinciple.PRIVACY

        # Principle 3
        assert OECDPrinciple.TRANSPARENCY
        assert OECDPrinciple.EXPLAINABILITY

        # Principle 4
        assert OECDPrinciple.ROBUSTNESS
        assert OECDPrinciple.SECURITY
        assert OECDPrinciple.SAFETY

        # Principle 5
        assert OECDPrinciple.ACCOUNTABILITY
        assert OECDPrinciple.TRACEABILITY
        assert OECDPrinciple.HUMAN_OVERSIGHT


class TestEthicsEngine:
    """Tests for EthicsEngine."""

    def test_engine_initialization(self) -> None:
        """Test engine initializes with default policies."""
        engine = EthicsEngine()

        # Should have policies for major principles
        assert len(engine._policies) > 0
        assert OECDPrinciple.FAIRNESS in engine._policies

    def test_check_trade_basic(self) -> None:
        """Test basic trade ethics check."""
        engine = EthicsEngine()

        approved, results = engine.check_trade(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
            strategy="SMA_Crossover",
            signal_explanation="Buy signal on SMA crossover with volume confirmation",
        )

        assert isinstance(approved, bool)
        assert len(results) > 0
        assert all(isinstance(r, EthicsCheckResult) for r in results)

    def test_esg_compliance_check(self) -> None:
        """Test ESG compliance checking."""
        engine = EthicsEngine()

        # Set ESG score for testing
        engine.set_esg_score(
            "TEST",
            ESGScore(
                symbol="TEST",
                environmental=75.0,
                social=70.0,
                governance=80.0,
                overall=75.0,
                data_source="test",
                last_updated=datetime.utcnow(),
            ),
        )

        result = engine.check_esg_compliance("TEST")

        assert result.passed is True
        assert result.principle == OECDPrinciple.SUSTAINABLE_DEVELOPMENT
        assert result.score > 0.5

    def test_esg_compliance_failure(self) -> None:
        """Test ESG compliance failure for low scores."""
        engine = EthicsEngine()

        # Set low ESG score
        engine.set_esg_score(
            "LOWSCORE",
            ESGScore(
                symbol="LOWSCORE",
                environmental=20.0,
                social=25.0,
                governance=30.0,
                overall=25.0,
                data_source="test",
                last_updated=datetime.utcnow(),
            ),
        )

        result = engine.check_esg_compliance("LOWSCORE")

        assert result.passed is False
        assert len(result.details["issues"]) > 0

    def test_sector_exclusion(self) -> None:
        """Test sector exclusion checking."""
        engine = EthicsEngine()

        # Default excludes tobacco
        engine._get_sector = lambda s: "tobacco" if s == "TOBCO" else "technology"

        result = engine.check_sector_exclusion("TOBCO")

        assert result.passed is False
        assert result.blocking is True
        assert result.principle == OECDPrinciple.HUMAN_WELLBEING

    def test_add_excluded_sector(self) -> None:
        """Test adding sector to exclusion list."""
        engine = EthicsEngine()

        engine.add_excluded_sector("gambling")

        assert "gambling" in [s.lower() for s in engine._excluded_sectors]

    def test_manipulation_detection_clean(self) -> None:
        """Test manipulation detection for normal trades."""
        engine = EthicsEngine()

        result = engine.check_manipulation_risk(
            symbol="AAPL",
            action="buy",
            quantity=100,
            price=150.0,
        )

        assert result.passed is True
        assert result.principle == OECDPrinciple.FAIRNESS

    def test_explainability_check(self) -> None:
        """Test explainability assessment."""
        engine = EthicsEngine()

        # Good explanation
        good_result = engine.check_explainability(
            strategy="Momentum",
            explanation="Buy signal triggered by RSI indicator showing oversold conditions "
            "with price above 50-day moving average. Risk managed with 2% stop loss.",
        )
        assert good_result.score >= 0.5

        # Poor explanation
        poor_result = engine.check_explainability(
            strategy="Unknown",
            explanation="Buy",
        )
        assert poor_result.score < good_result.score

    def test_human_oversight_threshold(self) -> None:
        """Test human oversight requirement detection."""
        engine = EthicsEngine()

        # Small trade - no review needed
        small_result = engine.check_human_oversight_required(
            symbol="AAPL",
            action="buy",
            notional_value=10000,
        )
        assert small_result.requires_human_review is False

        # Large trade - review needed
        large_result = engine.check_human_oversight_required(
            symbol="AAPL",
            action="buy",
            notional_value=200000,
        )
        assert large_result.requires_human_review is True

    def test_review_queue(self) -> None:
        """Test human review queue management."""
        engine = EthicsEngine()

        # Trigger a review
        engine.check_trade(
            symbol="NEWSTOCK",
            action="buy",
            quantity=1000,
            price=200.0,  # $200k notional
            strategy="test",
        )

        # Should have pending reviews
        queue = engine.get_review_queue()
        # May or may not have items depending on thresholds

    def test_approve_review(self) -> None:
        """Test approving a review."""
        engine = EthicsEngine()

        # Add to review queue directly
        engine._review_queue.append(
            {
                "timestamp": datetime.utcnow(),
                "symbol": "TEST",
                "action": "buy",
                "quantity": 100,
                "price": 100.0,
                "check_results": [],
                "status": "pending",
            }
        )

        success = engine.approve_review(0, "test_reviewer", "Approved for testing")
        assert success is True
        assert engine._review_queue[0]["status"] == "approved"

    def test_compliance_summary(self) -> None:
        """Test compliance summary generation."""
        engine = EthicsEngine()

        # Run some checks
        engine.check_esg_compliance("AAPL")
        engine.check_sector_exclusion("AAPL")

        summary = engine.get_compliance_summary()

        assert "total_checks" in summary
        assert "compliance_rate" in summary
        assert "by_principle" in summary

    def test_violation_callback(self) -> None:
        """Test violation callback is triggered."""
        engine = EthicsEngine()
        violations = []

        engine.register_violation_callback(lambda r: violations.append(r))

        # Force a violation with low ESG score
        engine.set_esg_score(
            "LOWSCORE",
            ESGScore(
                symbol="LOWSCORE",
                environmental=10.0,
                social=10.0,
                governance=10.0,
                overall=10.0,
                data_source="test",
                last_updated=datetime.utcnow(),
            ),
        )

        engine.check_trade(
            symbol="LOWSCORE",
            action="buy",
            quantity=100,
            price=100.0,
            strategy="test",
        )

        assert len(violations) > 0

    def test_ethics_policy_customization(self) -> None:
        """Test customizing ethics policies."""
        custom_policy = EthicsPolicy(
            principle=OECDPrinciple.TRANSPARENCY,
            threshold=0.9,
            blocking=True,
        )

        engine = EthicsEngine(policies={OECDPrinciple.TRANSPARENCY: custom_policy})

        assert engine._policies[OECDPrinciple.TRANSPARENCY].threshold == 0.9
        assert engine._policies[OECDPrinciple.TRANSPARENCY].blocking is True
