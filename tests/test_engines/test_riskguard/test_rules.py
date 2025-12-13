"""
Tests for RiskGuard rules and evaluation.

Tests cover:
- Rule creation and validation
- Rule evaluation logic
- Rule comparisons
"""

from datetime import datetime

import pytest

from ordinis.engines.riskguard.core.rules import RiskCheckResult, RiskRule, RuleCategory


@pytest.mark.unit
def test_risk_rule_creation():
    """Test creating a risk rule."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test description",
        condition="value <= threshold",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    )

    assert rule.rule_id == "TEST001"
    assert rule.category == RuleCategory.PRE_TRADE
    assert rule.threshold == 0.10
    assert rule.enabled is True


@pytest.mark.unit
def test_risk_rule_evaluate_less_than():
    """Test rule evaluation with < comparison."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test",
        condition="value < threshold",
        threshold=0.10,
        comparison="<",
        action_on_breach="reject",
        severity="high",
    )

    assert rule.evaluate(0.05) is True  # Pass: 0.05 < 0.10
    assert rule.evaluate(0.10) is False  # Fail: 0.10 not < 0.10
    assert rule.evaluate(0.15) is False  # Fail: 0.15 not < 0.10


@pytest.mark.unit
def test_risk_rule_evaluate_less_equal():
    """Test rule evaluation with <= comparison."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test",
        condition="value <= threshold",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    )

    assert rule.evaluate(0.05) is True  # Pass
    assert rule.evaluate(0.10) is True  # Pass
    assert rule.evaluate(0.15) is False  # Fail


@pytest.mark.unit
def test_risk_rule_evaluate_greater_than():
    """Test rule evaluation with > comparison."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.KILL_SWITCH,
        name="Test Rule",
        description="Test",
        condition="value > threshold",
        threshold=-0.03,
        comparison=">",
        action_on_breach="halt",
        severity="critical",
    )

    assert rule.evaluate(-0.01) is True  # Pass: -0.01 > -0.03
    assert rule.evaluate(-0.03) is False  # Fail
    assert rule.evaluate(-0.05) is False  # Fail


@pytest.mark.unit
def test_risk_rule_evaluate_greater_equal():
    """Test rule evaluation with >= comparison."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.KILL_SWITCH,
        name="Test Rule",
        description="Test",
        condition="value >= threshold",
        threshold=-0.03,
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    )

    assert rule.evaluate(-0.01) is True  # Pass
    assert rule.evaluate(-0.03) is True  # Pass
    assert rule.evaluate(-0.05) is False  # Fail


@pytest.mark.unit
def test_risk_rule_disabled():
    """Test that disabled rules always pass."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test",
        condition="value <= threshold",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
        enabled=False,
    )

    # Even though value exceeds threshold, rule is disabled
    assert rule.evaluate(1.0) is True


@pytest.mark.unit
def test_risk_rule_to_dict():
    """Test converting rule to dictionary."""
    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test description",
        condition="value <= threshold",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
        last_modified=datetime(2024, 1, 1),
        modified_by="test_user",
    )

    rule_dict = rule.to_dict()

    assert rule_dict["rule_id"] == "TEST001"
    assert rule_dict["category"] == "pre_trade"
    assert rule_dict["threshold"] == 0.10
    assert rule_dict["modified_by"] == "test_user"


@pytest.mark.unit
def test_risk_check_result_creation():
    """Test creating risk check result."""
    result = RiskCheckResult(
        rule_id="TEST001",
        rule_name="Test Rule",
        passed=False,
        current_value=0.15,
        threshold=0.10,
        comparison="<=",
        message="Position size exceeds limit",
        action_taken="reject",
        severity="high",
        timestamp=datetime.utcnow(),
    )

    assert result.rule_id == "TEST001"
    assert result.passed is False
    assert result.current_value == 0.15
    assert result.action_taken == "reject"


@pytest.mark.unit
def test_risk_check_result_to_dict():
    """Test converting check result to dictionary."""
    now = datetime.utcnow()
    result = RiskCheckResult(
        rule_id="TEST001",
        rule_name="Test Rule",
        passed=True,
        current_value=0.05,
        threshold=0.10,
        comparison="<=",
        message="Position size within limit",
        action_taken="pass",
        severity="high",
        timestamp=now,
    )

    result_dict = result.to_dict()

    assert result_dict["rule_id"] == "TEST001"
    assert result_dict["passed"] is True
    assert result_dict["timestamp"] == now.isoformat()
