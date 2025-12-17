"""
Tests for RiskGuard engine.

Tests cover:
- Signal evaluation against rules
- Kill switch triggering
- Trade resizing
- Capacity calculation
"""

from datetime import datetime

import pytest

from ordinis.engines.riskguard import STANDARD_RISK_RULES, RiskGuardEngine
from ordinis.engines.riskguard.core.engine import PortfolioState, ProposedTrade
from ordinis.engines.riskguard.core.rules import RiskRule, RuleCategory
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio state."""
    return PortfolioState(
        equity=100000.0,
        cash=50000.0,
        peak_equity=100000.0,
        daily_pnl=0.0,
        daily_trades=0,
        open_positions={},
        total_positions=0,
        total_exposure=0.0,
        sector_exposures={},
        correlated_exposure=0.0,
    )


@pytest.fixture
def mock_signal():
    """Create mock trading signal."""
    return Signal(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.65,
        expected_return=0.05,
        confidence_interval=(0.02, 0.08),
        score=0.7,
        model_id="test_model",
        model_version="1.0.0",
    )


@pytest.mark.unit
def test_engine_initialization():
    """Test RiskGuard engine initialization."""
    engine = RiskGuardEngine(rules={})

    assert len(engine.list_rules()) == 0
    assert engine.is_halted() is False


@pytest.mark.unit
def test_engine_with_standard_rules():
    """Test engine with standard rules."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    assert len(engine.list_rules()) == 25  # 25 standard rules
    assert len(engine.list_rules(enabled_only=True)) == 25


@pytest.mark.unit
def test_add_rule():
    """Test adding a rule to engine."""
    engine = RiskGuardEngine(rules={})

    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test",
        condition="test",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    )

    engine.add_rule(rule)

    assert len(engine.list_rules()) == 1
    assert engine.get_rule("TEST001") == rule


@pytest.mark.unit
def test_remove_rule():
    """Test removing a rule from engine."""
    engine = RiskGuardEngine(rules={})

    rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test",
        condition="test",
        threshold=0.10,
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    )

    engine.add_rule(rule)
    assert len(engine.list_rules()) == 1

    engine.remove_rule("TEST001")
    assert len(engine.list_rules()) == 0


@pytest.mark.unit
def test_list_rules_by_category():
    """Test filtering rules by category."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    pre_trade = engine.list_rules(category=RuleCategory.PRE_TRADE)
    kill_switches = engine.list_rules(category=RuleCategory.KILL_SWITCH)

    assert len(pre_trade) == 4  # RT001-RT004
    assert len(kill_switches) == 7  # RK001-RK007


@pytest.mark.unit
def test_evaluate_signal_passes(mock_portfolio, mock_signal):
    """Test signal evaluation when all rules pass."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Small trade that should pass all rules
    proposed = ProposedTrade(
        symbol="AAPL",
        direction="long",
        quantity=50,  # 50 shares
        entry_price=150.0,  # $7,500 total
        stop_price=145.0,  # $5 stop
    )

    passed, results, adjusted = engine.evaluate_signal(mock_signal, proposed, mock_portfolio)

    assert passed is True
    assert len(results) > 0
    assert all(r.passed for r in results)


@pytest.mark.unit
def test_evaluate_signal_position_too_large(mock_portfolio, mock_signal):
    """Test signal rejection when position size exceeds limit."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Large trade exceeding 10% position limit
    proposed = ProposedTrade(
        symbol="AAPL",
        direction="long",
        quantity=1000,  # 1000 shares
        entry_price=150.0,  # $150,000 total (150% of equity)
        stop_price=145.0,
    )

    passed, results, adjusted = engine.evaluate_signal(mock_signal, proposed, mock_portfolio)

    # Should be resized, not rejected
    assert len(results) > 0

    # Find position size check
    position_check = next(r for r in results if r.rule_id == "RT001")
    assert position_check.action_taken == "resize"


@pytest.mark.unit
def test_evaluate_signal_too_many_positions(mock_portfolio, mock_signal):
    """Test signal rejection when max positions exceeded."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Set portfolio to have 10 positions (at limit)
    mock_portfolio.total_positions = 10

    proposed = ProposedTrade(
        symbol="AAPL", direction="long", quantity=50, entry_price=150.0, stop_price=145.0
    )

    passed, results, adjusted = engine.evaluate_signal(mock_signal, proposed, mock_portfolio)

    assert passed is False

    # Find max positions check
    position_limit = next(r for r in results if r.rule_id == "RP001")
    assert position_limit.passed is False
    assert position_limit.action_taken == "reject"


@pytest.mark.unit
def test_check_kill_switches_daily_loss(mock_portfolio):
    """Test kill switch triggered by daily loss."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Set daily loss to -4% (exceeds -3% limit)
    mock_portfolio.daily_pnl = -4000.0  # -4% of $100k

    triggered, reason = engine.check_kill_switches(mock_portfolio)

    assert triggered is True
    assert reason is not None
    assert "Daily Loss Limit" in reason
    assert engine.is_halted() is True


@pytest.mark.unit
def test_check_kill_switches_max_drawdown(mock_portfolio):
    """Test kill switch triggered by max drawdown."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Set equity to 80k with peak at 100k = 20% drawdown
    mock_portfolio.equity = 80000.0
    mock_portfolio.peak_equity = 100000.0

    triggered, reason = engine.check_kill_switches(mock_portfolio)

    assert triggered is True
    assert reason is not None
    assert "Max Drawdown" in reason


@pytest.mark.unit
def test_check_kill_switches_none_triggered(mock_portfolio):
    """Test kill switches when none are triggered."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    triggered, reason = engine.check_kill_switches(mock_portfolio)

    assert triggered is False
    assert reason is None
    assert engine.is_halted() is False


@pytest.mark.unit
def test_halted_rejects_all_trades(mock_portfolio, mock_signal):
    """Test that halted engine rejects all trades."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    # Trigger halt
    mock_portfolio.daily_pnl = -5000.0
    engine.check_kill_switches(mock_portfolio)

    assert engine.is_halted() is True

    # Try to evaluate a signal
    proposed = ProposedTrade(
        symbol="AAPL", direction="long", quantity=50, entry_price=150.0, stop_price=145.0
    )

    passed, results, adjusted = engine.evaluate_signal(mock_signal, proposed, mock_portfolio)

    assert passed is False
    assert results[0].rule_name == "System Halted"


@pytest.mark.unit
def test_reset_halt():
    """Test resetting halt state."""
    engine = RiskGuardEngine()

    # Manually set halted
    engine._halted = True
    engine._halt_reason = "Test halt"

    assert engine.is_halted() is True

    engine.reset_halt()

    assert engine.is_halted() is False


@pytest.mark.unit
def test_get_available_capacity(mock_portfolio):
    """Test calculating available capacity."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    capacity = engine.get_available_capacity("AAPL", mock_portfolio)

    assert "max_value" in capacity
    assert "limiting_rule" in capacity
    assert capacity["max_value"] == 20000.0  # 20% of 100k equity (RT001)


@pytest.mark.unit
def test_engine_to_dict():
    """Test converting engine state to dictionary."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES)

    state = engine.to_dict()

    assert state["total_rules"] == 25  # 25 standard risk rules
    assert state["enabled_rules"] == 25
    assert state["halted"] is False
    assert "rules" in state
