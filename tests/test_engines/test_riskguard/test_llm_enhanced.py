"""
Tests for LLM-enhanced RiskGuard.

Tests cover:
- LLM-enhanced engine wrapping
- Trade evaluation explanations
- Risk scenario analysis
- Rule optimization suggestions
"""

from datetime import datetime

import pytest

from src.engines.riskguard import STANDARD_RISK_RULES, RiskGuardEngine
from src.engines.riskguard.core.engine import PortfolioState, ProposedTrade
from src.engines.riskguard.core.llm_enhanced import LLMEnhancedRiskGuard, LLMRiskAnalyzer
from src.engines.signalcore.core.signal import Direction, Signal, SignalType


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio state."""
    return PortfolioState(
        equity=100000.0,
        cash=50000.0,
        peak_equity=100000.0,
        daily_pnl=1500.0,
        daily_trades=3,
        open_positions={},
        total_positions=2,
        total_exposure=30000.0,
        sector_exposures={"Technology": 15000.0},
        correlated_exposure=5000.0,
    )


@pytest.fixture
def mock_signal():
    """Create mock trading signal."""
    return Signal(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.75,
        expected_return=0.08,
        confidence_interval=(0.05, 0.12),
        score=0.8,
        model_id="test_model",
        model_version="1.0.0",
    )


@pytest.fixture
def proposed_trade():
    """Create proposed trade."""
    return ProposedTrade(
        symbol="AAPL",
        direction="long",
        quantity=100,
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        sector="Technology",
    )


@pytest.fixture
def base_engine():
    """Create base RiskGuard engine."""
    engine = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())
    return engine


@pytest.mark.unit
def test_llm_enhanced_creation_no_llm(base_engine):
    """Test creating LLM-enhanced engine without LLM."""
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine, llm_enabled=False)

    assert enhanced.llm_enabled is False
    assert enhanced.nvidia_api_key is None
    assert len(enhanced.list_rules()) == len(base_engine.list_rules())


@pytest.mark.unit
def test_llm_enhanced_creation_with_api_key(base_engine):
    """Test creating LLM-enhanced engine with API key."""
    enhanced = LLMEnhancedRiskGuard(
        base_engine=base_engine, nvidia_api_key="test-key", llm_enabled=True
    )

    assert enhanced.llm_enabled is True
    assert enhanced.nvidia_api_key == "test-key"


@pytest.mark.unit
def test_llm_enhanced_creation_no_base():
    """Test creating LLM-enhanced engine without base engine."""
    enhanced = LLMEnhancedRiskGuard(llm_enabled=False)

    assert enhanced.llm_enabled is False
    assert len(enhanced.list_rules()) == 0


@pytest.mark.unit
def test_llm_enhanced_evaluate_no_llm(base_engine, mock_signal, proposed_trade, mock_portfolio):
    """Test evaluation without LLM."""
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine, llm_enabled=False)

    passed, results, adjusted_signal = enhanced.evaluate_signal(
        mock_signal, proposed_trade, mock_portfolio
    )

    # Basic evaluation should work
    assert isinstance(passed, bool)
    assert len(results) > 0
    # No LLM explanation
    assert "risk_explanation" not in mock_signal.metadata


@pytest.mark.unit
def test_llm_enhanced_evaluate_with_llm_no_api(
    base_engine, mock_signal, proposed_trade, mock_portfolio
):
    """Test evaluation with LLM enabled but no API key."""
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine, nvidia_api_key=None, llm_enabled=True)

    passed, results, adjusted_signal = enhanced.evaluate_signal(
        mock_signal, proposed_trade, mock_portfolio
    )

    # Should work but without LLM explanation
    assert isinstance(passed, bool)
    # No explanation added without API key
    if adjusted_signal:
        assert "risk_explanation" not in adjusted_signal.metadata


@pytest.mark.unit
def test_llm_enhanced_inherits_rules(base_engine):
    """Test that enhanced engine inherits rules from base."""
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine)

    base_rules = base_engine.list_rules()
    enhanced_rules = enhanced.list_rules()

    assert len(enhanced_rules) == len(base_rules)
    assert all(r.rule_id in [er.rule_id for er in enhanced_rules] for r in base_rules)


@pytest.mark.unit
def test_llm_enhanced_inherits_halt_state(base_engine):
    """Test that enhanced engine inherits halt state."""
    # Halt the base engine
    base_engine._halted = True
    base_engine._halt_reason = "Test halt"

    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine)

    assert enhanced.is_halted()
    assert enhanced._halt_reason == "Test halt"


@pytest.mark.unit
def test_risk_analyzer_creation():
    """Test creating risk analyzer."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key=None)

    assert analyzer.nvidia_api_key is None
    assert analyzer._llm_client is None


@pytest.mark.unit
def test_risk_analyzer_scenario_no_api(mock_portfolio):
    """Test scenario analysis without API key (fallback)."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key=None)

    result = analyzer.analyze_risk_scenario("Market crash scenario", mock_portfolio)

    # Should return basic analysis
    assert "scenario" in result
    assert "analysis" in result
    assert result["llm_model"] == "rule-based"


@pytest.mark.unit
def test_risk_analyzer_scenario_with_api(mock_portfolio):
    """Test scenario analysis with API key (will use fallback)."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key="test-key")

    result = analyzer.analyze_risk_scenario("High volatility scenario", mock_portfolio)

    # Should return analysis (fallback since no real API)
    assert "scenario" in result
    assert "analysis" in result
    assert isinstance(result, dict)


@pytest.mark.unit
def test_risk_analyzer_optimize_no_api(mock_portfolio):
    """Test rule optimization without API key."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key=None)

    performance = {"sharpe_ratio": 1.5, "max_drawdown": -0.15, "win_rate": 0.55}

    suggestions = analyzer.suggest_rule_optimization(mock_portfolio, performance)

    # Should return basic suggestions
    assert len(suggestions) > 0
    assert isinstance(suggestions, list)
    assert all(isinstance(s, str) for s in suggestions)


@pytest.mark.unit
def test_risk_analyzer_optimize_with_api(mock_portfolio):
    """Test rule optimization with API key (will use fallback)."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key="test-key")

    performance = {"sharpe_ratio": 0.8, "max_drawdown": -0.25, "win_rate": 0.45}

    suggestions = analyzer.suggest_rule_optimization(mock_portfolio, performance)

    # Should return suggestions (fallback since no real API)
    assert len(suggestions) > 0
    assert isinstance(suggestions, list)


@pytest.mark.unit
def test_risk_analyzer_explain_rule_no_api():
    """Test rule explanation without API key."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key=None)

    explanation = analyzer.explain_rule("Max Position Size", "Limits position to 10% of equity")

    assert "Max Position Size" in explanation
    assert isinstance(explanation, str)


@pytest.mark.unit
def test_risk_analyzer_explain_rule_with_api():
    """Test rule explanation with API key (will use fallback)."""
    analyzer = LLMRiskAnalyzer(nvidia_api_key="test-key")

    explanation = analyzer.explain_rule("Daily Loss Limit", "Halts trading at -3% daily loss")

    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.unit
def test_llm_enhanced_preserves_functionality(
    base_engine, mock_signal, proposed_trade, mock_portfolio
):
    """Test that LLM enhancement preserves base functionality."""
    # Evaluate with base engine
    base_passed, base_results, _ = base_engine.evaluate_signal(
        mock_signal, proposed_trade, mock_portfolio
    )

    # Evaluate with enhanced engine (LLM disabled)
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine, llm_enabled=False)
    enhanced_passed, enhanced_results, _ = enhanced.evaluate_signal(
        mock_signal, proposed_trade, mock_portfolio
    )

    # Results should match
    assert base_passed == enhanced_passed
    assert len(base_results) == len(enhanced_results)


@pytest.mark.unit
def test_llm_enhanced_add_rule():
    """Test adding rules to enhanced engine."""
    enhanced = LLMEnhancedRiskGuard(llm_enabled=False)

    from src.engines.riskguard.core.rules import RiskRule, RuleCategory

    test_rule = RiskRule(
        rule_id="TEST001",
        category=RuleCategory.PRE_TRADE,
        name="Test Rule",
        description="Test rule description",
        condition="test condition",
        threshold=0.5,
        comparison="<=",
        action_on_breach="reject",
        severity="medium",
    )

    enhanced.add_rule(test_rule)

    assert len(enhanced.list_rules()) == 1
    assert enhanced.get_rule("TEST001") == test_rule


@pytest.mark.unit
def test_llm_enhanced_to_dict(base_engine):
    """Test engine state serialization."""
    enhanced = LLMEnhancedRiskGuard(base_engine=base_engine, llm_enabled=True)

    state = enhanced.to_dict()

    assert "rules" in state
    assert "total_rules" in state
    assert "enabled_rules" in state
    assert "halted" in state
