"""
Tests for Cortex engine.

Tests cover:
- Engine initialization
- Hypothesis generation
- Code analysis
- Research synthesis
- Output review
"""

import pytest

from ordinis.engines.cortex.core.engine import CortexEngine
from ordinis.engines.cortex.core.outputs import OutputType


@pytest.fixture
def cortex_engine():
    """Create Cortex engine without NVIDIA integration."""
    return CortexEngine()


@pytest.fixture
def cortex_with_nvidia():
    """Create Cortex engine with NVIDIA enabled (but no API key)."""
    return CortexEngine(
        nvidia_api_key=None,  # Will use rule-based fallback
        usd_code_enabled=True,
        embeddings_enabled=True,
    )


@pytest.mark.unit
def test_engine_initialization(cortex_engine):
    """Test Cortex engine initialization."""
    assert cortex_engine.nvidia_api_key is None
    assert cortex_engine.usd_code_enabled is False
    assert cortex_engine.embeddings_enabled is False
    assert len(cortex_engine._outputs) == 0
    assert len(cortex_engine._hypotheses) == 0


@pytest.mark.unit
def test_generate_trend_following_hypothesis(cortex_engine):
    """Test generating trend-following hypothesis."""
    market_context = {
        "regime": "trending",
        "volatility": "low",
        "trend_strength": 0.75,
    }

    hypothesis = cortex_engine.generate_hypothesis(market_context)

    assert hypothesis.hypothesis_id.startswith("hyp-")
    assert hypothesis.strategy_type == "trend_following"
    assert hypothesis.name == "SMA Crossover Trend Following"
    assert hypothesis.confidence == 0.75
    assert len(hypothesis.entry_conditions) > 0


@pytest.mark.unit
def test_generate_mean_reversion_hypothesis(cortex_engine):
    """Test generating mean reversion hypothesis."""
    market_context = {
        "regime": "mean_reverting",
        "volatility": "high",
    }

    hypothesis = cortex_engine.generate_hypothesis(market_context)

    assert hypothesis.strategy_type == "mean_reversion"
    assert hypothesis.name == "RSI Mean Reversion"
    assert "rsi_period" in hypothesis.parameters


@pytest.mark.unit
def test_generate_balanced_hypothesis(cortex_engine):
    """Test generating balanced hypothesis for unknown conditions."""
    market_context = {
        "regime": "unknown",
        "volatility": "medium",
    }

    hypothesis = cortex_engine.generate_hypothesis(market_context)

    assert hypothesis.strategy_type == "adaptive"
    assert hypothesis.confidence == 0.60


@pytest.mark.unit
def test_generate_hypothesis_with_constraints(cortex_engine):
    """Test hypothesis generation with constraints."""
    market_context = {"regime": "trending", "volatility": "low"}
    constraints = {
        "instrument_class": "options",
        "max_position_pct": 0.05,
    }

    hypothesis = cortex_engine.generate_hypothesis(market_context, constraints)

    assert hypothesis.instrument_class == "options"
    assert hypothesis.max_position_size_pct == 0.05


@pytest.mark.unit
def test_analyze_code(cortex_engine):
    """Test code analysis."""
    code = """
def calculate_rsi(prices, period=14):
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    return 100 - (100 / (1 + sum(gains) / sum(losses)))
"""

    output = cortex_engine.analyze_code(code, "review")

    assert output.output_type == OutputType.CODE_ANALYSIS
    assert "analysis" in output.content
    assert output.confidence > 0.0


@pytest.mark.unit
def test_synthesize_research(cortex_engine):
    """Test research synthesis."""
    query = "What are the best risk management practices for trading?"
    sources = [
        "https://example.com/risk-management",
        "https://example.com/position-sizing",
    ]
    context = {"focus": "practical_application"}

    output = cortex_engine.synthesize_research(query, sources, context)

    assert output.output_type == OutputType.RESEARCH
    assert output.content["query"] == query
    assert output.content["research"]["sources_analyzed"] == 2


@pytest.mark.unit
def test_review_signalcore_output(cortex_engine):
    """Test reviewing SignalCore output."""
    signalcore_output = {
        "signal_type": "entry",
        "probability": 0.55,
        "expected_return": 0.05,
    }

    output = cortex_engine.review_output("signalcore", signalcore_output)

    assert output.output_type == OutputType.REVIEW
    assert output.content["engine"] == "signalcore"
    assert "assessment" in output.content


@pytest.mark.unit
def test_review_riskguard_halted(cortex_engine):
    """Test reviewing RiskGuard halted state."""
    riskguard_output = {
        "halted": True,
        "halt_reason": "Daily loss limit exceeded",
    }

    output = cortex_engine.review_output("riskguard", riskguard_output)

    assert output.content["assessment"] == "needs_attention"
    assert len(output.content["concerns"]) > 0


@pytest.mark.unit
def test_get_hypothesis(cortex_engine):
    """Test retrieving hypothesis by ID."""
    market_context = {"regime": "trending", "volatility": "low"}
    hypothesis = cortex_engine.generate_hypothesis(market_context)

    retrieved = cortex_engine.get_hypothesis(hypothesis.hypothesis_id)

    assert retrieved is not None
    assert retrieved.hypothesis_id == hypothesis.hypothesis_id


@pytest.mark.unit
def test_list_hypotheses(cortex_engine):
    """Test listing hypotheses."""
    # Generate multiple hypotheses
    for regime in ["trending", "mean_reverting", "unknown"]:
        cortex_engine.generate_hypothesis({"regime": regime, "volatility": "medium"})

    all_hypotheses = cortex_engine.list_hypotheses()
    assert len(all_hypotheses) == 3

    high_confidence = cortex_engine.list_hypotheses(min_confidence=0.70)
    assert len(high_confidence) <= 3


@pytest.mark.unit
def test_get_outputs_by_type(cortex_engine):
    """Test getting outputs filtered by type."""
    # Generate various outputs
    cortex_engine.generate_hypothesis({"regime": "trending", "volatility": "low"})
    cortex_engine.analyze_code("def test(): pass")
    cortex_engine.synthesize_research("test query", ["source1"])

    all_outputs = cortex_engine.get_outputs()
    assert len(all_outputs) == 3

    hypotheses = cortex_engine.get_outputs(OutputType.HYPOTHESIS)
    assert len(hypotheses) == 1
    assert all(o.output_type == OutputType.HYPOTHESIS for o in hypotheses)


@pytest.mark.unit
def test_engine_to_dict(cortex_engine):
    """Test converting engine state to dictionary."""
    # Generate some outputs
    cortex_engine.generate_hypothesis({"regime": "trending", "volatility": "low"})
    cortex_engine.analyze_code("code")

    state = cortex_engine.to_dict()

    assert state["total_outputs"] == 2
    assert state["total_hypotheses"] == 1
    assert "outputs_by_type" in state
