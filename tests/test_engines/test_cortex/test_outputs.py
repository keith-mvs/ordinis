"""
Tests for Cortex outputs.

Tests cover:
- Output creation and validation
- Strategy hypothesis creation
- Output serialization
"""

from datetime import datetime

import pytest

from src.engines.cortex.core.outputs import (
    CortexOutput,
    OutputType,
    StrategyHypothesis,
)


@pytest.mark.unit
def test_cortex_output_creation():
    """Test creating a Cortex output."""
    output = CortexOutput(
        output_type=OutputType.RESEARCH,
        content={"summary": "Market analysis complete"},
        confidence=0.8,
        reasoning="Analyzed multiple sources",
    )

    assert output.output_type == OutputType.RESEARCH
    assert output.confidence == 0.8
    assert output.requires_validation is True


@pytest.mark.unit
def test_cortex_output_confidence_validation():
    """Test output confidence validation."""
    with pytest.raises(ValueError, match="Confidence must be in"):
        CortexOutput(
            output_type=OutputType.RESEARCH,
            content={},
            confidence=1.5,  # Invalid
            reasoning="test",
        )


@pytest.mark.unit
def test_cortex_output_to_dict():
    """Test converting output to dictionary."""
    now = datetime.utcnow()
    output = CortexOutput(
        output_type=OutputType.HYPOTHESIS,
        content={"strategy": "test"},
        confidence=0.75,
        reasoning="Test reasoning",
        timestamp=now,
        model_used="test-model",
    )

    output_dict = output.to_dict()

    assert output_dict["output_type"] == "hypothesis"
    assert output_dict["confidence"] == 0.75
    assert output_dict["model_used"] == "test-model"
    assert output_dict["timestamp"] == now.isoformat()


@pytest.mark.unit
def test_strategy_hypothesis_creation():
    """Test creating a strategy hypothesis."""
    hypothesis = StrategyHypothesis(
        hypothesis_id="hyp-001",
        name="Test Strategy",
        description="A test trading strategy",
        rationale="For testing purposes",
        instrument_class="equity",
        time_horizon="swing",
        strategy_type="mean_reversion",
        parameters={"rsi_period": 14},
        entry_conditions=["RSI < 30"],
        exit_conditions=["RSI > 70"],
        max_position_size_pct=0.10,
        stop_loss_pct=0.05,
    )

    assert hypothesis.hypothesis_id == "hyp-001"
    assert hypothesis.instrument_class == "equity"
    assert hypothesis.requires_validation is True


@pytest.mark.unit
def test_strategy_hypothesis_to_dict():
    """Test converting hypothesis to dictionary."""
    now = datetime.utcnow()
    hypothesis = StrategyHypothesis(
        hypothesis_id="hyp-001",
        name="Test Strategy",
        description="Test description",
        rationale="Test rationale",
        instrument_class="equity",
        time_horizon="intraday",
        strategy_type="trend_following",
        parameters={"fast_period": 50},
        entry_conditions=["Condition 1"],
        exit_conditions=["Exit 1"],
        max_position_size_pct=0.08,
        stop_loss_pct=0.03,
        created_at=now,
        expected_sharpe=1.5,
    )

    hyp_dict = hypothesis.to_dict()

    assert hyp_dict["hypothesis_id"] == "hyp-001"
    assert hyp_dict["strategy_type"] == "trend_following"
    assert hyp_dict["expected_sharpe"] == 1.5
    assert hyp_dict["created_at"] == now.isoformat()


@pytest.mark.unit
def test_output_types_enum():
    """Test OutputType enum values."""
    assert OutputType.RESEARCH.value == "research"
    assert OutputType.HYPOTHESIS.value == "hypothesis"
    assert OutputType.STRATEGY_SPEC.value == "strategy_spec"
    assert OutputType.CODE_ANALYSIS.value == "code_analysis"
