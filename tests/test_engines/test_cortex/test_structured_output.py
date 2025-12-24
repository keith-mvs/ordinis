"""Tests for CortexEngine structured output (JSON mode) hypothesis generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.engines.cortex.core.config import CortexConfig
from ordinis.engines.cortex.core.outputs import StructuredHypothesis


class TestStructuredHypothesis:
    """Test StructuredHypothesis Pydantic model."""

    def test_valid_hypothesis(self) -> None:
        """Test creating a valid structured hypothesis."""
        hyp = StructuredHypothesis(
            name="Test Strategy",
            description="A test trading strategy",
            rationale="Based on mean reversion in high volatility",
            instrument_class="equity",
            time_horizon="swing",
            strategy_type="mean_reversion",
            entry_conditions=["RSI < 30", "Price below SMA"],
            exit_conditions=["RSI > 70", "Stop loss hit"],
            max_position_size_pct=5.0,
            stop_loss_pct=2.0,
            confidence=0.75,
        )
        
        assert hyp.name == "Test Strategy"
        assert hyp.strategy_type == "mean_reversion"
        assert hyp.confidence == 0.75
        assert len(hyp.entry_conditions) == 2

    def test_default_values(self) -> None:
        """Test default values are applied."""
        hyp = StructuredHypothesis(
            name="Minimal Strategy",
            description="A minimal strategy",
            rationale="Just testing defaults",
            entry_conditions=["Price signal"],
            exit_conditions=["Exit signal"],
        )
        
        assert hyp.instrument_class == "equity"  # default
        assert hyp.time_horizon == "swing"  # default
        assert hyp.strategy_type == "adaptive"  # default
        assert hyp.max_position_size_pct == 5.0  # default
        assert hyp.stop_loss_pct == 2.0  # default
        assert hyp.confidence == 0.5  # default

    def test_confidence_bounds(self) -> None:
        """Test confidence must be in [0, 1]."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StructuredHypothesis(
                name="Bad Confidence",
                description="Test",
                rationale="Test",
                entry_conditions=["test"],
                exit_conditions=["test"],
                confidence=1.5,  # Invalid - > 1.0
            )

    def test_strategy_type_validation(self) -> None:
        """Test strategy_type must be a valid literal."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StructuredHypothesis(
                name="Bad Strategy Type",
                description="Test",
                rationale="Test",
                entry_conditions=["test"],
                exit_conditions=["test"],
                strategy_type="invalid_strategy",  # Not in Literal
            )

    def test_risk_parameter_bounds(self) -> None:
        """Test risk parameters have valid bounds."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StructuredHypothesis(
                name="Bad Risk Params",
                description="Test",
                rationale="Test",
                entry_conditions=["test"],
                exit_conditions=["test"],
                max_position_size_pct=50.0,  # > 25% limit
            )


class TestStructuredOutputGeneration:
    """Test structured output hypothesis generation."""

    @pytest.mark.asyncio
    async def test_structured_output_enabled_by_default(self) -> None:
        """Test structured output is enabled in default config."""
        config = CortexConfig()
        assert config.use_structured_output is True

    @pytest.mark.asyncio
    async def test_structured_output_can_be_disabled(self) -> None:
        """Test structured output can be disabled."""
        config = CortexConfig(use_structured_output=False)
        assert config.use_structured_output is False

    @pytest.mark.asyncio
    async def test_structured_output_fallback_to_rules(self) -> None:
        """Test that structured output falls back to rules on failure."""
        from ordinis.engines.cortex.core.engine import CortexEngine

        # Mock Helix
        mock_helix = MagicMock()
        mock_helix.initialize = AsyncMock()
        mock_helix.shutdown = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Test rationale from LLM"
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        mock_helix.generate = AsyncMock(return_value=mock_response)
        mock_helix.health_check = AsyncMock(return_value=MagicMock(level=MagicMock(value="healthy")))

        # Create engine with structured output disabled
        config = CortexConfig(
            governance_enabled=False,
            use_structured_output=False,  # Disable to test rule-based path
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        await engine.initialize()

        # Generate hypothesis
        hypothesis = await engine.generate_hypothesis(
            market_context={"regime": "trending", "volatility": "low"},
        )

        # Should use rule-based trend following
        assert hypothesis.strategy_type == "trend_following"
        assert "SMA Crossover" in hypothesis.name

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_structured_output_uses_llm_when_available(self) -> None:
        """Test that structured output uses LLM-generated values when successful."""
        from ordinis.engines.cortex.core.engine import CortexEngine

        # Mock Helix
        mock_helix = MagicMock()
        mock_helix.initialize = AsyncMock()
        mock_helix.shutdown = AsyncMock()
        mock_helix.health_check = AsyncMock(return_value=MagicMock(level=MagicMock(value="healthy")))

        # Mock the structured output result
        mock_structured = StructuredHypothesis(
            name="LLM Generated Strategy",
            description="A strategy generated by the LLM",
            rationale="AI determined this based on market conditions",
            strategy_type="momentum",
            entry_conditions=["Momentum breakout detected"],
            exit_conditions=["Momentum fades"],
            confidence=0.85,
        )

        # Create engine
        config = CortexConfig(
            governance_enabled=False,
            use_structured_output=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)

        # Patch the structured generation method
        with patch.object(
            engine,
            "_generate_structured_hypothesis",
            AsyncMock(return_value=(mock_structured, "test-model", 0, 0)),
        ):
            await engine.initialize()

            hypothesis = await engine.generate_hypothesis(
                market_context={"regime": "trending", "volatility": "low"},
            )

            # Should use LLM-generated values
            assert hypothesis.name == "LLM Generated Strategy"
            assert hypothesis.strategy_type == "momentum"
            assert hypothesis.confidence == 0.85
            assert "source" in hypothesis.metadata
            assert hypothesis.metadata["source"] == "structured_output"

        await engine.shutdown()


class TestStructuredHypothesisToDict:
    """Test StructuredHypothesis serialization."""

    def test_model_dump(self) -> None:
        """Test Pydantic model serialization."""
        hyp = StructuredHypothesis(
            name="Serialization Test",
            description="Testing model_dump",
            rationale="For serialization",
            entry_conditions=["Entry"],
            exit_conditions=["Exit"],
        )
        
        data = hyp.model_dump()
        
        assert isinstance(data, dict)
        assert data["name"] == "Serialization Test"
        assert data["strategy_type"] == "adaptive"
        assert "entry_conditions" in data

    def test_model_json(self) -> None:
        """Test JSON serialization."""
        hyp = StructuredHypothesis(
            name="JSON Test",
            description="Testing JSON output",
            rationale="For testing JSON serialization",
            entry_conditions=["Entry"],
            exit_conditions=["Exit"],
            parameters={"lookback": 20, "threshold": 0.5},
        )
        
        json_str = hyp.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "JSON Test" in json_str
        assert "lookback" in json_str
