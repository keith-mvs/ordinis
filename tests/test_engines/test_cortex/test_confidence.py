"""
Tests for Cortex confidence calibration using reward model.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ordinis.engines.cortex.core.confidence import (
    ConfidenceCalibrator,
    ConfidenceConfig,
    ConfidenceScore,
    QualityDimension,
)


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_score_creation(self):
        """Test creating a confidence score."""
        score = ConfidenceScore(
            overall=0.85,
            helpfulness=0.9,
            correctness=0.8,
            coherence=0.85,
            complexity=0.7,
            verbosity=0.75,
        )

        assert score.overall == 0.85
        assert score.helpfulness == 0.9
        assert score.correctness == 0.8

    def test_is_high_confidence(self):
        """Test high confidence threshold."""
        high = ConfidenceScore(overall=0.75)
        low = ConfidenceScore(overall=0.65)

        assert high.is_high_confidence is True
        assert low.is_high_confidence is False

    def test_is_acceptable(self):
        """Test acceptable threshold."""
        acceptable = ConfidenceScore(overall=0.5)
        unacceptable = ConfidenceScore(overall=0.4)

        assert acceptable.is_acceptable is True
        assert unacceptable.is_acceptable is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        score = ConfidenceScore(
            overall=0.8,
            helpfulness=0.9,
            correctness=0.85,
            latency_ms=150.0,
        )

        result = score.to_dict()

        assert result["overall"] == 0.8
        assert result["dimensions"]["helpfulness"] == 0.9
        assert result["dimensions"]["correctness"] == 0.85
        assert result["latency_ms"] == 150.0
        assert "timestamp" in result


class TestConfidenceConfig:
    """Tests for ConfidenceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConfidenceConfig()

        assert config.reward_model == "nemotron-reward"
        assert config.enabled is True
        assert config.min_confidence == 0.5
        assert config.timeout == 30.0
        assert config.cache_scores is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConfidenceConfig(
            reward_model="custom-reward",
            enabled=False,
            min_confidence=0.7,
            timeout=10.0,
        )

        assert config.reward_model == "custom-reward"
        assert config.enabled is False
        assert config.min_confidence == 0.7


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    @pytest.fixture
    def mock_helix(self):
        """Create a mock Helix instance."""
        helix = MagicMock()
        response = MagicMock()
        response.content = """HELPFULNESS: 8
CORRECTNESS: 7
COHERENCE: 9
COMPLEXITY: 6
VERBOSITY: 7
OVERALL: 7.4"""
        helix.generate = AsyncMock(return_value=response)
        return helix

    @pytest.fixture
    def calibrator(self, mock_helix):
        """Create a calibrator with mock Helix."""
        return ConfidenceCalibrator(
            helix=mock_helix,
            config=ConfidenceConfig(),
        )

    @pytest.fixture
    def sample_hypothesis(self):
        """Create a sample hypothesis for testing."""
        return {
            "strategy_type": "trend_following",
            "time_horizon": "swing",
            "rationale": "Low volatility trending market favors momentum strategies",
            "entry_conditions": ["SMA crossover", "Volume confirmation"],
            "exit_conditions": ["Trailing stop", "Momentum divergence"],
            "expected_sharpe": 1.8,
            "expected_win_rate": 0.55,
        }

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_success(self, calibrator, mock_helix, sample_hypothesis):
        """Test successful hypothesis evaluation."""
        score = await calibrator.evaluate_hypothesis(
            sample_hypothesis,
            {"regime": "trending", "volatility": "low"},
        )

        assert score.overall == pytest.approx(0.74, abs=0.01)
        assert score.helpfulness == pytest.approx(0.8, abs=0.01)
        assert score.correctness == pytest.approx(0.7, abs=0.01)
        assert score.coherence == pytest.approx(0.9, abs=0.01)
        mock_helix.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_disabled(self, mock_helix, sample_hypothesis):
        """Test that disabled calibration returns default score."""
        calibrator = ConfidenceCalibrator(
            helix=mock_helix,
            config=ConfidenceConfig(enabled=False),
        )

        score = await calibrator.evaluate_hypothesis(sample_hypothesis)

        assert score.overall == 0.5
        mock_helix.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_timeout(self, mock_helix, sample_hypothesis):
        """Test timeout handling."""
        import asyncio

        mock_helix.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        calibrator = ConfidenceCalibrator(
            helix=mock_helix,
            config=ConfidenceConfig(timeout=1.0),
        )

        score = await calibrator.evaluate_hypothesis(sample_hypothesis)

        # Should return default score on timeout
        assert score.overall == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_error(self, mock_helix, sample_hypothesis):
        """Test error handling."""
        mock_helix.generate = AsyncMock(side_effect=Exception("API error"))

        calibrator = ConfidenceCalibrator(
            helix=mock_helix,
            config=ConfidenceConfig(),
        )

        score = await calibrator.evaluate_hypothesis(sample_hypothesis)

        # Should return default score on error
        assert score.overall == 0.5

    @pytest.mark.asyncio
    async def test_caching(self, calibrator, mock_helix, sample_hypothesis):
        """Test that scores are cached."""
        # First call
        score1 = await calibrator.evaluate_hypothesis(sample_hypothesis)

        # Second call with same hypothesis
        score2 = await calibrator.evaluate_hypothesis(sample_hypothesis)

        # Should only call Helix once
        assert mock_helix.generate.call_count == 1
        assert score1.overall == score2.overall

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_helix, sample_hypothesis):
        """Test that caching can be disabled."""
        calibrator = ConfidenceCalibrator(
            helix=mock_helix,
            config=ConfidenceConfig(cache_scores=False),
        )

        await calibrator.evaluate_hypothesis(sample_hypothesis)
        await calibrator.evaluate_hypothesis(sample_hypothesis)

        # Should call Helix twice
        assert mock_helix.generate.call_count == 2

    def test_clear_cache(self, calibrator):
        """Test cache clearing."""
        calibrator._cache["key1"] = ConfidenceScore(overall=0.8)
        calibrator._cache["key2"] = ConfidenceScore(overall=0.7)

        cleared = calibrator.clear_cache()

        assert cleared == 2
        assert calibrator.cache_size == 0

    def test_parse_scores_complete(self, calibrator):
        """Test parsing complete score response."""
        response = """Based on my evaluation:
HELPFULNESS: 9
CORRECTNESS: 8
COHERENCE: 8.5
COMPLEXITY: 7
VERBOSITY: 6
OVERALL: 7.7
"""
        score = calibrator._parse_scores(response, 100.0)

        assert score.overall == pytest.approx(0.77, abs=0.01)
        assert score.helpfulness == pytest.approx(0.9, abs=0.01)
        assert score.correctness == pytest.approx(0.8, abs=0.01)
        assert score.latency_ms == 100.0

    def test_parse_scores_partial(self, calibrator):
        """Test parsing partial score response."""
        response = """HELPFULNESS: 8
OVERALL: 7"""

        score = calibrator._parse_scores(response, 50.0)

        assert score.overall == pytest.approx(0.7, abs=0.01)
        assert score.helpfulness == pytest.approx(0.8, abs=0.01)
        # Defaults for missing
        assert score.correctness == 0.5
        assert score.coherence == 0.5

    def test_parse_scores_with_brackets(self, calibrator):
        """Test parsing scores with brackets."""
        response = """HELPFULNESS: [8]
CORRECTNESS: [7.5]
OVERALL: [7.8]"""

        score = calibrator._parse_scores(response, 75.0)

        assert score.overall == pytest.approx(0.78, abs=0.01)
        assert score.helpfulness == pytest.approx(0.8, abs=0.01)
        assert score.correctness == pytest.approx(0.75, abs=0.01)


class TestQualityDimension:
    """Tests for QualityDimension enum."""

    def test_dimension_values(self):
        """Test that all dimensions have correct values."""
        assert QualityDimension.HELPFULNESS.value == "helpfulness"
        assert QualityDimension.CORRECTNESS.value == "correctness"
        assert QualityDimension.COHERENCE.value == "coherence"
        assert QualityDimension.COMPLEXITY.value == "complexity"
        assert QualityDimension.VERBOSITY.value == "verbosity"
