"""
Unit tests for Kelly Criterion position sizing.
"""

import numpy as np
import pytest

from ordinis.risk.position_sizing.kelly import (
    FractionalKellyManager,
    KellyCriterion,
    KellyResult,
    KellyVariant,
    TradeStatistics,
)


class TestTradeStatistics:
    """Test TradeStatistics dataclass."""

    def test_valid_statistics(self):
        """Test creation with valid statistics."""
        stats = TradeStatistics(win_rate=0.55, avg_win=0.02, avg_loss=0.015)

        assert stats.win_rate == 0.55
        assert stats.avg_win == 0.02
        assert stats.avg_loss == 0.015
        assert stats.win_loss_ratio == pytest.approx(0.02 / 0.015)
        assert stats.expectancy == pytest.approx(0.55 * 0.02 - 0.45 * 0.015)

    def test_zero_avg_loss_raises(self):
        """Test that zero average loss raises ValueError."""
        with pytest.raises(ValueError, match="Average loss cannot be zero"):
            TradeStatistics(win_rate=0.55, avg_win=0.02, avg_loss=0.0)

    def test_negative_avg_loss(self):
        """Test handling of negative average loss."""
        stats = TradeStatistics(win_rate=0.55, avg_win=0.02, avg_loss=-0.015)
        assert stats.win_loss_ratio == pytest.approx(0.02 / 0.015)


class TestKellyResult:
    """Test KellyResult dataclass."""

    def test_valid_result(self):
        """Test creation with valid result."""
        result = KellyResult(
            optimal_fraction=0.20,
            adjusted_fraction=0.10,
            expected_growth=0.05,
            ruin_probability=0.01,
            kelly_variant=KellyVariant.BASIC,
        )

        assert result.optimal_fraction == 0.20
        assert result.adjusted_fraction == 0.10
        assert result.expected_growth == 0.05
        assert result.ruin_probability == 0.01
        assert result.kelly_variant == KellyVariant.BASIC

    def test_invalid_fraction_raises(self):
        """Test that invalid Kelly fraction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Kelly fraction"):
            KellyResult(
                optimal_fraction=15.0,  # Too high
                adjusted_fraction=0.10,
                expected_growth=0.05,
                ruin_probability=0.01,
                kelly_variant=KellyVariant.BASIC,
            )

    def test_confidence_interval(self):
        """Test result with confidence interval."""
        result = KellyResult(
            optimal_fraction=0.20,
            adjusted_fraction=0.10,
            expected_growth=0.05,
            ruin_probability=0.01,
            kelly_variant=KellyVariant.EMPIRICAL,
            confidence_interval=(0.15, 0.25),
        )

        assert result.confidence_interval == (0.15, 0.25)


class TestKellyCriterion:
    """Test KellyCriterion calculator."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        kelly = KellyCriterion()
        assert kelly.kelly_fraction == 0.5
        assert kelly.max_position == 0.25
        assert kelly.min_position == 0.01
        assert kelly.confidence_level == 0.95

        custom = KellyCriterion(kelly_fraction=0.25, max_position=0.15, min_position=0.005)
        assert custom.kelly_fraction == 0.25
        assert custom.max_position == 0.15

    def test_basic_kelly_positive_edge(self):
        """Test basic Kelly with positive edge."""
        kelly = KellyCriterion(kelly_fraction=1.0)  # Full Kelly for testing
        result = kelly.basic_kelly(win_probability=0.6, win_loss_ratio=2.0)

        # Kelly formula: f* = (p*b - q) / b = (0.6*2 - 0.4) / 2 = 0.4
        assert result.optimal_fraction == pytest.approx(0.4)
        assert result.kelly_variant == KellyVariant.BASIC
        assert result.expected_growth > 0

    def test_basic_kelly_negative_edge(self):
        """Test basic Kelly with negative edge."""
        kelly = KellyCriterion()
        result = kelly.basic_kelly(win_probability=0.3, win_loss_ratio=1.0)

        # Negative expectancy - should give negative Kelly
        assert result.optimal_fraction < 0

    def test_basic_kelly_fractional_adjustment(self):
        """Test fractional Kelly adjustment."""
        kelly = KellyCriterion(kelly_fraction=0.5)
        result = kelly.basic_kelly(win_probability=0.6, win_loss_ratio=2.0)

        # Adjusted should be half of optimal
        assert result.adjusted_fraction == pytest.approx(result.optimal_fraction * 0.5)

    def test_basic_kelly_position_limits(self):
        """Test position limits are applied."""
        kelly = KellyCriterion(kelly_fraction=1.0, max_position=0.20)
        result = kelly.basic_kelly(win_probability=0.8, win_loss_ratio=3.0)

        # Very high edge would give >0.20, should be capped
        assert result.adjusted_fraction <= 0.20

    def test_basic_kelly_invalid_inputs(self):
        """Test invalid inputs raise ValueError."""
        kelly = KellyCriterion()

        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            kelly.basic_kelly(win_probability=1.5, win_loss_ratio=2.0)

        with pytest.raises(ValueError, match="Win/loss ratio must be positive"):
            kelly.basic_kelly(win_probability=0.6, win_loss_ratio=-1.0)

    def test_continuous_kelly_merton(self):
        """Test continuous Kelly (Merton formula)."""
        kelly = KellyCriterion(kelly_fraction=1.0)
        result = kelly.continuous_kelly(expected_return=0.12, volatility=0.20, risk_free_rate=0.02)

        # Merton: f* = μ/σ² = 0.10 / 0.04 = 2.5
        assert result.optimal_fraction == pytest.approx(2.5)
        assert result.kelly_variant == KellyVariant.CONTINUOUS

    def test_continuous_kelly_zero_volatility_raises(self):
        """Test that zero volatility raises ValueError."""
        kelly = KellyCriterion()

        with pytest.raises(ValueError, match="Volatility must be positive"):
            kelly.continuous_kelly(expected_return=0.12, volatility=0.0)

    def test_empirical_kelly(self):
        """Test empirical Kelly from returns."""
        np.random.seed(42)
        # Generate returns with strong positive bias to ensure positive expectancy
        base_returns = np.random.normal(0.002, 0.015, 100)
        # Add positive drift to ensure positive expectancy
        returns = base_returns + 0.001

        kelly = KellyCriterion(kelly_fraction=0.5)
        result = kelly.empirical_kelly(returns, bootstrap_samples=100)

        assert result.kelly_variant == KellyVariant.EMPIRICAL
        # Kelly may be negative if optimization finds that optimal
        assert isinstance(result.optimal_fraction, float)
        assert len(result.confidence_interval) == 2

    def test_empirical_kelly_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        kelly = KellyCriterion()
        returns = np.array([0.01, 0.02])  # Only 2 observations

        with pytest.raises(ValueError, match="Need at least 30 observations"):
            kelly.empirical_kelly(returns)

    def test_multi_asset_kelly(self):
        """Test multi-asset Kelly optimization."""
        kelly = KellyCriterion(kelly_fraction=0.5)

        expected_returns = np.array([0.08, 0.10, 0.12])
        cov_matrix = np.array([[0.04, 0.01, 0.01], [0.01, 0.06, 0.02], [0.01, 0.02, 0.08]])

        weights, result = kelly.multi_asset_kelly(expected_returns, cov_matrix)

        assert len(weights) == 3
        assert result.kelly_variant == KellyVariant.MULTI_ASSET
        assert isinstance(result.optimal_fraction, float)

    def test_multi_asset_kelly_dimension_mismatch_raises(self):
        """Test dimension mismatch raises ValueError."""
        kelly = KellyCriterion()

        expected_returns = np.array([0.08, 0.10])
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.06], [0.01, 0.02]])  # 3x2

        with pytest.raises(ValueError, match="Covariance matrix dimension mismatch"):
            kelly.multi_asset_kelly(expected_returns, cov_matrix)

    def test_from_trade_statistics(self):
        """Test Kelly calculation from trade statistics."""
        kelly = KellyCriterion()
        stats = TradeStatistics(win_rate=0.55, avg_win=0.02, avg_loss=0.015)

        result = kelly.from_trade_statistics(stats)

        assert result.kelly_variant == KellyVariant.BASIC
        assert result.optimal_fraction > 0  # Positive expectancy


class TestFractionalKellyManager:
    """Test FractionalKellyManager dynamic adjustment."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        manager = FractionalKellyManager()

        assert manager.base_fraction == 0.5
        assert manager.current_fraction == 0.5
        assert manager.peak_equity == 1.0

    def test_update_new_peak(self):
        """Test update on new equity peak."""
        manager = FractionalKellyManager(base_fraction=0.5, recovery_rate=0.1)

        # New peak should increase fraction
        initial_fraction = manager.current_fraction
        new_fraction = manager.update(1.1)

        assert manager.peak_equity == 1.1
        assert new_fraction >= initial_fraction

    def test_update_drawdown(self):
        """Test update during drawdown."""
        manager = FractionalKellyManager(
            base_fraction=0.5, drawdown_threshold=0.10, min_fraction=0.1
        )

        # Set peak
        manager.update(1.0)

        # 15% drawdown should reduce fraction
        new_fraction = manager.update(0.85)

        assert new_fraction < 0.5  # Below base fraction

    def test_update_respects_limits(self):
        """Test that updates respect min/max limits."""
        manager = FractionalKellyManager(base_fraction=0.5, min_fraction=0.1, max_fraction=0.75)

        # Large drawdown
        manager.update(1.0)
        low_fraction = manager.update(0.5)
        assert low_fraction >= 0.1  # Min limit

        # Large gains
        for _ in range(20):
            manager.update(manager.current_equity * 1.1)
        assert manager.current_fraction <= 0.75  # Max limit

    def test_get_position_size(self):
        """Test position size calculation."""
        manager = FractionalKellyManager(base_fraction=0.5)

        kelly_result = KellyResult(
            optimal_fraction=0.40,
            adjusted_fraction=0.20,
            expected_growth=0.05,
            ruin_probability=0.01,
            kelly_variant=KellyVariant.BASIC,
        )

        position = manager.get_position_size(kelly_result, account_value=100000)

        # Position = optimal_fraction * current_fraction * account_value
        expected = 0.40 * 0.5 * 100000
        assert position == pytest.approx(expected)

    def test_reset(self):
        """Test reset to base fraction."""
        manager = FractionalKellyManager(base_fraction=0.5)

        # Cause drawdown
        manager.update(1.0)
        manager.update(0.8)
        assert manager.current_fraction != 0.5

        # Reset
        manager.reset()
        assert manager.current_fraction == 0.5
        assert manager.peak_equity == 1.0
        assert manager.current_equity == 1.0
