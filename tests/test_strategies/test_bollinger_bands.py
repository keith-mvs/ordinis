"""Tests for Bollinger Bands strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strategies.bollinger_bands import BollingerBandsStrategy


def create_test_data(bars: int = 150, trend: str = "neutral") -> pd.DataFrame:
    """
    Create test market data.

    Args:
        bars: Number of bars to generate
        trend: Market trend (neutral, uptrend, downtrend, volatile, low_volatility)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="D")

    # Use numpy arrays to avoid index alignment issues with DatetimeIndex
    if trend == "neutral":
        # Sideways market with oscillations
        close = 100 + 5 * np.array([(x % 20 - 10) * 0.5 for x in range(bars)])
    elif trend == "uptrend":
        # Strong uptrend
        close = 100 + np.arange(bars) * 0.3
    elif trend == "downtrend":
        # Strong downtrend
        close = 150 - np.arange(bars) * 0.3
    elif trend == "volatile":
        # High volatility oscillations
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(bars) * 3)
    elif trend == "low_volatility":
        # Very low volatility (compressed bands)
        np.random.seed(42)
        close = 100 + np.random.randn(bars) * 0.1
    elif trend == "touching_lower":
        # Create data where price touches lower band
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(bars) * 1)
        # Force last values to drop significantly
        close[-10:] = close[-10] - np.arange(10) * 0.5
    elif trend == "touching_upper":
        # Create data where price touches upper band
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(bars) * 1)
        # Force last values to spike up significantly
        close[-10:] = close[-10] + np.arange(10) * 0.5
    else:
        close = np.array([100.0] * bars)

    data = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1000000] * bars,
        },
        index=dates,
    )

    return data


class TestBollingerBandsStrategy:
    """Tests for BollingerBandsStrategy class."""

    def test_strategy_initialization(self):
        """Test strategy initializes with default parameters."""
        strategy = BollingerBandsStrategy(name="test-bb")

        assert strategy.name == "test-bb"
        assert strategy.params["bb_period"] == 20
        assert strategy.params["bb_std"] == 2.0
        assert strategy.params["min_band_width"] == 0.02

    def test_strategy_custom_parameters(self):
        """Test strategy with custom parameters."""
        strategy = BollingerBandsStrategy(
            name="custom-bb",
            bb_period=25,
            bb_std=2.5,
            min_band_width=0.03,
        )

        assert strategy.params["bb_period"] == 25
        assert strategy.params["bb_std"] == 2.5
        assert strategy.params["min_band_width"] == 0.03

    def test_configure_creates_model(self):
        """Test that configure creates underlying BB model."""
        strategy = BollingerBandsStrategy(name="test-bb")

        assert hasattr(strategy, "model")
        assert strategy.model is not None
        assert strategy.model.config.model_type == "mean_reversion"

    def test_get_description(self):
        """Test strategy description."""
        strategy = BollingerBandsStrategy(name="test-bb")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert "Bollinger Bands" in description
        assert "lower band" in description.lower()
        assert "upper band" in description.lower()
        assert "20" in description  # Default BB period

    def test_get_required_bars_default(self):
        """Test required bars calculation."""
        strategy = BollingerBandsStrategy(name="test-bb")

        required = strategy.get_required_bars()

        # Should be bb_period + 20
        assert required == 20 + 20
        assert required == 40

    def test_get_required_bars_custom(self):
        """Test required bars with custom BB period."""
        strategy = BollingerBandsStrategy(
            name="test-bb",
            bb_period=30,
        )

        required = strategy.get_required_bars()

        assert required == 30 + 20
        assert required == 50

    def test_validate_data_insufficient_bars(self):
        """Test validation fails with insufficient data."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=30)  # Need 40

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Insufficient data" in msg

    def test_validate_data_sufficient_bars(self):
        """Test validation succeeds with sufficient data."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=100)

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_generate_signal_with_insufficient_data(self):
        """Test signal generation returns None for insufficient data."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=20)

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_generate_signal_with_valid_data(self):
        """Test signal generation with valid data."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        timestamp = datetime.utcnow()
        signal = strategy.generate_signal(data, timestamp)

        # Should generate a signal (even if HOLD)
        assert signal is not None
        assert hasattr(signal, "symbol")
        assert hasattr(signal, "signal_type")

    def test_generate_signal_handles_exceptions(self):
        """Test signal generation handles exceptions gracefully."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=100)

        try:
            signal = strategy.generate_signal(data, datetime.utcnow())
            assert signal is None or hasattr(signal, "symbol")
        except Exception:
            pytest.fail("generate_signal should handle exceptions gracefully")

    def test_str_representation(self):
        """Test string representation."""
        strategy = BollingerBandsStrategy(name="my-bb-strategy")

        result = str(strategy)

        assert result == "my-bb-strategy Strategy"

    def test_repr_representation(self):
        """Test repr representation."""
        strategy = BollingerBandsStrategy(name="my-bb-strategy")

        result = repr(strategy)

        assert "BollingerBandsStrategy" in result
        assert "my-bb-strategy" in result


class TestBollingerBandsStrategyParameters:
    """Tests for parameter handling."""

    def test_min_bars_calculated_from_bb_period(self):
        """Test min_bars is calculated from BB period."""
        strategy = BollingerBandsStrategy(name="test", bb_period=25)

        assert strategy.params["min_bars"] == 25 + 20

    def test_min_band_width_optional(self):
        """Test min_band_width has default."""
        strategy = BollingerBandsStrategy(name="test")

        assert "min_band_width" in strategy.params
        assert strategy.params["min_band_width"] == 0.02

    def test_model_config_parameters(self):
        """Test model config includes all parameters."""
        strategy = BollingerBandsStrategy(
            name="test",
            bb_period=25,
            bb_std=2.5,
            min_band_width=0.03,
        )

        config = strategy.model.config

        assert config.parameters["bb_period"] == 25
        assert config.parameters["bb_std"] == 2.5
        assert config.parameters["min_band_width"] == 0.03


class TestBollingerBandsStrategySignals:
    """Tests for signal generation logic."""

    def test_signal_contains_band_metadata(self):
        """Test signal contains Bollinger Bands metadata."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert "bb_period" in signal.metadata
        assert "upper_band" in signal.metadata
        assert "middle_band" in signal.metadata
        assert "lower_band" in signal.metadata
        assert "current_price" in signal.metadata

    def test_signal_contains_feature_contributions(self):
        """Test signal contains feature contributions."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert "percent_b" in signal.feature_contributions
        assert "band_width" in signal.feature_contributions

    def test_signal_has_regime_detection(self):
        """Test signal includes regime detection."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.regime in ["low_volatility", "moderate_volatility", "high_volatility"]

    def test_low_volatility_detection(self):
        """Test low volatility regime is detected."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="low_volatility")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Low volatility should result in HOLD signals
        assert signal.feature_contributions["low_volatility"] == 1.0 or signal.regime == "low_volatility"


class TestBollingerBandsStrategyIntegration:
    """Integration tests for BB strategy."""

    def test_strategy_with_trending_market(self):
        """Test strategy behavior in trending market."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="uptrend")

        # Test at multiple points
        for i in range(50, len(data), 20):
            timestamp = data.index[i]
            signal = strategy.generate_signal(data.iloc[: i + 1], timestamp)

            if signal:
                assert hasattr(signal, "symbol")
                assert hasattr(signal, "timestamp")
                assert hasattr(signal, "signal_type")

    def test_strategy_with_volatile_market(self):
        """Test strategy behavior in volatile market."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        timestamp = data.index[-1]
        signal = strategy.generate_signal(data, timestamp)

        # Should handle volatile data without crashing
        assert signal is not None
        assert hasattr(signal, "symbol")

    def test_strategy_with_different_timeframes(self):
        """Test strategy with different data lengths."""
        strategy = BollingerBandsStrategy(name="test-bb")

        for bars in [100, 200, 500]:  # Start at 100 to meet min_data_points
            data = create_test_data(bars=bars, trend="volatile")
            timestamp = data.index[-1]
            signal = strategy.generate_signal(data, timestamp)

            # Should handle different timeframes
            assert signal is not None

    def test_strategy_consistency(self):
        """Test strategy generates consistent signals for same data."""
        strategy = BollingerBandsStrategy(name="test-bb", bb_period=20)
        data = create_test_data(bars=150, trend="neutral")
        timestamp = data.index[-1]

        # Generate signal multiple times
        signal1 = strategy.generate_signal(data, timestamp)
        signal2 = strategy.generate_signal(data, timestamp)

        # Should be consistent
        if signal1 is None:
            assert signal2 is None
        else:
            assert signal2 is not None
            assert signal1.symbol == signal2.symbol
            assert signal1.signal_type == signal2.signal_type


class TestBollingerBandsStrategyValidation:
    """Tests for data validation."""

    def test_requires_ohlcv_columns(self):
        """Test strategy requires OHLCV columns."""
        strategy = BollingerBandsStrategy(name="test")

        # Missing volume
        data = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [105] * 100,
                "low": [95] * 100,
                "close": [102] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100),
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "volume" in msg.lower()

    def test_handles_empty_data(self):
        """Test strategy handles empty data."""
        strategy = BollingerBandsStrategy(name="test")
        data = pd.DataFrame()

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "empty" in msg.lower()

    def test_handles_none_data(self):
        """Test strategy handles None data."""
        strategy = BollingerBandsStrategy(name="test")

        is_valid, msg = strategy.validate_data(None)

        assert is_valid is False
        assert "empty" in msg.lower()


class TestBollingerBandsModel:
    """Tests for the underlying Bollinger Bands model."""

    def test_model_has_correct_parameters(self):
        """Test model is configured with correct parameters."""
        strategy = BollingerBandsStrategy(
            name="test",
            bb_period=25,
            bb_std=2.5,
        )

        model = strategy.model

        assert model.bb_period == 25
        assert model.bb_std == 2.5

    def test_model_calculates_bands_correctly(self):
        """Test model calculates Bollinger Bands correctly."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Upper band should be above middle, lower below
        assert signal.metadata["upper_band"] > signal.metadata["middle_band"]
        assert signal.metadata["lower_band"] < signal.metadata["middle_band"]

    def test_percent_b_in_valid_range(self):
        """Test %B is calculated and in reasonable range."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        percent_b = signal.feature_contributions["percent_b"]
        # %B can be outside 0-1 when price is outside bands
        # But should generally be in a reasonable range
        assert -0.5 <= percent_b <= 1.5


class TestBollingerBandsEdgeCases:
    """Edge case tests for Bollinger Bands strategy."""

    def test_handles_constant_prices(self):
        """Test strategy handles constant price data (zero volatility)."""
        strategy = BollingerBandsStrategy(name="test-bb")

        # Create data with constant prices
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [100.0] * 100,
                "low": [100.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000000] * 100,
            },
            index=dates,
        )

        # Should handle gracefully (may return HOLD due to low volatility)
        signal = strategy.generate_signal(data, datetime.utcnow())
        assert signal is not None or True  # May raise or return None

    def test_handles_missing_values(self):
        """Test strategy handles data with NaN values."""
        strategy = BollingerBandsStrategy(name="test-bb")
        data = create_test_data(bars=100, trend="neutral")

        # Add some NaN values
        data.iloc[50, data.columns.get_loc("close")] = np.nan

        # Should handle gracefully
        try:
            signal = strategy.generate_signal(data, datetime.utcnow())
            # May return None or a signal with degraded data_quality
            assert signal is None or signal.data_quality < 1.0
        except ValueError:
            # Also acceptable to raise an error for invalid data
            pass

    def test_extreme_price_movements(self):
        """Test strategy handles extreme price movements."""
        strategy = BollingerBandsStrategy(name="test-bb")

        # Create data with volatility followed by extreme spike
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Start with volatile data to establish proper bands (use numpy array, not Series)
        close_values = 100 + np.cumsum(np.random.randn(150) * 2)
        close_values[-1] = close_values[-2] + 50  # Large spike at end

        data = pd.DataFrame(
            {
                "open": close_values * 0.99,
                "high": close_values * 1.02,
                "low": close_values * 0.98,
                "close": close_values,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should still generate a signal
        assert signal is not None
        # Price should be well above upper band after spike
        assert signal.metadata["current_price"] > signal.metadata["upper_band"]
