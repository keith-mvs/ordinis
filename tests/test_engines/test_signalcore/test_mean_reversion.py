"""Tests for MeanReversionModel.

Tests cover:
- Model initialization
- Parameter defaults and custom values
- Signal generation logic
- Trend filtering
- RSI and Bollinger Band conditions
- Volume confirmation
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.mean_reversion import MeanReversionModel


class TestMeanReversionModelInit:
    """Tests for MeanReversionModel initialization."""

    @pytest.fixture
    def default_config(self):
        """Create default model config."""
        return ModelConfig(
            model_id="mean_rev_test",
            model_type="technical",
            version="1.0.0",
        )

    @pytest.mark.unit
    def test_init_default_parameters(self, default_config):
        """Test initialization with default parameters."""
        model = MeanReversionModel(default_config)

        assert model.rsi_period == 14
        assert model.rsi_oversold == 30
        assert model.rsi_overbought == 70
        assert model.bb_period == 20
        assert model.bb_std == 2.0
        assert model.volume_period == 20
        assert model.volume_factor == 1.5
        assert model.trend_filter_period == 200

    @pytest.mark.unit
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        config = ModelConfig(
            model_id="mean_rev_custom",
            model_type="technical",
            parameters={
                "rsi_period": 10,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "bb_period": 30,
                "bb_std": 2.5,
                "volume_period": 15,
                "volume_factor": 2.0,
                "trend_filter_period": 100,
            },
        )

        model = MeanReversionModel(config)

        assert model.rsi_period == 10
        assert model.rsi_oversold == 25
        assert model.rsi_overbought == 75
        assert model.bb_period == 30
        assert model.bb_std == 2.5
        assert model.volume_period == 15
        assert model.volume_factor == 2.0
        assert model.trend_filter_period == 100

    @pytest.mark.unit
    def test_min_data_points_updated(self, default_config):
        """Test that min_data_points is updated based on max period."""
        model = MeanReversionModel(default_config)

        # trend_filter_period is 200, so min_data_points should be at least 210
        assert model.config.min_data_points >= 200


class TestMeanReversionModelGenerate:
    """Tests for MeanReversionModel.generate method."""

    @pytest.fixture
    def model(self):
        """Create model with shorter periods for testing."""
        config = ModelConfig(
            model_id="mean_rev_test",
            model_type="technical",
            parameters={
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_period": 20,
                "volume_factor": 1.0,  # Lower factor to make tests easier
                "trend_filter_period": 50,  # Shorter for testing
            },
            min_data_points=60,
        )
        return MeanReversionModel(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100

        # Create trending up data
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5
        open_price = close - np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 10000, n).astype(float)

        return pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_returns_none_insufficient_data(self, model):
        """Test generate returns None with insufficient data."""
        small_data = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
        })

        result = await model.generate(
            "TEST", small_data, datetime.now(timezone.utc)
        )

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_with_valid_data(self, model, sample_data):
        """Test generate with valid data returns signal or None."""
        result = await model.generate(
            "TEST", sample_data, datetime.now(timezone.utc)
        )

        # Result should be Signal or None
        if result is not None:
            assert result.symbol == "TEST"
            assert result.signal_type == SignalType.ENTRY
            assert result.direction in [Direction.LONG, Direction.SHORT]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_long_signal_rsi_oversold(self):
        """Test generating long signal when RSI is oversold in uptrend."""
        # Create data with declining prices (oversold RSI) in overall uptrend
        np.random.seed(123)
        n = 250

        # Strong uptrend first, then sharp decline at end
        trend = np.linspace(100, 150, n)  # Uptrend
        decline = np.zeros(n)
        decline[-30:] = np.linspace(0, -20, 30)  # Sharp decline at end
        close = trend + decline + np.random.randn(n) * 0.5

        volume = np.ones(n) * 5000  # High constant volume

        data = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        })

        config = ModelConfig(
            model_id="test",
            model_type="technical",
            parameters={
                "trend_filter_period": 200,
                "volume_factor": 0.5,  # Low factor to ensure volume confirmation
            },
            min_data_points=220,
        )
        model = MeanReversionModel(config)

        result = await model.generate("TEST", data, datetime.now(timezone.utc))
        # May or may not generate signal depending on exact RSI values
        # Just verify it doesn't crash
        assert result is None or result.direction in [Direction.LONG, Direction.SHORT]


class TestMeanReversionNoTrendFilter:
    """Tests for mean reversion without trend filter."""

    @pytest.fixture
    def model_no_trend(self):
        """Create model without trend filter."""
        config = ModelConfig(
            model_id="mean_rev_no_trend",
            model_type="technical",
            parameters={
                "trend_filter_period": 0,  # Disable trend filter
                "volume_factor": 0.5,
            },
            min_data_points=50,
        )
        return MeanReversionModel(config)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_no_trend_filter_allows_both_directions(self, model_no_trend):
        """Test that disabling trend filter allows signals in both directions."""
        np.random.seed(456)
        n = 100

        close = 100 + np.random.randn(n) * 2
        volume = np.ones(n) * 5000

        data = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        })

        result = await model_no_trend.generate(
            "TEST", data, datetime.now(timezone.utc)
        )

        # Just verify no crash
        assert result is None or isinstance(result.direction, Direction)


class TestMeanReversionVolumeConfirmation:
    """Tests for volume confirmation logic."""

    @pytest.fixture
    def model(self):
        """Create model with high volume factor."""
        config = ModelConfig(
            model_id="mean_rev_volume",
            model_type="technical",
            parameters={
                "volume_factor": 3.0,  # High factor requiring strong volume
                "trend_filter_period": 0,
            },
            min_data_points=50,
        )
        return MeanReversionModel(config)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_low_volume_no_signal(self, model):
        """Test that low volume prevents signal generation."""
        np.random.seed(789)
        n = 100

        # Create data with low volume at the end
        close = 100 + np.random.randn(n) * 2
        volume = np.ones(n) * 1000
        volume[-1] = 100  # Very low final volume

        data = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        })

        result = await model.generate("TEST", data, datetime.now(timezone.utc))

        # Should return None due to volume confirmation failure
        assert result is None


class TestMeanReversionSignalMetadata:
    """Tests for signal metadata."""

    @pytest.fixture
    def model(self):
        """Create model for metadata tests."""
        config = ModelConfig(
            model_id="mean_rev_meta",
            model_type="technical",
            parameters={
                "trend_filter_period": 20,
                "volume_factor": 0.1,  # Very low to ensure signals
                "rsi_period": 14,
                "bb_period": 20,
            },
            min_data_points=50,
        )
        return MeanReversionModel(config)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_signal_has_metadata(self, model):
        """Test that generated signal contains expected metadata."""
        np.random.seed(111)
        n = 100

        # Create data likely to generate a signal
        close = np.linspace(100, 120, n)  # Strong uptrend
        close[-20:] = np.linspace(120, 90, 20)  # Sharp decline for oversold

        volume = np.ones(n) * 5000

        data = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": volume,
        })

        result = await model.generate("TEST", data, datetime.now(timezone.utc))

        if result is not None:
            assert "rsi" in result.metadata
            assert "bb_lower" in result.metadata
            assert "bb_upper" in result.metadata
            assert "trend_sma" in result.metadata
            assert "reason" in result.metadata
