"""Tests for Advanced Technical Models.

Tests cover:
- IchimokuValues dataclass
- IchimokuModel
- ChartPatternModel
- VolumeProfileModel
- OptionsSignalsModel
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.advanced_technical import (
    ChartPatternModel,
    IchimokuModel,
    IchimokuValues,
    OptionsSignalsModel,
    VolumeProfileModel,
)


class TestIchimokuValues:
    """Tests for IchimokuValues dataclass."""

    @pytest.mark.unit
    def test_create_ichimoku_values(self):
        """Test creating IchimokuValues."""
        values = IchimokuValues(
            tenkan=100.0,
            kijun=98.0,
            senkou_a=99.0,
            senkou_b=97.0,
            chikou=101.0,
        )

        assert values.tenkan == 100.0
        assert values.kijun == 98.0
        assert values.senkou_a == 99.0
        assert values.senkou_b == 97.0
        assert values.chikou == 101.0


class TestIchimokuModel:
    """Tests for IchimokuModel."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for Ichimoku testing."""
        np.random.seed(42)
        n = 60  # Need at least 52 bars for Ichimoku
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create trending up data
        base_price = 100
        trend = np.linspace(0, 20, n)
        noise = np.random.normal(0, 1, n)
        close = base_price + trend + noise

        return pd.DataFrame(
            {
                "open": close - np.random.uniform(0, 1, n),
                "high": close + np.random.uniform(0, 2, n),
                "low": close - np.random.uniform(0, 2, n),
                "close": close,
                "volume": np.random.uniform(1000, 5000, n),
            },
            index=dates,
        )

    @pytest.mark.unit
    def test_init_default_config(self):
        """Test initialization with default config."""
        model = IchimokuModel()

        assert model.config.model_id == "ichimoku_cloud"
        assert model.config.parameters["tenkan_period"] == 9
        assert model.config.parameters["kijun_period"] == 26
        assert model.config.parameters["senkou_b_period"] == 52

    @pytest.mark.unit
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ModelConfig(
            model_id="custom_ichimoku",
            model_type="technical",
            version="2.0.0",
            parameters={"tenkan_period": 7, "kijun_period": 22, "senkou_b_period": 44},
        )
        model = IchimokuModel(config=config)

        assert model.config.model_id == "custom_ichimoku"
        assert model.config.parameters["tenkan_period"] == 7

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_bullish_signal(self, sample_data):
        """Test generating bullish signal when price above cloud."""
        model = IchimokuModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(sample_data, timestamp)

        assert signal is not None
        assert signal.model_id == "ichimoku_cloud"
        assert "tenkan" in signal.metadata
        assert "kijun" in signal.metadata
        assert "senkou_a" in signal.metadata
        assert "senkou_b" in signal.metadata
        assert "cloud_thickness" in signal.metadata

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_with_downtrend_data(self):
        """Test generating bearish signal with downtrend data."""
        np.random.seed(42)
        n = 60
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create trending down data
        base_price = 120
        trend = np.linspace(0, -25, n)
        close = base_price + trend

        data = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        model = IchimokuModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(data, timestamp)

        # Should generate a bearish signal
        assert signal.score < 0


class TestChartPatternModel:
    """Tests for ChartPatternModel."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for pattern testing."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")
        close = 100 + np.random.normal(0, 2, n).cumsum()

        return pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.uniform(1000, 5000, n),
            },
            index=dates,
        )

    @pytest.mark.unit
    def test_init_default_config(self):
        """Test initialization with default config."""
        model = ChartPatternModel()

        assert model.config.model_id == "chart_patterns"
        assert model.config.parameters["min_bars_for_pattern"] == 20
        assert model.config.parameters["tolerance_pct"] == 0.02

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_signal(self, sample_data):
        """Test generating signal from pattern detection."""
        model = ChartPatternModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(sample_data, timestamp)

        assert signal is not None
        assert signal.model_id == "chart_patterns"
        assert "patterns" in signal.metadata

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_head_and_shoulders_pattern(self):
        """Test detection of head and shoulders pattern."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create head and shoulders pattern manually
        close = np.concatenate(
            [
                np.linspace(100, 105, 8),  # Left rise
                np.linspace(105, 100, 8),  # Left fall
                np.linspace(100, 110, 8),  # Head rise
                np.linspace(110, 100, 8),  # Head fall
                np.linspace(100, 105, 8),  # Right rise
                np.linspace(105, 100, 10),  # Right fall
            ]
        )

        data = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        model = ChartPatternModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(data, timestamp)

        assert signal is not None

    @pytest.mark.unit
    def test_find_peaks(self):
        """Test _find_peaks helper method."""
        # _find_peaks uses min_distance=2, so peaks must be at least 2 from edges
        arr = np.array([1, 3, 2, 5, 3, 4, 2, 6, 3])
        peaks = ChartPatternModel._find_peaks(arr)

        # With min_distance=2: index 3 (value 5) and index 5 are candidates
        # index 7 (value 6) is also a peak
        assert 3 in peaks  # value 5 at index 3
        # Note: index 1 and 7 are too close to edges with min_distance=2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_triangle_pattern(self):
        """Test detection of triangle pattern (converging highs/lows)."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create converging triangle
        close = 100 + np.random.normal(0, 0.5, n)
        # Converging highs (decreasing) and lows (increasing)
        high = close + np.linspace(5, 1, n)
        low = close - np.linspace(5, 1, n)

        data = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        model = ChartPatternModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(data, timestamp)

        # Should detect triangle
        assert signal is not None
        assert "triangle" in signal.metadata.get("patterns", [])


class TestVolumeProfileModel:
    """Tests for VolumeProfileModel."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for volume profile testing."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create price data with varying volume at different levels
        close = 100 + np.random.normal(0, 2, n).cumsum()

        return pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.uniform(1000, 5000, n),
            },
            index=dates,
        )

    @pytest.mark.unit
    def test_init_default_config(self):
        """Test initialization with default config."""
        model = VolumeProfileModel()

        assert model.config.model_id == "volume_profile"
        assert model.config.parameters["bins"] == 10
        assert model.config.min_data_points == 100

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_signal(self, sample_data):
        """Test generating signal from volume profile."""
        model = VolumeProfileModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(sample_data, timestamp)

        assert signal is not None
        assert signal.model_id == "volume_profile"
        assert "poc" in signal.metadata  # Point of Control
        assert "va_high" in signal.metadata  # Value Area High
        assert "va_low" in signal.metadata  # Value Area Low

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_bullish_when_below_val(self):
        """Test bullish signal when price below Value Area Low."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")

        # Create data where most volume is at higher prices
        # and current price is at the low end
        close = np.concatenate(
            [
                np.ones(80) * 110 + np.random.normal(0, 1, 80),  # High volume area
                np.linspace(110, 90, 20),  # Drop to lower price
            ]
        )

        data = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.concatenate(
                    [np.ones(80) * 5000, np.ones(20) * 1000]  # High vol at higher prices
                ),
            },
            index=dates,
        )

        model = VolumeProfileModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(data, timestamp)

        # Price below VAL should be bullish
        assert signal.score > 0


class TestOptionsSignalsModel:
    """Tests for OptionsSignalsModel."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for options testing."""
        np.random.seed(42)
        n = 30
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1d")
        close = 100 + np.random.normal(0, 1, n).cumsum()

        return pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.uniform(1000, 5000, n),
            },
            index=dates,
        )

    @pytest.mark.unit
    def test_init_default_config(self):
        """Test initialization with default config."""
        model = OptionsSignalsModel()

        assert model.config.model_id == "options_signals"
        assert model.config.model_type == "options"
        assert model.config.parameters["iv_percentile_threshold"] == 0.7
        assert model.config.parameters["oi_increase_threshold"] == 0.2

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_signal(self, sample_data):
        """Test generating signal from options data."""
        model = OptionsSignalsModel()
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(sample_data, timestamp)

        assert signal is not None
        assert signal.model_id == "options_signals"
        assert "iv_percentile" in signal.metadata
        assert "options_data_available" in signal.metadata

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_with_custom_config(self, sample_data):
        """Test generating signal with custom config."""
        config = ModelConfig(
            model_id="custom_options",
            model_type="options",
            version="2.0.0",
            parameters={
                "iv_percentile_threshold": 0.8,
                "oi_increase_threshold": 0.15,
            },
        )
        model = OptionsSignalsModel(config=config)
        timestamp = datetime.now(timezone.utc)

        signal = await model.generate(sample_data, timestamp)

        assert signal is not None
        assert signal.model_id == "custom_options"
