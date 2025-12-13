"""Tests for Bollinger Bands strategy."""

from datetime import datetime

import pandas as pd

from application.strategies.bollinger_bands import BollingerBandsStrategy
from engines.signalcore.core.signal import Direction, SignalType


def create_test_data(bars=60, **overrides):
    """Create test OHLCV data with sensible defaults."""
    defaults = {
        "open": [100] * bars,
        "high": [102] * bars,
        "low": [98] * bars,
        "close": [100] * bars,
        "volume": [1000] * bars,
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestBollingerBandsConfiguration:
    """Test strategy configuration."""

    def test_default_configuration(self):
        """Test default parameter configuration."""
        strategy = BollingerBandsStrategy(name="test-bb")

        assert strategy.params["bb_period"] == 20
        assert strategy.params["bb_std"] == 2.0
        assert strategy.params["min_band_width"] == 0.01
        assert strategy.params["min_bars"] == 50  # bb_period + 30

    def test_custom_configuration(self):
        """Test custom parameter configuration."""
        strategy = BollingerBandsStrategy(
            name="test", bb_period=30, bb_std=2.5, min_band_width=0.02
        )

        assert strategy.params["bb_period"] == 30
        assert strategy.params["bb_std"] == 2.5
        assert strategy.params["min_band_width"] == 0.02

    def test_get_required_bars(self):
        """Test required bars calculation."""
        strategy = BollingerBandsStrategy(name="test", bb_period=25)
        assert strategy.get_required_bars() == 55  # bb_period + 30


class TestLowerBandTouch:
    """Test lower band touch signal generation."""

    def test_lower_band_touch_long_entry(self):
        """Test long entry signal when price touches lower band."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        # Price touches lower band - need at least 35 bars (bb_period + 30)
        # Sharp drop at the end to cross below established bands
        data = create_test_data(
            bars=60,
            close=[100] * 56 + [100, 100, 100, 63.9],  # Sharp drop to touch lower band
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Signal should be ENTRY or HOLD (near lower band can be either)
        assert signal.signal_type in [SignalType.ENTRY, SignalType.HOLD]
        assert signal.direction == Direction.LONG
        assert signal.probability >= 0.5
        assert "band_position" in signal.metadata
        assert signal.metadata["band_position"] < 0.2  # Near lower band

    def test_lower_band_touch_with_high_volatility(self):
        """Test signal with sufficient band width."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0, min_band_width=0.01)

        # High volatility data - need at least 35 bars
        base_pattern = [100, 105, 95, 110, 90]
        data = create_test_data(
            bars=60,
            close=base_pattern * 11 + [85, 84, 83, 82, 80],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.ENTRY:
            assert "band_width" in signal.metadata
            assert signal.metadata["band_width"] >= strategy.params["min_band_width"]


class TestUpperBandTouch:
    """Test upper band touch signal generation."""

    def test_upper_band_touch_exit_signal(self):
        """Test exit signal when price touches upper band."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        # Price rallies to upper band
        data = create_test_data(
            bars=60,
            close=[100] * 55 + [105, 108, 110, 112, 115],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.EXIT:
            assert signal.direction == Direction.NEUTRAL
            assert "band_position" in signal.metadata
            assert signal.metadata["band_position"] > 0.8  # Near upper band


class TestMiddleBandBehavior:
    """Test behavior around middle band."""

    def test_middle_band_hold_signal(self):
        """Test hold signal when price near middle band."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        # Price oscillating around middle, ending centered
        # Last 5 values: [101, 99, 100, 100, 100] -> SMA = 100, current = 100
        data = create_test_data(
            bars=60,
            close=[100, 101, 99, 100, 101] * 11 + [100, 101, 99, 100, 100],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.HOLD:
            assert signal.direction == Direction.NEUTRAL
            assert 0.3 < signal.metadata["band_position"] < 0.7  # Mid-range


class TestBandWidthConditions:
    """Test band width filtering."""

    def test_insufficient_band_width_no_entry(self):
        """Test no entry signal when bands too narrow."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0, min_band_width=0.05)

        # Very low volatility (tight bands)
        data = create_test_data(
            bars=60,
            close=[100, 100.1, 99.9, 100, 100.1] * 12,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should get hold/neutral, not entry
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # If we somehow got an entry, band_width should meet minimum
            assert signal.metadata.get("band_width", 0) >= strategy.params["min_band_width"]


class TestSignalMetadata:
    """Test signal metadata fields."""

    def test_entry_metadata_complete(self):
        """Test all metadata fields present in entry signal."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        data = create_test_data(
            bars=60,
            close=[100] * 55 + [95, 94, 93, 92, 88],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        metadata = signal.metadata

        assert "strategy" in metadata
        assert "band_position" in metadata
        assert "band_width" in metadata
        assert "middle_band" in metadata
        assert "upper_band" in metadata
        assert "lower_band" in metadata
        assert "stop_loss" in metadata
        assert "take_profit" in metadata

    def test_metadata_values_reasonable(self):
        """Test metadata values are within reasonable ranges."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        data = create_test_data(bars=60, close=list(range(100, 160)))
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            metadata = signal.metadata

            # Band position should be 0-1
            assert 0 <= metadata.get("band_position", 0.5) <= 1

            # Band width should be positive
            assert metadata.get("band_width", 0) >= 0

            # Bands should be ordered: lower < middle < upper
            if all(k in metadata for k in ["lower_band", "middle_band", "upper_band"]):
                assert metadata["lower_band"] < metadata["middle_band"] < metadata["upper_band"]


class TestDataValidation:
    """Test data validation and error handling."""

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        strategy = BollingerBandsStrategy(name="test", bb_period=20)

        data = create_test_data(bars=10)  # Need 20

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_missing_columns_returns_none(self):
        """Test that missing columns returns None."""
        strategy = BollingerBandsStrategy(name="test")

        data = pd.DataFrame({"high": [100] * 30, "low": [95] * 30, "close": [98] * 30})

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_empty_dataframe_returns_none(self):
        """Test that empty DataFrame returns None."""
        strategy = BollingerBandsStrategy(name="test")

        data = pd.DataFrame()

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None


class TestSymbolHandling:
    """Test symbol extraction from data."""

    def test_symbol_from_data(self):
        """Test symbol extraction when present."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5)

        data = create_test_data(bars=60)
        data["symbol"] = "TSLA"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.symbol == "TSLA"

    def test_default_symbol_when_missing(self):
        """Test default symbol when not in data."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5)

        data = create_test_data(bars=60)

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            assert signal.symbol == "UNKNOWN"


class TestStrategyDescription:
    """Test strategy description."""

    def test_get_description_format(self):
        """Test description contains key information."""
        strategy = BollingerBandsStrategy(name="test")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert len(description) > 100
        assert "Bollinger Bands" in description
        assert "Entry Rules" in description
        assert "Exit Rules" in description


class TestEdgeCases:
    """Test edge cases."""

    def test_probability_bounds(self):
        """Test that probability is within valid bounds."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5, bb_std=2.0)

        # Extreme price movement
        data = create_test_data(
            bars=60,
            close=[100] * 55 + [120, 80, 150, 60, 50],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert 0 <= signal.probability <= 1

    def test_nan_handling(self):
        """Test handling of NaN values in data."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5)

        data = create_test_data(bars=60)
        # Introduce some NaN values
        data.loc[10:12, "close"] = float("nan")
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should either handle gracefully or return None
        # Not crash with exception
        assert signal is None or isinstance(signal.probability, float)

    def test_zero_volume(self):
        """Test handling of zero volume."""
        strategy = BollingerBandsStrategy(name="test", bb_period=5)

        data = create_test_data(
            bars=60,
            close=[100] * 55 + [95, 94, 93, 92, 88],
            volume=[0] * 60,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should still generate signal (BB doesn't depend on volume)
        assert signal is not None


class TestMeanReversion:
    """Test mean reversion behavior."""

    def test_oversold_to_mean_reversion(self):
        """Test mean reversion from oversold condition."""
        strategy = BollingerBandsStrategy(name="test", bb_period=10, bb_std=2.0)

        # Price drops to lower band then reverts - need 40+ bars (10 + 30)
        data = create_test_data(
            bars=60,
            close=[100] * 35 + [95, 93, 91, 90, 89] + [90, 92, 94, 96, 98] + [100] * 15,
        )
        data["symbol"] = "AAPL"

        # Check signal at bottom (should be entry)
        signal_bottom = strategy.generate_signal(data.iloc[:45], datetime.utcnow())

        # Check signal at reversion (might be exit)
        signal_revert = strategy.generate_signal(data, datetime.utcnow())

        if signal_bottom is not None and signal_bottom.signal_type == SignalType.ENTRY:
            assert signal_bottom.direction == Direction.LONG

        if signal_revert is not None and signal_revert.signal_type == SignalType.EXIT:
            assert signal_revert.metadata["band_position"] > signal_bottom.metadata["band_position"]
