"""Tests for Momentum Breakout strategy."""

from datetime import datetime

import pandas as pd

from engines.signalcore.core.signal import Direction, SignalType
from strategies.momentum_breakout import MomentumBreakoutStrategy


def create_test_data(bars=21, **overrides):
    """Create test OHLCV data with sensible defaults."""
    defaults = {
        "open": [97] * bars,
        "high": [100] * bars,
        "low": [95] * bars,
        "close": [98] * bars,
        "volume": [1000] * bars,
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestMomentumBreakoutConfiguration:
    """Test strategy configuration."""

    def test_default_configuration(self):
        """Test default parameter configuration."""
        strategy = MomentumBreakoutStrategy(name="test-momentum")

        assert strategy.params["lookback_period"] == 20
        assert strategy.params["atr_period"] == 14
        assert strategy.params["volume_multiplier"] == 1.5
        assert strategy.params["breakout_threshold"] == 0.02
        assert strategy.params["min_bars"] == 20

    def test_custom_configuration(self):
        """Test custom parameter configuration."""
        strategy = MomentumBreakoutStrategy(
            name="test", lookback_period=30, atr_period=20, volume_multiplier=2.0
        )

        assert strategy.params["lookback_period"] == 30
        assert strategy.params["atr_period"] == 20
        assert strategy.params["volume_multiplier"] == 2.0

    def test_get_required_bars(self):
        """Test required bars calculation."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=25)
        assert strategy.get_required_bars() == 25


class TestATRCalculation:
    """Test Average True Range calculation."""

    def test_calculate_atr_basic(self):
        """Test basic ATR calculation."""
        strategy = MomentumBreakoutStrategy(name="test")

        data = create_test_data(
            bars=10,
            high=[102, 105, 108, 107, 110, 112, 115, 113, 116, 118],
            low=[98, 101, 104, 103, 106, 108, 111, 109, 112, 114],
            close=[100, 103, 106, 105, 108, 110, 113, 111, 114, 116],
        )

        atr = strategy._calculate_atr(data, period=3)

        assert not pd.isna(atr.iloc[-1])
        assert atr.iloc[-1] > 0


class TestUpsideBreakout:
    """Test upside breakout signal generation."""

    def test_upside_breakout_with_volume_confirmation(self):
        """Test upside breakout signal with volume confirmation."""
        strategy = MomentumBreakoutStrategy(
            name="test", lookback_period=5, atr_period=3, breakout_threshold=0.02
        )

        # Upside breakout: close > high * 1.02, volume surge
        data = create_test_data(
            bars=21,
            open=[97] * 20 + [100],
            high=[100] * 20 + [103],  # Rolling max = 103
            low=[95] * 20 + [97],
            close=[98] * 20 + [106],  # 106 > 103 * 1.02 = 105.06
            volume=[1000] * 20 + [2000],  # 2x avg
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.signal_type == SignalType.ENTRY
        assert signal.direction == Direction.LONG
        assert signal.probability > 0.55
        assert signal.expected_return > 0
        assert "breakout_type" in signal.metadata
        assert signal.metadata["breakout_type"] == "upside"

    def test_upside_breakout_without_volume_no_signal(self):
        """Test upside breakout without volume confirmation."""
        strategy = MomentumBreakoutStrategy(
            name="test", lookback_period=5, atr_period=3, volume_multiplier=2.0
        )

        # Breakout but no volume surge
        data = create_test_data(
            bars=21,
            high=[100] * 20 + [103],
            close=[98] * 20 + [106],
            volume=[1000] * 21,  # No surge
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should not get long breakout signal without volume
        if signal is not None:
            assert not (
                signal.signal_type == SignalType.ENTRY and signal.direction == Direction.LONG
            )


class TestDownsideBreakout:
    """Test downside breakout signal generation."""

    def test_downside_breakout_with_volume_confirmation(self):
        """Test downside breakout signal with volume confirmation."""
        strategy = MomentumBreakoutStrategy(
            name="test", lookback_period=5, atr_period=3, breakout_threshold=0.02
        )

        # Downside breakout: close < low * 0.98, volume surge
        data = create_test_data(
            bars=21,
            open=[102] * 20 + [95],
            high=[105] * 20 + [103],
            low=[100] * 20 + [92],  # Rolling min = 92
            close=[102] * 20 + [90],  # 90 < 92 * 0.98 = 90.16
            volume=[1000] * 20 + [2000],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.signal_type == SignalType.ENTRY
        assert signal.direction == Direction.SHORT
        assert signal.probability > 0.55
        assert signal.expected_return < 0
        assert signal.metadata["breakout_type"] == "downside"


class TestConsolidationDetection:
    """Test consolidation signal generation."""

    def test_consolidation_signal(self):
        """Test consolidation detection."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=5, atr_period=3)

        # Tight range < 2%
        data = create_test_data(
            bars=21,
            high=[100.5] * 21,
            low=[99.5] * 21,  # 1% range
            close=[100] * 21,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.signal_type == SignalType.HOLD
        assert signal.direction == Direction.NEUTRAL
        assert signal.metadata["market_state"] == "consolidation"


class TestSignalMetadata:
    """Test signal metadata fields."""

    def test_upside_breakout_metadata_complete(self):
        """Test all metadata fields present."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=5, atr_period=3)

        data = create_test_data(
            bars=21,
            high=[100] * 20 + [103],
            close=[98] * 20 + [106],
            volume=[1000] * 20 + [2000],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        metadata = signal.metadata

        assert "strategy" in metadata
        assert "breakout_type" in metadata
        assert "distance_from_high" in metadata
        assert "volume_ratio" in metadata
        assert "atr" in metadata
        assert "stop_loss" in metadata
        assert "take_profit" in metadata


class TestDataValidation:
    """Test data validation and error handling."""

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=20)

        data = create_test_data(bars=10)  # Need 20

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_missing_columns_returns_none(self):
        """Test that missing columns returns None."""
        strategy = MomentumBreakoutStrategy(name="test")

        data = pd.DataFrame({"high": [100] * 30, "low": [95] * 30, "close": [98] * 30})

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None


class TestSymbolHandling:
    """Test symbol extraction from data."""

    def test_symbol_from_data(self):
        """Test symbol extraction when present."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=5)

        data = create_test_data(bars=21, high=[100.5] * 21, low=[99.5] * 21)
        data["symbol"] = "TSLA"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.symbol == "TSLA"

    def test_default_symbol_when_missing(self):
        """Test default symbol when not in data."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=5)

        data = create_test_data(bars=21, high=[100.5] * 21, low=[99.5] * 21)

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            assert signal.symbol == "UNKNOWN"


class TestStrategyDescription:
    """Test strategy description."""

    def test_get_description_format(self):
        """Test description contains key information."""
        strategy = MomentumBreakoutStrategy(name="test")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert len(description) > 100
        assert "Momentum Breakout" in description
        assert "Entry Rules" in description
        assert "Exit Rules" in description


class TestEdgeCases:
    """Test edge cases."""

    def test_probability_capping(self):
        """Test that probability is capped at 0.75."""
        strategy = MomentumBreakoutStrategy(
            name="test", lookback_period=5, atr_period=3, breakout_threshold=0.02
        )

        # Extreme breakout
        data = create_test_data(
            bars=21,
            high=[100] * 20 + [102],
            close=[98] * 20 + [125],  # Very large breakout
            volume=[1000] * 20 + [5000],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.probability <= 0.75
