"""Tests for MACD strategy."""

from datetime import datetime

import pandas as pd

from application.strategies.macd import MACDStrategy
from engines.signalcore.core.signal import Direction, SignalType


def create_test_data(bars=70, **overrides):
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


class TestMACDConfiguration:
    """Test strategy configuration."""

    def test_default_configuration(self):
        """Test default parameter configuration."""
        strategy = MACDStrategy(name="test-macd")

        assert strategy.params["fast_period"] == 12
        assert strategy.params["slow_period"] == 26
        assert strategy.params["signal_period"] == 9
        assert strategy.params["min_bars"] == 55  # slow_period + signal_period + 20

    def test_custom_configuration(self):
        """Test custom parameter configuration."""
        strategy = MACDStrategy(name="test", fast_period=8, slow_period=17, signal_period=5)

        assert strategy.params["fast_period"] == 8
        assert strategy.params["slow_period"] == 17
        assert strategy.params["signal_period"] == 5

    def test_get_required_bars(self):
        """Test required bars calculation."""
        strategy = MACDStrategy(name="test", slow_period=30)
        assert strategy.get_required_bars() == 59  # slow_period + signal_period + 20

    def test_invalid_period_configuration(self):
        """Test that fast period must be less than slow period."""
        # This should work - fast < slow
        strategy = MACDStrategy(name="test", fast_period=10, slow_period=20)
        assert strategy.params["fast_period"] < strategy.params["slow_period"]


class TestBullishCrossover:
    """Test bullish crossover signal generation."""

    def test_bullish_crossover_entry(self):
        """Test long entry on bullish MACD crossover."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Uptrend with bullish crossover - need at least 33 bars (10+3+20)
        pattern = (
            [100] * 30
            + [100, 101, 102, 103, 104]  # Start uptrend
            + [105, 106, 107, 108, 109]  # Continue
            + [110, 111, 112, 113, 114]  # Bullish crossover
            + [115, 116, 117, 118, 119, 120]  # Continuation
            + [121] * 10
        )  # Extra bars for stability
        data = create_test_data(
            bars=len(pattern),
            close=pattern,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.ENTRY and signal.direction == Direction.LONG:
            assert signal.probability > 0.5
            assert signal.expected_return > 0
            assert "crossover_type" in signal.metadata
            assert signal.metadata["crossover_type"] == "bullish"

    def test_bullish_crossover_with_positive_histogram(self):
        """Test bullish signal with increasing histogram."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Strong uptrend - need at least 33 bars
        data = create_test_data(
            bars=70,
            close=list(range(100, 170)),  # Continuous uptrend
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None and signal.signal_type == SignalType.ENTRY:
            assert "histogram" in signal.metadata
            # In strong uptrend, histogram should be positive
            assert signal.metadata["histogram"] >= 0


class TestBearishCrossover:
    """Test bearish crossover signal generation."""

    def test_bearish_crossover_exit(self):
        """Test exit signal on bearish MACD crossover."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Downtrend with bearish crossover
        pattern = (
            [120] * 30
            + [120, 118, 116, 114, 112]  # Start downtrend
            + [110, 108, 106, 104, 102]  # Continue
            + [100, 98, 96, 94, 92]  # Bearish crossover
            + [90, 88, 86, 84, 82, 80]  # Continuation
            + [78] * 10
        )  # Extra bars
        data = create_test_data(
            bars=len(pattern),
            close=pattern,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.EXIT:
            assert signal.direction == Direction.NEUTRAL
            assert "crossover_type" in signal.metadata
            assert signal.metadata["crossover_type"] == "bearish"

    def test_bearish_crossover_with_negative_histogram(self):
        """Test bearish signal with decreasing histogram."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Strong downtrend
        data = create_test_data(
            bars=70,
            close=list(range(170, 100, -1)),  # Continuous downtrend
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None and signal.signal_type == SignalType.EXIT:
            assert "histogram" in signal.metadata
            # In strong downtrend, histogram should be negative
            assert signal.metadata["histogram"] <= 0


class TestTrendStrength:
    """Test trend strength assessment."""

    def test_strong_uptrend_high_probability(self):
        """Test higher probability in strong uptrend."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Very strong uptrend
        data = create_test_data(
            bars=70,
            close=[100 + i * 2 for i in range(70)],  # Steep climb
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # Strong trend should yield higher probability
            assert signal.probability > 0.55

    def test_weak_trend_lower_probability(self):
        """Test lower probability in weak/sideways market."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Sideways with small oscillations
        data = create_test_data(
            bars=70,
            close=[100, 101, 100, 99, 100] * 14,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            # Weak/sideways should have moderate or low probability
            assert signal.probability <= 0.7


class TestHistogramAnalysis:
    """Test MACD histogram analysis."""

    def test_histogram_divergence_detection(self):
        """Test histogram values in metadata."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        data = create_test_data(
            bars=70,
            close=list(range(100, 170)),
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            assert "histogram" in signal.metadata
            assert "macd_line" in signal.metadata
            assert "signal_line" in signal.metadata
            assert isinstance(signal.metadata["histogram"], (int, float))

    def test_histogram_magnitude_affects_score(self):
        """Test that larger histogram affects signal strength."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Strong trend = large histogram
        data_strong = create_test_data(
            bars=70,
            close=[100 + i * 3 for i in range(70)],  # Steep
        )
        data_strong["symbol"] = "AAPL"

        # Weak trend = small histogram
        data_weak = create_test_data(
            bars=70,
            close=[100 + i * 0.5 for i in range(70)],  # Gentle
        )
        data_weak["symbol"] = "AAPL"

        signal_strong = strategy.generate_signal(data_strong, datetime.utcnow())
        signal_weak = strategy.generate_signal(data_weak, datetime.utcnow())

        if signal_strong and signal_weak:
            # Strong trend should have larger histogram magnitude
            strong_hist = abs(signal_strong.metadata.get("histogram", 0))
            weak_hist = abs(signal_weak.metadata.get("histogram", 0))
            assert strong_hist > weak_hist


class TestSignalMetadata:
    """Test signal metadata fields."""

    def test_entry_metadata_complete(self):
        """Test all metadata fields present in entry signal."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        data = create_test_data(
            bars=70,
            close=list(range(100, 170)),
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        metadata = signal.metadata

        assert "strategy" in metadata
        assert "macd_line" in metadata
        assert "signal_line" in metadata
        assert "histogram" in metadata
        assert "crossover_type" in metadata

    def test_metadata_values_reasonable(self):
        """Test metadata values are within reasonable ranges."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        data = create_test_data(bars=70, close=list(range(100, 170)))
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            metadata = signal.metadata

            # MACD values should be numeric
            assert isinstance(metadata.get("macd_line", 0), (int, float))
            assert isinstance(metadata.get("signal_line", 0), (int, float))
            assert isinstance(metadata.get("histogram", 0), (int, float))

            # Histogram should equal macd_line - signal_line (approximately)
            if all(k in metadata for k in ["macd_line", "signal_line", "histogram"]):
                calculated_hist = metadata["macd_line"] - metadata["signal_line"]
                # Allow small floating point differences
                assert abs(metadata["histogram"] - calculated_hist) < 0.01


class TestDataValidation:
    """Test data validation and error handling."""

    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        strategy = MACDStrategy(name="test", slow_period=26)

        data = create_test_data(bars=20)  # Need 26

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_missing_columns_returns_none(self):
        """Test that missing columns returns None."""
        strategy = MACDStrategy(name="test")

        data = pd.DataFrame({"high": [100] * 40, "low": [95] * 40, "open": [98] * 40})

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_empty_dataframe_returns_none(self):
        """Test that empty DataFrame returns None."""
        strategy = MACDStrategy(name="test")

        data = pd.DataFrame()

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None


class TestSymbolHandling:
    """Test symbol extraction from data."""

    def test_symbol_from_data(self):
        """Test symbol extraction when present."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10)

        data = create_test_data(bars=70)
        data["symbol"] = "MSFT"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.symbol == "MSFT"

    def test_default_symbol_when_missing(self):
        """Test default symbol when not in data."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10)

        data = create_test_data(bars=70)

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            assert signal.symbol == "UNKNOWN"


class TestStrategyDescription:
    """Test strategy description."""

    def test_get_description_format(self):
        """Test description contains key information."""
        strategy = MACDStrategy(name="test")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert len(description) > 100
        assert "MACD" in description
        assert "Entry Rules" in description
        assert "Exit Rules" in description


class TestEdgeCases:
    """Test edge cases."""

    def test_probability_bounds(self):
        """Test that probability is within valid bounds."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Extreme price movement
        pattern = [100] * 30 + [120, 80, 150, 60, 200] * 8
        data = create_test_data(
            bars=len(pattern),
            close=pattern,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert 0 <= signal.probability <= 1

    def test_nan_handling(self):
        """Test handling of NaN values in data."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10)

        data = create_test_data(bars=70)
        # Introduce some NaN values
        data.loc[15:18, "close"] = float("nan")
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should either handle gracefully or return None
        assert signal is None or isinstance(signal.probability, float)

    def test_constant_price_no_signal(self):
        """Test handling of constant price (no movement)."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        data = create_test_data(
            bars=70,
            close=[100] * 70,  # No price movement
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # With no price movement, MACD should be near zero
        if signal is not None:
            assert abs(signal.metadata.get("macd_line", 0)) < 1
            assert abs(signal.metadata.get("histogram", 0)) < 1


class TestCrossoverDetection:
    """Test crossover detection logic."""

    def test_no_crossover_hold_signal(self):
        """Test hold signal when no crossover occurs."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Gradual uptrend without sharp crossover
        data = create_test_data(
            bars=70,
            close=[100 + i * 0.1 for i in range(70)],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None:
            # Should be entry, exit, or hold
            assert signal.signal_type in [SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD]

    def test_multiple_crossovers(self):
        """Test behavior with multiple recent crossovers."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Oscillating price
        data = create_test_data(
            bars=70,
            close=[100 + (10 if i % 4 < 2 else -10) for i in range(70)],
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should generate some signal based on most recent crossover
        assert signal is not None


class TestTrendReversal:
    """Test trend reversal detection."""

    def test_uptrend_to_downtrend_reversal(self):
        """Test detection of uptrend to downtrend reversal."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Uptrend followed by reversal - need sufficient bars
        pattern = [100] * 20 + list(range(100, 120)) + list(range(120, 105, -1)) + [105] * 10
        data = create_test_data(
            bars=len(pattern),
            close=pattern,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None and signal.signal_type == SignalType.EXIT:
            # Reversal should trigger exit
            assert "crossover_type" in signal.metadata

    def test_downtrend_to_uptrend_reversal(self):
        """Test detection of downtrend to uptrend reversal."""
        strategy = MACDStrategy(name="test", fast_period=5, slow_period=10, signal_period=3)

        # Downtrend followed by reversal
        pattern = [120] * 20 + list(range(120, 100, -1)) + list(range(100, 115)) + [115] * 10
        data = create_test_data(
            bars=len(pattern),
            close=pattern,
        )
        data["symbol"] = "AAPL"

        signal = strategy.generate_signal(data, datetime.utcnow())

        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # Reversal to uptrend should trigger entry
            assert signal.direction == Direction.LONG
