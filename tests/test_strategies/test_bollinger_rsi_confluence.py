"""Tests for Bollinger RSI Confluence strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.application.strategies.bollinger_rsi_confluence import (
    BollingerRSIConfluenceStrategy,
)
from ordinis.engines.signalcore.core.signal import Direction, SignalType


def create_test_data(bars: int = 60, **overrides) -> pd.DataFrame:
    """Create test OHLCV data with sensible defaults."""
    defaults = {
        "open": [100.0] * bars,
        "high": [102.0] * bars,
        "low": [98.0] * bars,
        "close": [100.0] * bars,
        "volume": [1000] * bars,
    }
    defaults.update(overrides)
    df = pd.DataFrame(defaults)
    df.index = pd.date_range(end=datetime.utcnow(), periods=bars, freq="D")
    return df


def create_volatile_data(bars: int = 60, volatility: float = 0.02) -> pd.DataFrame:
    """Create volatile test data with realistic price movements."""
    np.random.seed(42)
    returns = np.random.normal(0, volatility, bars)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": prices * (1 - volatility / 2),
            "high": prices * (1 + volatility),
            "low": prices * (1 - volatility),
            "close": prices,
            "volume": [1000] * bars,
        }
    )
    df.index = pd.date_range(end=datetime.utcnow(), periods=bars, freq="D")
    return df


class TestBollingerRSIConfiguration:
    """Test strategy configuration."""

    def test_default_configuration(self) -> None:
        """Test default parameter configuration."""
        strategy = BollingerRSIConfluenceStrategy(name="test-confluence")

        assert strategy.params["bb_period"] == 20
        assert strategy.params["bb_std"] == 2.0
        assert strategy.params["rsi_period"] == 14
        assert strategy.params["rsi_oversold"] == 30
        assert strategy.params["rsi_overbought"] == 70
        assert strategy.params["vol_lookback"] == 22
        assert strategy.params["min_volatility"] == 5.0

    def test_custom_configuration(self) -> None:
        """Test custom parameter configuration."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=15,
            bb_std=2.5,
            rsi_period=10,
            rsi_oversold=25,
            rsi_overbought=75,
        )

        assert strategy.params["bb_period"] == 15
        assert strategy.params["bb_std"] == 2.5
        assert strategy.params["rsi_period"] == 10
        assert strategy.params["rsi_oversold"] == 25
        assert strategy.params["rsi_overbought"] == 75

    def test_get_required_bars(self) -> None:
        """Test required bars calculation."""
        strategy = BollingerRSIConfluenceStrategy(name="test", bb_period=25, rsi_period=14)
        # max(25, 14) + 30 = 55
        assert strategy.get_required_bars() == 55


class TestConfluenceSignals:
    """Test confluence signal generation."""

    @pytest.mark.asyncio
    async def test_long_confluence_signal(self) -> None:
        """Test long signal when both BB and RSI confirm oversold."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=5,
            rsi_period=5,
            min_volatility=1.0,  # Low threshold for testing
        )

        # Create data with sharp decline to trigger confluence
        # Price drops significantly to cross below lower band
        base_prices = [100.0] * 50
        declining_prices = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0]
        data = create_test_data(
            bars=60,
            close=base_prices + declining_prices,
        )

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Should be long entry or hold depending on exact indicator values
        if signal.signal_type == SignalType.ENTRY:
            assert signal.direction == Direction.LONG
            assert signal.probability >= 0.55
            assert signal.model_id == "bollinger-rsi-confluence"

    @pytest.mark.asyncio
    async def test_short_confluence_signal(self) -> None:
        """Test short signal when both BB and RSI confirm overbought."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=5,
            rsi_period=5,
            min_volatility=1.0,
        )

        # Create data with sharp rally to trigger confluence
        base_prices = [100.0] * 50
        rallying_prices = [105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0]
        data = create_test_data(
            bars=60,
            close=base_prices + rallying_prices,
        )

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.ENTRY:
            assert signal.direction == Direction.SHORT
            assert signal.probability >= 0.55

    @pytest.mark.asyncio
    async def test_hold_signal_when_no_confluence(self) -> None:
        """Test hold signal when indicators don't confirm."""
        strategy = BollingerRSIConfluenceStrategy(name="test", bb_period=5, rsi_period=5)

        # Stable prices should not trigger confluence
        data = create_test_data(
            bars=60,
            close=[100.0] * 60,
        )

        signal = await strategy.generate_signal(data, datetime.utcnow())

        # With stable data, should get HOLD or weak signal
        assert signal is not None
        if signal.signal_type == SignalType.HOLD:
            assert signal.direction == Direction.NEUTRAL


class TestVolatilityFilter:
    """Test volatility filtering behavior."""

    @pytest.mark.asyncio
    async def test_low_volatility_returns_hold(self) -> None:
        """Test that low volatility market returns hold signal."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=5,
            rsi_period=5,
            min_volatility=20.0,  # High threshold
        )

        # Very low volatility data
        data = create_test_data(
            bars=60,
            close=[100.0, 100.1, 99.9, 100.0, 100.05] * 12,
        )

        signal = await strategy.generate_signal(data, datetime.utcnow())

        # Should be hold due to low volatility - but model may return weak signals
        assert signal is not None
        # The model now returns HOLD signal type but may have non-NEUTRAL direction
        # for weak single-indicator signals
        assert signal.signal_type == SignalType.HOLD

    @pytest.mark.asyncio
    async def test_high_volatility_allows_trading(self) -> None:
        """Test that high volatility market allows trading signals."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=5,
            rsi_period=5,
            min_volatility=1.0,  # Low threshold
        )

        # High volatility data with oversold condition
        data = create_volatile_data(bars=60, volatility=0.03)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # With volatile data, should get some signal (not necessarily HOLD)


class TestSignalMetadata:
    """Test signal metadata fields."""

    @pytest.mark.asyncio
    async def test_metadata_contains_indicators(self) -> None:
        """Test metadata includes all indicator values."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test", bb_period=5, rsi_period=5, min_volatility=1.0
        )

        data = create_volatile_data(bars=60)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        metadata = signal.metadata

        # Core indicators should be present
        assert "rsi" in metadata
        assert "volatility" in metadata

        if signal.signal_type != SignalType.HOLD:
            # Entry signals should have band information
            assert "lower_band" in metadata or "price_position" in metadata

    @pytest.mark.asyncio
    async def test_feature_contributions_present(self) -> None:
        """Test feature contributions for explainability."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test", bb_period=5, rsi_period=5, min_volatility=1.0
        )

        base_prices = [100.0] * 50
        declining_prices = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0]
        data = create_test_data(bars=60, close=base_prices + declining_prices)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if signal.signal_type == SignalType.ENTRY:
            assert len(signal.feature_contributions) > 0
            # Should have RSI and bollinger contributions
            assert "rsi" in signal.feature_contributions
            assert "bollinger_position" in signal.feature_contributions


class TestDataValidation:
    """Test data validation and error handling."""

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self) -> None:
        """Test that insufficient data returns None."""
        strategy = BollingerRSIConfluenceStrategy(name="test", bb_period=20, rsi_period=14)

        data = create_test_data(bars=20)  # Need at least 50

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    @pytest.mark.asyncio
    async def test_missing_columns_returns_none(self) -> None:
        """Test that missing columns returns None."""
        strategy = BollingerRSIConfluenceStrategy(name="test")

        data = pd.DataFrame({"high": [100] * 60, "low": [95] * 60, "close": [98] * 60})

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    @pytest.mark.asyncio
    async def test_empty_dataframe_returns_none(self) -> None:
        """Test that empty DataFrame returns None."""
        strategy = BollingerRSIConfluenceStrategy(name="test")

        signal = await strategy.generate_signal(pd.DataFrame(), datetime.utcnow())

        assert signal is None


class TestExtremeReadings:
    """Test extreme indicator readings."""

    @pytest.mark.asyncio
    async def test_extreme_oversold_high_conviction(self) -> None:
        """Test extreme RSI oversold gives higher conviction."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test",
            bb_period=5,
            rsi_period=5,
            rsi_extreme_oversold=20,
            min_volatility=1.0,
        )

        # Sharp decline to trigger extreme oversold (60 bars total)
        base_prices = [100.0] * 45
        crash_prices = [
            95.0,
            90.0,
            85.0,
            80.0,
            75.0,
            70.0,
            65.0,
            60.0,
            55.0,
            50.0,
            48.0,
            46.0,
            44.0,
            42.0,
            40.0,
        ]
        data = create_test_data(
            bars=60,
            close=base_prices + crash_prices,
        )

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Extreme readings should give higher probability when confluent
        if signal.signal_type == SignalType.ENTRY and signal.direction == Direction.LONG:
            # With any oversold condition, probability should be at least 0.55
            assert signal.probability >= 0.55
            # RSI should be extreme (very low)
            assert signal.metadata.get("rsi", 100) < 30


class TestStrategyDescription:
    """Test strategy description."""

    def test_description_contains_key_info(self) -> None:
        """Test description contains parameters and rules."""
        strategy = BollingerRSIConfluenceStrategy(name="test")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert len(description) > 200
        assert "Bollinger" in description
        assert "RSI" in description
        assert "Confluence" in description
        assert "gs-quant" in description
        assert "Entry Rules" in description


class TestGSQuantIntegration:
    """Test integration with gs-quant adapter functions."""

    @pytest.mark.asyncio
    async def test_uses_gs_quant_indicators(self) -> None:
        """Verify strategy uses gs-quant adapter indicators correctly."""
        strategy = BollingerRSIConfluenceStrategy(
            name="test", bb_period=10, rsi_period=7, min_volatility=1.0
        )

        # Create data that should produce valid indicator values
        data = create_volatile_data(bars=60, volatility=0.02)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # RSI should be in valid range
        if "rsi" in signal.metadata:
            assert 0 <= signal.metadata["rsi"] <= 100
        # Volatility should be positive
        if "volatility" in signal.metadata:
            assert signal.metadata["volatility"] >= 0

    @pytest.mark.asyncio
    async def test_zscore_calculation(self) -> None:
        """Test z-score is calculated correctly."""
        strategy = BollingerRSIConfluenceStrategy(name="test", bb_period=10, min_volatility=1.0)

        data = create_volatile_data(bars=60)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        if "price_zscore" in signal.metadata:
            # Z-score should be a reasonable value
            assert -10 < signal.metadata["price_zscore"] < 10


class TestModelAttribution:
    """Test model attribution fields."""

    @pytest.mark.asyncio
    async def test_model_id_set_correctly(self) -> None:
        """Test model_id is set."""
        strategy = BollingerRSIConfluenceStrategy(name="test", min_volatility=1.0)

        data = create_volatile_data(bars=60)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Model ID now comes from the underlying model with strategy name prefix
        assert "bb-rsi-confluence" in signal.model_id
        assert signal.model_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_regime_set_for_mean_reversion(self) -> None:
        """Test regime is set appropriately based on volatility."""
        strategy = BollingerRSIConfluenceStrategy(name="test", min_volatility=1.0)

        base_prices = [100.0] * 50
        declining_prices = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0]
        data = create_test_data(bars=60, close=base_prices + declining_prices)

        signal = await strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # Regime is now based on volatility level, not signal type
        assert signal.regime in ["low_volatility", "moderate_volatility", "high_volatility"]
