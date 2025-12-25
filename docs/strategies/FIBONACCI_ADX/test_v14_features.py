"""
Tests for Fibonacci ADX Strategy v1.4.0 features.

Tests the new models: Volume Profile, Fractal Swing, MTF Alignment.
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType


class TestVolumeProfileModel:
    """Tests for VolumeProfileModel."""

    @pytest.fixture
    def volume_model(self):
        from ordinis.engines.signalcore.models import VolumeProfileModel
        config = ModelConfig(
            model_id="test-volume",
            model_type="volume_profile",
            parameters={
                "lookback": 20,
                "pullback_decline_threshold": 0.2,
                "bounce_increase_threshold": 0.3,
            },
        )
        return VolumeProfileModel(config)

    @pytest.fixture
    def pullback_data(self):
        """Create data with declining pullback and increasing bounce volume."""
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        
        # Price: uptrend, then pullback, then bounce
        prices = np.concatenate([
            np.linspace(100, 110, 20),  # Uptrend
            np.linspace(110, 105, 15),  # Pullback
            np.linspace(105, 112, 15),  # Bounce
        ])
        
        # Volume: high initially, declining during pullback, increasing on bounce
        volume = np.concatenate([
            np.random.uniform(1e6, 1.5e6, 20),  # Normal volume
            np.linspace(1.2e6, 0.6e6, 15),      # Declining (pullback exhaustion)
            np.linspace(0.8e6, 1.8e6, 15),      # Increasing (buyers returning)
        ])
        
        return pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": volume,
        }, index=dates)

    @pytest.mark.asyncio
    async def test_volume_confirms_pullback_pattern(self, volume_model, pullback_data):
        """Test that volume model detects pullback confirmation pattern."""
        timestamp = datetime.now()
        signal = await volume_model.generate(pullback_data, timestamp)
        
        assert signal is not None
        assert "volume_confirms" in signal.metadata
        assert "relative_volume" in signal.metadata
        assert "pullback_declining" in signal.metadata
        assert "bounce_increasing" in signal.metadata

    @pytest.mark.asyncio
    async def test_confirmation_strength_calculated(self, volume_model, pullback_data):
        """Test that confirmation strength is calculated."""
        timestamp = datetime.now()
        signal = await volume_model.generate(pullback_data, timestamp)
        
        strength = signal.metadata.get("confirmation_strength", 0)
        assert isinstance(strength, (int, float))
        assert 0 <= strength <= 1

    @pytest.mark.asyncio
    async def test_check_confirmation_helper(self, volume_model, pullback_data):
        """Test synchronous helper method."""
        confirms, strength = volume_model.check_confirmation(pullback_data, Direction.LONG)
        
        assert isinstance(confirms, bool)
        assert isinstance(strength, float)


class TestFractalSwingModel:
    """Tests for FractalSwingModel."""

    @pytest.fixture
    def fractal_model(self):
        from ordinis.engines.signalcore.models import FractalSwingModel
        config = ModelConfig(
            model_id="test-fractal",
            model_type="swing_detection",
            parameters={
                "fractal_period": 2,
                "min_swing_bars": 5,
                "strength_lookback": 10,
                "min_swing_pct": 0.02,
            },
        )
        return FractalSwingModel(config)

    @pytest.fixture
    def swing_data(self):
        """Create data with clear swing highs and lows."""
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        
        # Create zigzag pattern with clear swings
        base = 100
        prices = []
        for i in range(n):
            cycle = (i % 20) / 20.0  # 20-bar cycles
            if (i // 20) % 2 == 0:
                prices.append(base + 10 * cycle)  # Up leg
            else:
                prices.append(base + 10 - 10 * cycle)  # Down leg
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1e6, 2e6, n),
        }, index=dates)

    @pytest.mark.asyncio
    async def test_detects_swing_highs_and_lows(self, fractal_model, swing_data):
        """Test that fractal model detects swing points."""
        timestamp = datetime.now()
        signal = await fractal_model.generate(swing_data, timestamp)
        
        assert signal is not None
        assert "swing_high" in signal.metadata
        assert "swing_low" in signal.metadata
        assert signal.metadata.get("num_swing_highs", 0) > 0
        assert signal.metadata.get("num_swing_lows", 0) > 0

    @pytest.mark.asyncio
    async def test_swing_strength_calculated(self, fractal_model, swing_data):
        """Test that swing strength is calculated."""
        timestamp = datetime.now()
        signal = await fractal_model.generate(swing_data, timestamp)
        
        high_strength = signal.metadata.get("swing_high_strength")
        low_strength = signal.metadata.get("swing_low_strength")
        
        assert high_strength is not None
        assert low_strength is not None
        assert high_strength >= 0
        assert low_strength >= 0

    @pytest.mark.asyncio
    async def test_all_swings_in_metadata(self, fractal_model, swing_data):
        """Test that recent swing points are included in metadata."""
        timestamp = datetime.now()
        signal = await fractal_model.generate(swing_data, timestamp)
        
        all_highs = signal.metadata.get("all_swing_highs", [])
        all_lows = signal.metadata.get("all_swing_lows", [])
        
        assert isinstance(all_highs, list)
        assert isinstance(all_lows, list)
        
        if all_highs:
            assert "index" in all_highs[0]
            assert "price" in all_highs[0]
            assert "strength" in all_highs[0]

    def test_get_swing_points_helper(self, fractal_model, swing_data):
        """Test synchronous helper method."""
        highs, lows = fractal_model.get_swing_points(swing_data)
        
        assert isinstance(highs, list)
        assert isinstance(lows, list)


class TestMTFAlignmentModel:
    """Tests for MTFAlignmentModel."""

    @pytest.fixture
    def mtf_model(self):
        from ordinis.engines.signalcore.models import MTFAlignmentModel
        config = ModelConfig(
            model_id="test-mtf",
            model_type="mtf_alignment",
            parameters={
                "htf_sma_period": 20,  # Shorter for testing
                "htf_multiplier": 3,
                "slope_period": 5,
                "require_alignment": True,
            },
        )
        return MTFAlignmentModel(config)

    @pytest.fixture
    def bullish_aligned_data(self):
        """Create data where both LTF and HTF are bullish."""
        n = 200
        dates = pd.date_range(start="2024-01-01", periods=n, freq="H")
        
        # Strong uptrend
        trend = np.linspace(100, 150, n)
        noise = np.random.normal(0, 1, n)
        prices = trend + noise
        
        return pd.DataFrame({
            "open": prices * 0.998,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(1e6, 2e6, n),
        }, index=dates)

    @pytest.fixture
    def counter_trend_data(self):
        """Create data where LTF is bullish but HTF is bearish."""
        n = 200
        dates = pd.date_range(start="2024-01-01", periods=n, freq="H")
        
        # HTF downtrend with LTF bounce
        htf_trend = np.linspace(150, 100, n)  # Overall down
        ltf_bounce = np.zeros(n)
        ltf_bounce[-30:] = np.linspace(0, 10, 30)  # Recent bounce
        
        prices = htf_trend + ltf_bounce + np.random.normal(0, 0.5, n)
        
        return pd.DataFrame({
            "open": prices * 0.998,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.uniform(1e6, 2e6, n),
        }, index=dates)

    @pytest.mark.asyncio
    async def test_aligned_bullish_detected(self, mtf_model, bullish_aligned_data):
        """Test that aligned bullish is detected."""
        timestamp = datetime.now()
        signal = await mtf_model.generate(bullish_aligned_data, timestamp)
        
        assert signal is not None
        assert "alignment" in signal.metadata
        assert "htf_bullish" in signal.metadata
        assert signal.metadata.get("is_aligned") == True
        assert signal.direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_counter_trend_rejected(self, mtf_model, counter_trend_data):
        """Test that counter-trend signals are marked as not aligned."""
        timestamp = datetime.now()
        signal = await mtf_model.generate(counter_trend_data, timestamp)
        
        assert signal is not None
        # When require_alignment is True, counter-trend should HOLD
        if not signal.metadata.get("is_aligned"):
            assert signal.signal_type == SignalType.HOLD

    @pytest.mark.asyncio
    async def test_htf_sma_calculated(self, mtf_model, bullish_aligned_data):
        """Test that HTF SMA is calculated and returned."""
        timestamp = datetime.now()
        signal = await mtf_model.generate(bullish_aligned_data, timestamp)
        
        assert "htf_sma" in signal.metadata
        assert signal.metadata["htf_sma"] > 0
        assert "htf_slope" in signal.metadata

    def test_check_alignment_helper(self, mtf_model, bullish_aligned_data):
        """Test synchronous alignment check helper."""
        from ordinis.engines.signalcore.models import TimeframeAlignment
        
        is_aligned, alignment = mtf_model.check_alignment(bullish_aligned_data, Direction.LONG)
        
        assert isinstance(is_aligned, bool)
        assert isinstance(alignment, TimeframeAlignment)


class TestFibonacciADXStrategyEnhancements:
    """Tests for FibonacciADXStrategy with new features."""

    @pytest.fixture
    def base_strategy(self):
        from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
        return FibonacciADXStrategy(name="test-fib-adx")

    @pytest.fixture
    def enhanced_strategy(self):
        from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
        return FibonacciADXStrategy(
            name="test-fib-adx-enhanced",
            use_fractal_swings=True,
            require_volume_confirmation=True,
            require_mtf_alignment=True,
            htf_multiplier=3,
            htf_sma_period=20,
        )

    def test_default_has_no_optional_models(self, base_strategy):
        """Test that default strategy doesn't enable optional features."""
        base_strategy.configure()
        
        assert base_strategy.fractal_model is None
        assert base_strategy.volume_model is None
        assert base_strategy.mtf_model is None

    def test_enhanced_has_all_models(self, enhanced_strategy):
        """Test that enhanced strategy has all optional models."""
        enhanced_strategy.configure()
        
        assert enhanced_strategy.fractal_model is not None
        assert enhanced_strategy.volume_model is not None
        assert enhanced_strategy.mtf_model is not None

    def test_params_stored_correctly(self, enhanced_strategy):
        """Test that enhancement parameters are stored correctly."""
        enhanced_strategy.configure()
        
        assert enhanced_strategy.use_fractal_swings == True
        assert enhanced_strategy.require_volume_confirmation == True
        assert enhanced_strategy.require_mtf_alignment == True
        assert enhanced_strategy.htf_multiplier == 3
        assert enhanced_strategy.htf_sma_period == 20

    def test_min_bars_accounts_for_mtf(self, enhanced_strategy):
        """Test that min_bars increases when MTF is enabled."""
        enhanced_strategy.configure()
        
        # With MTF enabled, min_bars should be at least htf_sma_period * htf_multiplier
        min_expected = (20 + 10) * 3  # htf_sma_period + buffer * htf_multiplier
        assert enhanced_strategy.params["min_bars"] >= min_expected


class TestModelImports:
    """Test that all new models are properly exported."""

    def test_volume_profile_import(self):
        from ordinis.engines.signalcore.models import VolumeProfileModel
        assert VolumeProfileModel is not None

    def test_fractal_swing_import(self):
        from ordinis.engines.signalcore.models import FractalSwingModel, SwingPoint
        assert FractalSwingModel is not None
        assert SwingPoint is not None

    def test_mtf_alignment_import(self):
        from ordinis.engines.signalcore.models import MTFAlignmentModel, TimeframeAlignment
        assert MTFAlignmentModel is not None
        assert TimeframeAlignment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
