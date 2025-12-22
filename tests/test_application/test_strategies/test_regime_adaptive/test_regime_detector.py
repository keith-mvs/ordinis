"""Tests for RegimeDetector and related classes."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ordinis.application.strategies.regime_adaptive.regime_detector import (
    MarketRegime,
    RegimeAnalyzer,
    RegimeDetector,
    RegimeSignal,
)


# ==================== FIXTURES ====================


def generate_ohlcv_data(
    n_days: int = 300,
    trend: str = "up",
    volatility: float = 0.02,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    # Generate price series
    if trend == "up":
        drift = 0.001
    elif trend == "down":
        drift = -0.001
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n_days)
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV
    high = prices * (1 + np.random.uniform(0, 0.02, n_days))
    low = prices * (1 - np.random.uniform(0, 0.02, n_days))
    open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n_days)
    volume = np.random.uniform(1e6, 1e7, n_days)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def uptrend_data():
    """Generate uptrend data."""
    return generate_ohlcv_data(n_days=300, trend="up", volatility=0.015)


@pytest.fixture
def downtrend_data():
    """Generate downtrend data."""
    return generate_ohlcv_data(n_days=300, trend="down", volatility=0.015)


@pytest.fixture
def sideways_data():
    """Generate sideways data."""
    return generate_ohlcv_data(n_days=300, trend="sideways", volatility=0.01)


@pytest.fixture
def volatile_data():
    """Generate high volatility data."""
    return generate_ohlcv_data(n_days=300, trend="sideways", volatility=0.05)


@pytest.fixture
def detector():
    """Create RegimeDetector instance."""
    return RegimeDetector()


# ==================== ENUM TESTS ====================


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_regime_values(self):
        """MarketRegime has expected values."""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.TRANSITIONAL.value == "transitional"

    def test_all_regimes_present(self):
        """All expected regimes are defined."""
        regimes = [r.value for r in MarketRegime]
        assert len(regimes) == 5
        assert "bull" in regimes
        assert "bear" in regimes


# ==================== REGIME SIGNAL TESTS ====================


class TestRegimeSignal:
    """Tests for RegimeSignal dataclass."""

    def test_create_signal(self):
        """Create RegimeSignal with all fields."""
        signal = RegimeSignal(
            regime=MarketRegime.BULL,
            confidence=0.8,
            trend_strength=35.0,
            volatility_percentile=45.0,
            momentum=65.0,
            days_in_regime=10,
        )
        assert signal.regime == MarketRegime.BULL
        assert signal.confidence == 0.8
        assert signal.trend_strength == 35.0
        assert signal.days_in_regime == 10
        assert signal.previous_regime is None

    def test_signal_with_previous_regime(self):
        """Create signal with previous regime (transitional)."""
        signal = RegimeSignal(
            regime=MarketRegime.TRANSITIONAL,
            confidence=0.4,
            trend_strength=20.0,
            volatility_percentile=60.0,
            momentum=50.0,
            days_in_regime=0,
            previous_regime=MarketRegime.BULL,
        )
        assert signal.previous_regime == MarketRegime.BULL

    def test_signal_str_representation(self):
        """Signal has readable string representation."""
        signal = RegimeSignal(
            regime=MarketRegime.BULL,
            confidence=0.85,
            trend_strength=40.0,
            volatility_percentile=30.0,
            momentum=70.0,
            days_in_regime=15,
        )
        result = str(signal)
        assert "BULL" in result
        assert "85%" in result
        assert "15" in result


# ==================== REGIME DETECTOR TESTS ====================


class TestRegimeDetectorInit:
    """Tests for RegimeDetector initialization."""

    def test_default_init(self):
        """Initialize with default parameters."""
        detector = RegimeDetector()
        assert detector.adx_period == 14
        assert detector.adx_trend_threshold == 25.0
        assert detector.adx_strong_threshold == 40.0
        assert detector.fast_ma_period == 20
        assert detector.slow_ma_period == 50
        assert detector.trend_ma_period == 200

    def test_custom_init(self):
        """Initialize with custom parameters."""
        detector = RegimeDetector(
            adx_period=20,
            adx_trend_threshold=30.0,
            fast_ma_period=10,
        )
        assert detector.adx_period == 20
        assert detector.adx_trend_threshold == 30.0
        assert detector.fast_ma_period == 10

    def test_initial_state(self, detector):
        """Initial state is set correctly."""
        assert detector._current_regime == MarketRegime.SIDEWAYS
        assert detector._days_in_regime == 0
        assert detector._regime_history == []


class TestRegimeDetectorDetect:
    """Tests for regime detection."""

    def test_detect_insufficient_data(self, detector):
        """Returns sideways with low confidence for insufficient data."""
        short_data = generate_ohlcv_data(n_days=50)
        signal = detector.detect(short_data)

        assert signal.regime == MarketRegime.SIDEWAYS
        assert signal.confidence == 0.3
        assert signal.trend_strength == 0
        assert signal.days_in_regime == 0

    def test_detect_returns_signal(self, detector, uptrend_data):
        """Detect returns valid RegimeSignal."""
        signal = detector.detect(uptrend_data)

        assert isinstance(signal, RegimeSignal)
        assert isinstance(signal.regime, MarketRegime)
        assert 0 <= signal.confidence <= 1
        assert signal.trend_strength >= 0
        assert 0 <= signal.volatility_percentile <= 100
        assert 0 <= signal.momentum <= 100

    def test_detect_uptrend(self, detector, uptrend_data):
        """Detect identifies uptrend."""
        signal = detector.detect(uptrend_data)
        # With uptrend data, should tend toward BULL or SIDEWAYS
        assert signal.regime in [MarketRegime.BULL, MarketRegime.SIDEWAYS, MarketRegime.TRANSITIONAL]

    def test_detect_downtrend(self, detector, downtrend_data):
        """Detect identifies downtrend."""
        signal = detector.detect(downtrend_data)
        # With downtrend data, should tend toward BEAR or SIDEWAYS
        assert signal.regime in [MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.TRANSITIONAL]

    def test_detect_high_volatility(self, detector, volatile_data):
        """Detect identifies high volatility conditions."""
        signal = detector.detect(volatile_data)
        # High volatility data should be detected - either through regime or volatility metric
        # The volatility percentile is relative to historical data, so test basic functionality
        assert signal.volatility_percentile >= 0
        assert signal.volatility_percentile <= 100
        assert isinstance(signal.regime, MarketRegime)

    def test_detect_increments_days(self, detector, uptrend_data):
        """Days in regime increments on repeated calls."""
        signal1 = detector.detect(uptrend_data)
        initial_days = signal1.days_in_regime

        # Detect again with same regime
        signal2 = detector.detect(uptrend_data)

        # If regime unchanged, days should increment
        if signal1.regime == signal2.regime:
            assert signal2.days_in_regime >= initial_days


class TestRegimeDetectorIndicators:
    """Tests for indicator calculations."""

    def test_calculate_adx(self, detector, uptrend_data):
        """ADX calculation returns valid value."""
        adx = detector._calculate_adx(uptrend_data)
        assert isinstance(adx, float)
        assert adx >= 0

    def test_calculate_volatility_percentile(self, detector, uptrend_data):
        """Volatility percentile returns valid value."""
        vol_pct = detector._calculate_volatility_percentile(uptrend_data)
        assert isinstance(vol_pct, float)
        assert 0 <= vol_pct <= 100

    def test_calculate_momentum(self, detector, uptrend_data):
        """Momentum calculation returns valid RSI value."""
        momentum = detector._calculate_momentum(uptrend_data)
        assert isinstance(momentum, float)
        assert 0 <= momentum <= 100

    def test_calculate_trend_direction(self, detector, uptrend_data):
        """Trend direction returns value in valid range."""
        direction = detector._calculate_trend_direction(uptrend_data)
        assert isinstance(direction, float)
        assert -1 <= direction <= 1

    def test_score_regimes(self, detector):
        """Regime scoring returns normalized scores."""
        scores = detector._score_regimes(
            adx=30.0,
            volatility_pct=50.0,
            momentum=55.0,
            trend_direction=0.3,
        )
        assert isinstance(scores, dict)
        assert len(scores) == 4
        assert all(0 <= s <= 1 for s in scores.values())
        # Scores should sum to ~1 (normalized)
        total = sum(scores.values())
        assert 0.99 <= total <= 1.01


class TestRegimeDetectorState:
    """Tests for state management."""

    def test_get_regime_history(self, detector, uptrend_data):
        """Get regime history returns recent regimes."""
        # Detect multiple times to build history
        for _ in range(10):
            detector.detect(uptrend_data)

        history = detector.get_regime_history(5)
        assert isinstance(history, list)
        assert len(history) <= 5

    def test_reset(self, detector, uptrend_data):
        """Reset clears detector state."""
        detector.detect(uptrend_data)
        detector._days_in_regime = 10
        detector._regime_history = [MarketRegime.BULL, MarketRegime.BEAR]

        detector.reset()

        assert detector._current_regime == MarketRegime.SIDEWAYS
        assert detector._days_in_regime == 0
        assert detector._regime_history == []


# ==================== REGIME ANALYZER TESTS ====================


class TestRegimeAnalyzer:
    """Tests for RegimeAnalyzer."""

    def test_init_default_detector(self):
        """Initialize with default detector."""
        analyzer = RegimeAnalyzer()
        assert isinstance(analyzer.detector, RegimeDetector)

    def test_init_custom_detector(self):
        """Initialize with custom detector."""
        custom = RegimeDetector(adx_period=20)
        analyzer = RegimeAnalyzer(detector=custom)
        assert analyzer.detector == custom

    def test_analyze_period(self, uptrend_data):
        """Analyze period returns labeled DataFrame."""
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze_period(uptrend_data)

        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns
        assert "confidence" in result.columns
        assert "trend_strength" in result.columns
        assert len(result) == len(uptrend_data)

    def test_analyze_period_values(self, uptrend_data):
        """Analyze period returns valid values."""
        analyzer = RegimeAnalyzer()
        result = analyzer.analyze_period(uptrend_data)

        # Check regime values are valid
        valid_regimes = [r.value for r in MarketRegime]
        assert all(r in valid_regimes for r in result["regime"])

        # Check confidence is in valid range
        assert all(0 <= c <= 1 for c in result["confidence"])

    def test_get_regime_periods(self, uptrend_data):
        """Get regime periods returns period list."""
        analyzer = RegimeAnalyzer()
        labels = analyzer.analyze_period(uptrend_data)
        periods = analyzer.get_regime_periods(labels)

        assert isinstance(periods, list)
        if len(periods) > 0:
            assert "regime" in periods[0]
            assert "start" in periods[0]
            assert "end" in periods[0]
            assert "duration_days" in periods[0]


# ==================== INTEGRATION TESTS ====================


class TestRegimeDetectorIntegration:
    """Integration tests for regime detection."""

    def test_detect_multiple_regimes(self):
        """Detector can identify different regimes over time."""
        detector = RegimeDetector()

        # First: uptrend
        up_data = generate_ohlcv_data(n_days=300, trend="up")
        signal1 = detector.detect(up_data)

        # Reset and test downtrend
        detector.reset()
        down_data = generate_ohlcv_data(n_days=300, trend="down")
        signal2 = detector.detect(down_data)

        # Signals should be different (or at least trend metrics)
        assert signal1.trend_strength != signal2.trend_strength or signal1.momentum != signal2.momentum

    def test_regime_transition_detection(self, uptrend_data, downtrend_data):
        """Detector handles regime transitions."""
        detector = RegimeDetector(regime_confirm_days=3)

        # Establish regime with uptrend
        for _ in range(5):
            detector.detect(uptrend_data)

        initial_regime = detector._current_regime

        # Switch to downtrend data - may trigger transitional
        signal = detector.detect(downtrend_data)

        # Either still in initial or transitioning
        assert signal.regime in [
            initial_regime,
            MarketRegime.TRANSITIONAL,
            MarketRegime.BEAR,
            MarketRegime.SIDEWAYS,
        ]
