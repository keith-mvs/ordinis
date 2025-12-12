"""
Market Regime Detection System.

Identifies current market conditions to enable adaptive strategy selection:
- BULL: Strong uptrend, stay long, trend-following preferred
- BEAR: Strong downtrend, defensive/short, trend-following or cash
- SIDEWAYS: Range-bound, mean-reversion strategies preferred
- VOLATILE: High volatility, scalping/options strategies preferred
- TRANSITIONAL: Regime change detected, reduce exposure

Uses multiple indicators for robust regime classification:
- ADX for trend strength
- Price vs moving averages for direction
- Bollinger Band width for volatility
- RSI for momentum extremes
- Historical volatility percentile
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classification."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRANSITIONAL = "transitional"


@dataclass
class RegimeSignal:
    """Current regime signal with confidence and metadata."""

    regime: MarketRegime
    confidence: float  # 0-1, how certain we are
    trend_strength: float  # ADX value
    volatility_percentile: float  # Current vol vs historical
    momentum: float  # RSI or similar
    days_in_regime: int
    previous_regime: MarketRegime | None = None

    def __str__(self):
        return (
            f"{self.regime.value.upper()} (conf={self.confidence:.0%}, days={self.days_in_regime})"
        )


class RegimeDetector:
    """
    Multi-indicator market regime detection system.

    Combines several technical indicators to classify market conditions
    and provide confidence-weighted regime signals.
    """

    def __init__(
        self,
        # Trend detection params
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_strong_threshold: float = 40.0,
        # Moving average params
        fast_ma_period: int = 20,
        slow_ma_period: int = 50,
        trend_ma_period: int = 200,
        # Volatility params
        volatility_lookback: int = 20,
        volatility_history: int = 252,
        high_vol_percentile: float = 75.0,
        # Regime change params
        regime_confirm_days: int = 5,
        transition_threshold: float = 0.6,
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.trend_ma_period = trend_ma_period
        self.volatility_lookback = volatility_lookback
        self.volatility_history = volatility_history
        self.high_vol_percentile = high_vol_percentile
        self.regime_confirm_days = regime_confirm_days
        self.transition_threshold = transition_threshold

        # State tracking
        self._current_regime = MarketRegime.SIDEWAYS
        self._days_in_regime = 0
        self._regime_history: list[MarketRegime] = []

    def detect(self, data: pd.DataFrame) -> RegimeSignal:
        """
        Detect current market regime from OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  Index should be datetime

        Returns:
            RegimeSignal with current regime and metadata
        """
        if len(data) < self.trend_ma_period:
            return RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.3,
                trend_strength=0,
                volatility_percentile=50,
                momentum=50,
                days_in_regime=0,
            )

        # Calculate indicators
        adx = self._calculate_adx(data)
        volatility_pct = self._calculate_volatility_percentile(data)
        momentum = self._calculate_momentum(data)
        trend_direction = self._calculate_trend_direction(data)

        # Score each regime
        scores = self._score_regimes(
            adx=adx,
            volatility_pct=volatility_pct,
            momentum=momentum,
            trend_direction=trend_direction,
        )

        # Determine regime with highest score
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime]

        # Check for transitional state
        if self._current_regime != best_regime:
            if confidence < self.transition_threshold:
                best_regime = MarketRegime.TRANSITIONAL
                confidence = 1 - confidence
            else:
                # Confirm regime change
                self._regime_history.append(best_regime)
                if len(self._regime_history) >= self.regime_confirm_days:
                    recent = self._regime_history[-self.regime_confirm_days :]
                    if all(r == best_regime for r in recent):
                        previous = self._current_regime
                        self._current_regime = best_regime
                        self._days_in_regime = 0
                    else:
                        best_regime = MarketRegime.TRANSITIONAL
        else:
            self._days_in_regime += 1

        return RegimeSignal(
            regime=best_regime,
            confidence=confidence,
            trend_strength=adx,
            volatility_percentile=volatility_pct,
            momentum=momentum,
            days_in_regime=self._days_in_regime,
            previous_regime=self._current_regime
            if best_regime == MarketRegime.TRANSITIONAL
            else None,
        )

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Calculate Average Directional Index for trend strength."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=self.adx_period).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0

    def _calculate_volatility_percentile(self, data: pd.DataFrame) -> float:
        """Calculate current volatility percentile vs historical."""
        returns = data["close"].pct_change().dropna()

        # Current volatility (recent window)
        current_vol = returns.iloc[-self.volatility_lookback :].std() * np.sqrt(252)

        # Historical volatility distribution
        rolling_vol = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 20:
            return 50.0

        # Percentile rank
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100
        return percentile

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate RSI-based momentum indicator."""
        close = data["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50

    def _calculate_trend_direction(self, data: pd.DataFrame) -> float:
        """
        Calculate trend direction score.

        Returns:
            Score from -1 (strong bearish) to +1 (strong bullish)
        """
        close = data["close"]

        # Moving averages
        fast_ma = close.rolling(window=self.fast_ma_period).mean()
        slow_ma = close.rolling(window=self.slow_ma_period).mean()
        trend_ma = close.rolling(window=self.trend_ma_period).mean()

        current_price = close.iloc[-1]

        # Score components
        scores = []

        # Price vs trend MA
        if current_price > trend_ma.iloc[-1]:
            scores.append(0.3)
        else:
            scores.append(-0.3)

        # Fast vs slow MA
        if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
            scores.append(0.3)
        else:
            scores.append(-0.3)

        # MA slope (trend direction)
        trend_slope = (trend_ma.iloc[-1] - trend_ma.iloc[-20]) / trend_ma.iloc[-20]
        scores.append(np.clip(trend_slope * 10, -0.4, 0.4))

        return sum(scores)

    def _score_regimes(
        self,
        adx: float,
        volatility_pct: float,
        momentum: float,
        trend_direction: float,
    ) -> dict[MarketRegime, float]:
        """Score each regime based on current indicators."""
        scores = {
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.VOLATILE: 0.0,
        }

        # High volatility check (takes precedence)
        if volatility_pct > self.high_vol_percentile:
            scores[MarketRegime.VOLATILE] += 0.4
        else:
            scores[MarketRegime.VOLATILE] -= 0.2

        # Trend strength (ADX)
        if adx > self.adx_strong_threshold:
            # Strong trend
            if trend_direction > 0.3:
                scores[MarketRegime.BULL] += 0.5
            elif trend_direction < -0.3:
                scores[MarketRegime.BEAR] += 0.5
        elif adx > self.adx_trend_threshold:
            # Moderate trend
            if trend_direction > 0.2:
                scores[MarketRegime.BULL] += 0.3
            elif trend_direction < -0.2:
                scores[MarketRegime.BEAR] += 0.3
        else:
            # No clear trend
            scores[MarketRegime.SIDEWAYS] += 0.4

        # Momentum confirmation
        if momentum > 60:
            scores[MarketRegime.BULL] += 0.2
        elif momentum < 40:
            scores[MarketRegime.BEAR] += 0.2
        else:
            scores[MarketRegime.SIDEWAYS] += 0.2

        # Normalize scores
        total = sum(max(0, s) for s in scores.values())
        if total > 0:
            scores = {k: max(0, v) / total for k, v in scores.items()}

        return scores

    def get_regime_history(self, n: int = 20) -> list[MarketRegime]:
        """Get recent regime history."""
        return self._regime_history[-n:]

    def reset(self):
        """Reset detector state."""
        self._current_regime = MarketRegime.SIDEWAYS
        self._days_in_regime = 0
        self._regime_history = []


class RegimeAnalyzer:
    """
    Analyzes historical data to identify regime periods.

    Useful for:
    - Labeling training data
    - Backtesting regime detection accuracy
    - Understanding historical regime transitions
    """

    def __init__(self, detector: RegimeDetector | None = None):
        self.detector = detector or RegimeDetector()

    def analyze_period(
        self,
        data: pd.DataFrame,
        window_size: int = 20,
    ) -> pd.DataFrame:
        """
        Analyze entire period and label each day with regime.

        Returns DataFrame with regime labels and confidence scores.
        """
        results = []

        for i in range(len(data)):
            if i < self.detector.trend_ma_period:
                results.append(
                    {
                        "date": data.index[i],
                        "regime": MarketRegime.SIDEWAYS.value,
                        "confidence": 0.3,
                        "trend_strength": 0,
                        "volatility_pct": 50,
                        "momentum": 50,
                    }
                )
                continue

            window = data.iloc[: i + 1]
            signal = self.detector.detect(window)

            results.append(
                {
                    "date": data.index[i],
                    "regime": signal.regime.value,
                    "confidence": signal.confidence,
                    "trend_strength": signal.trend_strength,
                    "volatility_pct": signal.volatility_percentile,
                    "momentum": signal.momentum,
                }
            )

            self.detector.reset()

        return pd.DataFrame(results).set_index("date")

    def get_regime_periods(
        self,
        regime_labels: pd.DataFrame,
    ) -> list[dict]:
        """
        Extract continuous regime periods from labeled data.

        Returns list of period dicts with start, end, regime, duration.
        """
        periods = []
        current_regime = None
        period_start = None

        for date, row in regime_labels.iterrows():
            regime = row["regime"]

            if regime != current_regime:
                if current_regime is not None:
                    periods.append(
                        {
                            "regime": current_regime,
                            "start": period_start,
                            "end": date,
                            "duration_days": (date - period_start).days,
                        }
                    )

                current_regime = regime
                period_start = date

        # Add final period
        if current_regime is not None:
            periods.append(
                {
                    "regime": current_regime,
                    "start": period_start,
                    "end": regime_labels.index[-1],
                    "duration_days": (regime_labels.index[-1] - period_start).days,
                }
            )

        return periods
