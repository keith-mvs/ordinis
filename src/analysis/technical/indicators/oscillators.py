"""
Oscillator Indicators.

Bounded indicators that oscillate between fixed values,
used to identify overbought/oversold conditions and divergences.

Indicators:
- RSI: Relative Strength Index (0-100)
- Stochastic: %K and %D (0-100)
- CCI: Commodity Channel Index (unbounded, typically -100 to +100)
- Williams %R: Williams Percent Range (0 to -100)
- ROC: Rate of Change (momentum oscillator)
- Ultimate Oscillator: Multi-timeframe momentum
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class OscillatorCondition(Enum):
    """Market condition based on oscillator."""

    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    NEUTRAL = "neutral"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"


@dataclass
class OscillatorSignal:
    """Oscillator signal data."""

    indicator: str
    value: float
    condition: OscillatorCondition
    previous_value: float
    threshold_upper: float
    threshold_lower: float
    signal: str | None = None  # "buy", "sell", None


class Oscillators:
    """
    Oscillator calculations and signals.

    Provides momentum oscillators for overbought/oversold
    identification and divergence detection.
    """

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Measures speed and magnitude of price changes.
        RSI > 70: Overbought
        RSI < 30: Oversold

        Args:
            data: Price series (typically close)
            period: RSI period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        delta = data.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Compares closing price to price range over a period.
        %K > 80: Overbought
        %K < 20: Oversold

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K lookback period
            d_period: %D smoothing period
            smooth_k: %K smoothing (for slow stochastic)

        Returns:
            Tuple of (%K series, %D series)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # Smooth %K (slow stochastic)
        k = raw_k.rolling(window=smooth_k).mean()

        # %D is SMA of %K
        d = k.rolling(window=d_period).mean()

        return k, d

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.

        Measures current price level relative to average price.
        CCI > +100: Strong uptrend / overbought
        CCI < -100: Strong downtrend / oversold

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: CCI period

        Returns:
            Series of CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
        return cci

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Williams %R.

        Similar to stochastic but inverted scale (0 to -100).
        %R > -20: Overbought
        %R < -80: Oversold

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period

        Returns:
            Series of Williams %R values (0 to -100)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return r

    @staticmethod
    def roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change (momentum).

        Percentage change over N periods.

        Args:
            data: Price series
            period: ROC period

        Returns:
            Series of ROC values (percentage)
        """
        return (data - data.shift(period)) / data.shift(period) * 100

    @staticmethod
    def ultimate_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pd.Series:
        """
        Ultimate Oscillator.

        Multi-timeframe momentum that reduces false signals.
        UO > 70: Overbought
        UO < 30: Oversold

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period1: Short period
            period2: Medium period
            period3: Long period

        Returns:
            Series of Ultimate Oscillator values (0-100)
        """
        prev_close = close.shift(1)

        # Buying Pressure
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)

        # True Range
        tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(
            axis=1
        )

        # Averages for each period
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()

        # Weighted average
        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return uo

    @staticmethod
    def money_flow_index(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index (Volume-weighted RSI).

        Incorporates volume into momentum calculation.
        MFI > 80: Overbought
        MFI < 20: Oversold

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: MFI period

        Returns:
            Series of MFI values (0-100)
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        delta = typical_price.diff()
        positive_flow = raw_money_flow.where(delta > 0, 0)
        negative_flow = raw_money_flow.where(delta < 0, 0)

        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))

        return mfi

    @classmethod
    def rsi_signal(
        cls, data: pd.DataFrame, period: int = 14, overbought: float = 70, oversold: float = 30
    ) -> OscillatorSignal:
        """
        Generate RSI signal with overbought/oversold detection.

        Args:
            data: DataFrame with 'close' column
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold

        Returns:
            OscillatorSignal with condition and signal
        """
        rsi = cls.rsi(data["close"], period)
        current = rsi.iloc[-1]
        previous = rsi.iloc[-2] if len(rsi) > 1 else current

        # Determine condition
        if current > overbought:
            condition = OscillatorCondition.OVERBOUGHT
            signal = "sell" if previous <= overbought else None
        elif current < oversold:
            condition = OscillatorCondition.OVERSOLD
            signal = "buy" if previous >= oversold else None
        else:
            condition = OscillatorCondition.NEUTRAL
            signal = None

        return OscillatorSignal(
            indicator="RSI",
            value=current,
            condition=condition,
            previous_value=previous,
            threshold_upper=overbought,
            threshold_lower=oversold,
            signal=signal,
        )

    @classmethod
    def stochastic_signal(
        cls,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        overbought: float = 80,
        oversold: float = 20,
    ) -> OscillatorSignal:
        """
        Generate Stochastic signal with crossovers.

        Args:
            data: DataFrame with high, low, close columns
            k_period: %K period
            d_period: %D period
            overbought: Overbought threshold
            oversold: Oversold threshold

        Returns:
            OscillatorSignal with condition and signal
        """
        k, d = cls.stochastic(
            data["high"], data["low"], data["close"], k_period=k_period, d_period=d_period
        )

        k_current = k.iloc[-1]
        k_previous = k.iloc[-2] if len(k) > 1 else k_current
        d_current = d.iloc[-1]
        d_previous = d.iloc[-2] if len(d) > 1 else d_current

        # Condition
        if k_current > overbought:
            condition = OscillatorCondition.OVERBOUGHT
        elif k_current < oversold:
            condition = OscillatorCondition.OVERSOLD
        else:
            condition = OscillatorCondition.NEUTRAL

        # Signal from %K/%D crossover
        signal = None
        if k_previous <= d_previous and k_current > d_current:
            signal = "buy" if condition == OscillatorCondition.OVERSOLD else None
        elif k_previous >= d_previous and k_current < d_current:
            signal = "sell" if condition == OscillatorCondition.OVERBOUGHT else None

        return OscillatorSignal(
            indicator="Stochastic",
            value=k_current,
            condition=condition,
            previous_value=k_previous,
            threshold_upper=overbought,
            threshold_lower=oversold,
            signal=signal,
        )

    @classmethod
    def detect_divergence(
        cls, price: pd.Series, oscillator: pd.Series, lookback: int = 20
    ) -> str | None:
        """
        Detect bullish or bearish divergence.

        Bullish: Price makes lower low, oscillator makes higher low
        Bearish: Price makes higher high, oscillator makes lower high

        Args:
            price: Price series
            oscillator: Oscillator series (RSI, etc.)
            lookback: Number of bars to check

        Returns:
            "bullish", "bearish", or None
        """
        if len(price) < lookback or len(oscillator) < lookback:
            return None

        price_window = price.iloc[-lookback:]
        osc_window = oscillator.iloc[-lookback:]

        # Find local extremes
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        osc_at_price_min = oscillator.loc[price_min_idx]
        osc_at_price_max = oscillator.loc[price_max_idx]

        # Current values
        price_current = price.iloc[-1]
        osc_current = oscillator.iloc[-1]

        # Bullish divergence: price lower low, oscillator higher low
        if price_current <= price_window.min() and osc_current > osc_at_price_min:
            return "bullish"

        # Bearish divergence: price higher high, oscillator lower high
        if price_current >= price_window.max() and osc_current < osc_at_price_max:
            return "bearish"

        return None
