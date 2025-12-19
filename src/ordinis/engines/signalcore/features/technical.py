"""
Technical indicator calculations for feature engineering.

All indicators are based on standard financial formulas.

NOTE: This module delegates core calculations to ordinis.quant.gs_quant_adapter
for consistency across the codebase. The wrapper methods here maintain backwards
compatibility with existing call sites.

TODO(refactor): Consider deprecating this class in favor of direct ordinis.quant imports.
See docs/gs_quant_integration_analysis.md for consolidation recommendations.
"""

import numpy as np
import pandas as pd

# Import canonical implementations from gs_quant_adapter
from ordinis.quant import (
    bollinger_bands as _gs_bollinger_bands,
)
from ordinis.quant import (
    macd as _gs_macd,
)
from ordinis.quant import (
    moving_average as _gs_moving_average,
)
from ordinis.quant import (
    rsi as _gs_rsi,
)
from ordinis.quant import (
    zscores as _gs_zscores,
)


class TechnicalIndicators:
    """
    Technical indicator calculations.

    NOTE: Core indicator functions (RSI, Bollinger Bands, MACD, etc.) now delegate
    to ordinis.quant for canonical implementations. This class provides backwards-
    compatible wrappers that preserve the original API signatures.
    """

    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            data: Price series
            window: Number of periods

        Returns:
            SMA series
        """
        # Delegate to gs_quant_adapter, but preserve NaN behavior for compatibility
        result = _gs_moving_average(data, w=window)
        # Reindex to match original series length (gs_quant trims NaN)
        return result.reindex(data.index)

    @staticmethod
    def ema(data: pd.Series, span: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            data: Price series
            span: Number of periods

        Returns:
            EMA series
        """
        return data.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            data: Price series
            window: Number of periods (default 14)

        Returns:
            RSI series (0-100)
        """
        # Delegate to gs_quant_adapter
        result = _gs_rsi(data, w=window)
        # Reindex to match original series length
        return result.reindex(data.index)

    @staticmethod
    def bollinger_bands(
        data: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            data: Price series
            window: Number of periods
            num_std: Number of standard deviations

        Returns:
            (middle_band, upper_band, lower_band) tuple
        """
        # Delegate to gs_quant_adapter (returns DataFrame)
        bb_df = _gs_bollinger_bands(data, w=window, k=num_std)

        # Convert to tuple format for backwards compatibility
        middle = bb_df["middle"].reindex(data.index)
        upper = bb_df["upper"].reindex(data.index)
        lower = bb_df["lower"].reindex(data.index)

        return middle, upper, lower

    @staticmethod
    def macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            (macd, signal, histogram) tuple
        """
        # Delegate to gs_quant_adapter (returns DataFrame)
        macd_df = _gs_macd(data, fast=fast, slow=slow, signal=signal)

        # Convert to tuple format for backwards compatibility
        macd_line = macd_df["macd"].reindex(data.index)
        signal_line = macd_df["signal"].reindex(data.index)
        histogram = macd_df["histogram"].reindex(data.index)

        return macd_line, signal_line, histogram

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Number of periods

        Returns:
            ATR series
        """
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def z_score(data: pd.Series, window: int = 20) -> pd.Series:
        """
        Rolling Z-Score.

        (Value - Mean) / StdDev

        Args:
            data: Price series
            window: Rolling window size

        Returns:
            Z-Score series
        """
        # Delegate to gs_quant_adapter
        result = _gs_zscores(data, w=window)
        # Reindex to match original series length
        return result.reindex(data.index)

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Number of periods

        Returns:
            (%K, %D) tuple
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()

        return k, d

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average Directional Index.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Number of periods

        Returns:
            ADX series
        """
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)  # type: ignore[operator]
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)  # type: ignore[operator]

        # True Range
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()

        return adx

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV series
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()

        return obv

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int | None = None,
    ) -> pd.Series:
        """
        Volume-Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            window: Rolling window size (optional). If None, calculates cumulative VWAP.

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3.0
        pv = typical_price * volume

        if window:
            cum_pv = pv.rolling(window=window).sum()
            cum_vol = volume.rolling(window=window).sum()
        else:
            cum_pv = pv.cumsum()
            cum_vol = volume.cumsum()

        return cum_pv / cum_vol
