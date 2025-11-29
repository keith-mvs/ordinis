"""
Technical indicator calculations for feature engineering.

All indicators are based on standard financial formulas.
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Technical indicator calculations."""

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
        return data.rolling(window=window).mean()

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
        delta = data.diff()

        gain = delta.where(delta > 0, 0.0)  # type: ignore[operator]
        loss = -delta.where(delta < 0, 0.0)  # type: ignore[operator]

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

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
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

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
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

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
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume-Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3.0
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap
