"""
Volatility Indicators.

Measure price dispersion, risk, and identify potential breakout/reversal zones.

Indicators:
- ATR: Average True Range (absolute volatility)
- Bollinger Bands: Standard deviation bands around SMA
- Keltner Channels: ATR-based bands around EMA
- Donchian Channels: High/low breakout channels
- Standard Deviation: Historical volatility
- Chaikin Volatility: Rate of change of ATR
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BandSignal:
    """Bollinger/Keltner band signal."""

    upper_band: float
    middle_band: float
    lower_band: float
    current_price: float
    bandwidth: float  # Band width as percentage
    position: str  # "above_upper", "below_lower", "within"
    squeeze: bool  # True if bands are contracting (low volatility)


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics."""

    atr: float
    atr_percent: float  # ATR as % of price
    historical_vol: float  # Annualized standard deviation
    vol_percentile: float  # Current vol vs historical (0-100)
    vol_regime: str  # "low", "normal", "high", "extreme"


class VolatilityIndicators:
    """
    Volatility indicator calculations and analysis.

    Provides tools for measuring and analyzing price volatility.
    """

    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate True Range.

        TR = max(high-low, |high-prev_close|, |low-prev_close|)

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Series of True Range values
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @classmethod
    def atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range.

        Measures volatility as the average of true ranges.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            Series of ATR values
        """
        tr = cls.true_range(high, low, close)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Price bands at N standard deviations from SMA.
        - Upper Band: SMA + (std_dev * std)
        - Lower Band: SMA - (std_dev * std)

        Args:
            data: Price series (typically close)
            period: SMA period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    @classmethod
    def keltner_channels(
        cls,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_multiplier: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.

        ATR-based bands around EMA.
        - Upper Channel: EMA + (multiplier * ATR)
        - Lower Channel: EMA - (multiplier * ATR)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            ema_period: EMA period for middle line
            atr_period: ATR period
            atr_multiplier: ATR multiplier for bands

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        middle = close.ewm(span=ema_period, adjust=False).mean()
        atr = cls.atr(high, low, close, atr_period)

        upper = middle + (atr_multiplier * atr)
        lower = middle - (atr_multiplier * atr)

        return upper, middle, lower

    @staticmethod
    def donchian_channels(
        high: pd.Series, low: pd.Series, period: int = 20
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels (Price Channels).

        Highest high and lowest low over N periods.
        Used for breakout trading.

        Args:
            high: High prices
            low: Low prices
            period: Channel period

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    @staticmethod
    def historical_volatility(
        data: pd.Series, period: int = 20, annualize: bool = True, trading_days: int = 252
    ) -> pd.Series:
        """
        Historical Volatility (Standard Deviation of Returns).

        Measures how much returns vary from their average.

        Args:
            data: Price series
            period: Lookback period
            annualize: Whether to annualize (multiply by sqrt(252))
            trading_days: Trading days per year

        Returns:
            Series of volatility values
        """
        returns = data.pct_change()
        vol = returns.rolling(window=period).std()

        if annualize:
            vol = vol * np.sqrt(trading_days)

        return vol

    @staticmethod
    def chaikin_volatility(
        high: pd.Series, low: pd.Series, ema_period: int = 10, roc_period: int = 10
    ) -> pd.Series:
        """
        Chaikin Volatility.

        Rate of change of the trading range EMA.
        Rising: Increasing volatility
        Falling: Decreasing volatility

        Args:
            high: High prices
            low: Low prices
            ema_period: EMA period for range
            roc_period: Rate of change period

        Returns:
            Series of Chaikin Volatility values
        """
        high_low = high - low
        hl_ema = high_low.ewm(span=ema_period, adjust=False).mean()

        cv = (hl_ema - hl_ema.shift(roc_period)) / hl_ema.shift(roc_period) * 100

        return cv

    @classmethod
    def bollinger_signal(
        cls, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> BandSignal:
        """
        Generate Bollinger Band signal.

        Args:
            data: DataFrame with 'close' column
            period: BB period
            std_dev: Standard deviation multiplier

        Returns:
            BandSignal with band analysis
        """
        close = data["close"]
        upper, middle, lower = cls.bollinger_bands(close, period, std_dev)

        current_price = close.iloc[-1]
        upper_val = upper.iloc[-1]
        middle_val = middle.iloc[-1]
        lower_val = lower.iloc[-1]

        # Bandwidth (volatility indicator)
        bandwidth = (upper_val - lower_val) / middle_val * 100

        # Position relative to bands
        if current_price > upper_val:
            position = "above_upper"
        elif current_price < lower_val:
            position = "below_lower"
        else:
            position = "within"

        # Squeeze detection (low volatility)
        recent_bw = ((upper - lower) / middle * 100).iloc[-20:]
        squeeze = bandwidth < recent_bw.quantile(0.25)

        return BandSignal(
            upper_band=upper_val,
            middle_band=middle_val,
            lower_band=lower_val,
            current_price=current_price,
            bandwidth=bandwidth,
            position=position,
            squeeze=squeeze,
        )

    @classmethod
    def volatility_analysis(
        cls,
        data: pd.DataFrame,
        atr_period: int = 14,
        vol_period: int = 20,
        history_period: int = 252,
    ) -> VolatilityMetrics:
        """
        Comprehensive volatility analysis.

        Args:
            data: DataFrame with OHLC columns
            atr_period: ATR calculation period
            vol_period: Historical volatility period
            history_period: Period for volatility percentile

        Returns:
            VolatilityMetrics with comprehensive analysis
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # ATR
        atr = cls.atr(high, low, close, atr_period)
        atr_current = atr.iloc[-1]
        atr_percent = atr_current / close.iloc[-1] * 100

        # Historical volatility
        hist_vol = cls.historical_volatility(close, vol_period)
        vol_current = hist_vol.iloc[-1]

        # Volatility percentile
        if len(hist_vol) >= history_period:
            vol_history = hist_vol.iloc[-history_period:]
            vol_percentile = (vol_history < vol_current).sum() / len(vol_history) * 100
        else:
            vol_percentile = 50.0

        # Volatility regime
        if vol_percentile < 20:
            regime = "low"
        elif vol_percentile < 50:
            regime = "normal"
        elif vol_percentile < 80:
            regime = "high"
        else:
            regime = "extreme"

        return VolatilityMetrics(
            atr=atr_current,
            atr_percent=atr_percent,
            historical_vol=vol_current,
            vol_percentile=vol_percentile,
            vol_regime=regime,
        )

    @classmethod
    def calculate_stop_loss(
        cls, data: pd.DataFrame, method: str = "atr", multiplier: float = 2.0, period: int = 14
    ) -> float:
        """
        Calculate volatility-based stop loss level.

        Args:
            data: DataFrame with OHLC columns
            method: "atr" or "bollinger"
            multiplier: Multiplier for stop distance
            period: Indicator period

        Returns:
            Stop loss price level
        """
        close = data["close"]
        current_price = close.iloc[-1]

        if method == "atr":
            atr = cls.atr(data["high"], data["low"], close, period)
            stop_distance = multiplier * atr.iloc[-1]
        elif method == "bollinger":
            _, middle, lower = cls.bollinger_bands(close, period)
            stop_distance = middle.iloc[-1] - lower.iloc[-1]
        else:
            raise ValueError(f"Unknown method: {method}")

        return current_price - stop_distance
