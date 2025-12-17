"""
Moving Average Indicators.

Smooth out price data to identify trend direction and potential
support/resistance levels.

Types:
- SMA: Simple Moving Average - equal weight to all periods
- EMA: Exponential Moving Average - more weight to recent prices
- WMA: Weighted Moving Average - linear weight decay
- VWAP: Volume Weighted Average Price - volume-weighted average
- DEMA: Double EMA - reduced lag
- TEMA: Triple EMA - further reduced lag
- Hull MA: Faster and smoother combination
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MASignal:
    """Moving average signal data."""

    current_price: float
    ma_value: float
    ma_type: str
    period: int
    price_vs_ma: float  # Percentage above/below
    slope: float  # MA slope (trend direction)
    crossover: str | None = None  # "golden", "death", None


class MovingAverages:
    """
    Moving Average calculations and signals.

    Provides various moving average types for trend identification
    and crossover signal generation.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Equal weight to all periods in the lookback window.

        Args:
            data: Price series (typically close prices)
            period: Number of periods for averaging

        Returns:
            Series of SMA values
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """
        Exponential Moving Average.

        More weight to recent prices, reacts faster to price changes.

        Args:
            data: Price series
            period: Span for EMA calculation
            adjust: Whether to adjust for initial values

        Returns:
            Series of EMA values
        """
        return data.ewm(span=period, adjust=adjust).mean()

    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """
        Weighted Moving Average.

        Linear weight decay - most recent price has highest weight.

        Args:
            data: Price series
            period: Number of periods

        Returns:
            Series of WMA values
        """
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price.

        Average price weighted by volume - shows "fair value" for the day.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data

        Returns:
            Series of VWAP values
        """
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def dema(data: pd.Series, period: int) -> pd.Series:
        """
        Double Exponential Moving Average.

        Reduces lag compared to standard EMA.
        DEMA = 2 * EMA - EMA(EMA)

        Args:
            data: Price series
            period: EMA period

        Returns:
            Series of DEMA values
        """
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(data: pd.Series, period: int) -> pd.Series:
        """
        Triple Exponential Moving Average.

        Further lag reduction compared to DEMA.
        TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

        Args:
            data: Price series
            period: EMA period

        Returns:
            Series of TEMA values
        """
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def hull_ma(data: pd.Series, period: int) -> pd.Series:
        """
        Hull Moving Average.

        Extremely fast and smooth MA that reduces lag significantly.
        HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

        Args:
            data: Price series
            period: Hull MA period

        Returns:
            Series of Hull MA values
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        wma_half = MovingAverages.wma(data, half_period)
        wma_full = MovingAverages.wma(data, period)
        raw_hma = 2 * wma_half - wma_full

        return MovingAverages.wma(raw_hma, sqrt_period)

    @staticmethod
    def adaptive_ma(data: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """
        Kaufman Adaptive Moving Average (KAMA).

        Adapts to market volatility - faster in trends, slower in ranges.

        Args:
            data: Price series
            period: Efficiency ratio period
            fast: Fast EMA constant
            slow: Slow EMA constant

        Returns:
            Series of KAMA values
        """
        # Efficiency Ratio
        change = abs(data - data.shift(period))
        volatility = abs(data.diff()).rolling(window=period).sum()
        er = change / (volatility + 1e-10)

        # Smoothing constant
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA calculation
        kama = pd.Series(index=data.index, dtype=float)
        kama.iloc[period] = data.iloc[period]

        for i in range(period + 1, len(data)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i - 1])

        return kama

    @classmethod
    def crossover_signal(
        cls, data: pd.DataFrame, fast_period: int = 20, slow_period: int = 50, ma_type: str = "ema"
    ) -> MASignal:
        """
        Generate crossover signal from dual moving averages.

        Args:
            data: DataFrame with 'close' column
            fast_period: Fast MA period
            slow_period: Slow MA period
            ma_type: Type of MA ("sma", "ema", "wma", "hull")

        Returns:
            MASignal with crossover information
        """
        close = data["close"]

        # Calculate MAs based on type
        ma_func = {
            "sma": cls.sma,
            "ema": cls.ema,
            "wma": cls.wma,
            "hull": cls.hull_ma,
        }.get(ma_type, cls.ema)

        fast_ma = ma_func(close, fast_period)
        slow_ma = ma_func(close, slow_period)

        current_price = close.iloc[-1]
        fast_current = fast_ma.iloc[-1]
        slow_current = slow_ma.iloc[-1]

        # Price vs MA
        price_vs_ma = (current_price - slow_current) / slow_current * 100

        # MA slope (5-period rate of change)
        slope = (
            (slow_ma.iloc[-1] - slow_ma.iloc[-5]) / slow_ma.iloc[-5] * 100
            if len(slow_ma) > 5
            else 0
        )

        # Crossover detection
        crossover = None
        if len(fast_ma) > 1 and len(slow_ma) > 1:
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]

            if prev_fast <= prev_slow and fast_current > slow_current:
                crossover = "golden"
            elif prev_fast >= prev_slow and fast_current < slow_current:
                crossover = "death"

        return MASignal(
            current_price=current_price,
            ma_value=slow_current,
            ma_type=ma_type,
            period=slow_period,
            price_vs_ma=price_vs_ma,
            slope=slope,
            crossover=crossover,
        )

    @classmethod
    def multi_ma_analysis(cls, data: pd.DataFrame, periods: list[int] | None = None) -> dict:
        """
        Analyze price relative to multiple MAs.

        Args:
            data: DataFrame with 'close' column
            periods: List of MA periods (default: [10, 20, 50, 100, 200])

        Returns:
            Dictionary with MA values and trend analysis
        """
        periods = periods or [10, 20, 50, 100, 200]
        close = data["close"]
        current_price = close.iloc[-1]

        result = {
            "current_price": current_price,
            "mas": {},
            "above_count": 0,
            "trend_strength": 0,
            "alignment": "mixed",
        }

        ma_values = []
        for period in periods:
            if len(close) >= period:
                ma = cls.ema(close, period).iloc[-1]
                result["mas"][f"ema_{period}"] = ma
                ma_values.append(ma)
                if current_price > ma:
                    result["above_count"] += 1

        # Trend strength (0-100)
        result["trend_strength"] = result["above_count"] / len(periods) * 100

        # MA alignment check
        if len(ma_values) >= 3:
            sorted_mas = sorted(ma_values, reverse=True)
            if ma_values == sorted_mas:
                result["alignment"] = "bullish"  # All MAs in bullish order
            elif ma_values == sorted_mas[::-1]:
                result["alignment"] = "bearish"  # All MAs in bearish order

        return result
