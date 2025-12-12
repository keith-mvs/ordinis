"""
Volume Indicators.

Analyze trading volume to confirm price movements and identify
accumulation/distribution patterns.

Indicators:
- OBV: On-Balance Volume
- Volume Profile: Volume distribution by price
- A/D Line: Accumulation/Distribution Line
- CMF: Chaikin Money Flow
- VWAP: Volume Weighted Average Price
- Force Index: Price change * volume
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeSignal:
    """Volume analysis signal."""

    indicator: str
    value: float
    trend: str  # "accumulation", "distribution", "neutral"
    divergence: str | None  # "bullish", "bearish", None
    confirmation: bool  # Volume confirms price move


class VolumeIndicators:
    """
    Volume indicator calculations and analysis.

    Provides tools for volume-based trend confirmation.
    """

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.

        Cumulative volume based on price direction.
        Rising OBV = accumulation
        Falling OBV = distribution

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            Series of OBV values
        """
        price_change = close.diff()
        volume_direction = np.where(
            price_change > 0, volume, np.where(price_change < 0, -volume, 0)
        )
        return pd.Series(volume_direction, index=volume.index).cumsum()

    @staticmethod
    def accumulation_distribution(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Accumulation/Distribution Line.

        Measures money flow based on where close falls within range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data

        Returns:
            Series of A/D Line values
        """
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # Money Flow Volume
        mfv = mfm * volume

        # A/D Line (cumulative)
        return mfv.cumsum()

    @staticmethod
    def chaikin_money_flow(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Chaikin Money Flow.

        Measures buying/selling pressure over a period.
        CMF > 0: Buying pressure
        CMF < 0: Selling pressure

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: CMF period

        Returns:
            Series of CMF values (-1 to +1)
        """
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume

        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        return cmf

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        reset_daily: bool = False,
    ) -> pd.Series:
        """
        Volume Weighted Average Price.

        Average price weighted by volume.
        Price above VWAP = bullish
        Price below VWAP = bearish

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            reset_daily: Reset cumsum daily (intraday)

        Returns:
            Series of VWAP values
        """
        typical_price = (high + low + close) / 3

        if reset_daily:
            # Group by date for daily reset
            dates = typical_price.index.date
            groups = pd.Series(dates, index=typical_price.index)

            def calc_vwap(group):
                tp_vol = (typical_price.loc[group.index] * volume.loc[group.index]).cumsum()
                vol_cum = volume.loc[group.index].cumsum()
                return tp_vol / vol_cum

            return typical_price.groupby(groups).apply(calc_vwap).droplevel(0)
        tp_vol = (typical_price * volume).cumsum()
        vol_cum = volume.cumsum()
        return tp_vol / vol_cum

    @staticmethod
    def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """
        Force Index.

        Measures the "force" behind price movements.
        Force = Price Change * Volume

        Args:
            close: Close prices
            volume: Volume data
            period: EMA smoothing period

        Returns:
            Series of Force Index values
        """
        raw_force = close.diff() * volume
        return raw_force.ewm(span=period, adjust=False).mean()

    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Price Trend (VPT).

        Cumulative volume adjusted by percentage price change.

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            Series of VPT values
        """
        pct_change = close.pct_change()
        vpt = (volume * pct_change).cumsum()
        return vpt

    @staticmethod
    def negative_volume_index(
        close: pd.Series, volume: pd.Series, initial_value: float = 1000
    ) -> pd.Series:
        """
        Negative Volume Index (NVI).

        Changes only on days when volume decreases.
        Theory: "Smart money" trades on low volume days.

        Args:
            close: Close prices
            volume: Volume data
            initial_value: Starting NVI value

        Returns:
            Series of NVI values
        """
        nvi = pd.Series(index=close.index, dtype=float)
        nvi.iloc[0] = initial_value

        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i - 1]:
                # Volume decreased - update NVI
                pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct_change)
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]

        return nvi

    @staticmethod
    def positive_volume_index(
        close: pd.Series, volume: pd.Series, initial_value: float = 1000
    ) -> pd.Series:
        """
        Positive Volume Index (PVI).

        Changes only on days when volume increases.

        Args:
            close: Close prices
            volume: Volume data
            initial_value: Starting PVI value

        Returns:
            Series of PVI values
        """
        pvi = pd.Series(index=close.index, dtype=float)
        pvi.iloc[0] = initial_value

        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i - 1]:
                pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct_change)
            else:
                pvi.iloc[i] = pvi.iloc[i - 1]

        return pvi

    @staticmethod
    def volume_relative(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Relative Volume (RVOL).

        Current volume relative to average volume.
        RVOL > 1.5 = high volume
        RVOL < 0.5 = low volume

        Args:
            volume: Volume data
            period: Average volume period

        Returns:
            Series of relative volume values
        """
        avg_volume = volume.rolling(period).mean()
        return volume / avg_volume

    @classmethod
    def volume_confirmation(
        cls, data: pd.DataFrame, price_change_threshold: float = 0.01, volume_threshold: float = 1.5
    ) -> VolumeSignal:
        """
        Analyze if volume confirms price movement.

        Args:
            data: DataFrame with close and volume columns
            price_change_threshold: Minimum price change to consider
            volume_threshold: RVOL threshold for high volume

        Returns:
            VolumeSignal with confirmation analysis
        """
        close = data["close"]
        volume = data["volume"]

        # Calculate indicators
        obv = cls.obv(close, volume)
        rvol = cls.volume_relative(volume)

        # Current values
        price_change = close.pct_change().iloc[-1]
        current_rvol = rvol.iloc[-1]
        obv_change = obv.diff().iloc[-1]

        # Trend determination
        if obv.iloc[-1] > obv.iloc[-5]:
            trend = "accumulation"
        elif obv.iloc[-1] < obv.iloc[-5]:
            trend = "distribution"
        else:
            trend = "neutral"

        # Divergence check
        divergence = None
        if len(close) > 10:
            price_higher = close.iloc[-1] > close.iloc[-10]
            obv_higher = obv.iloc[-1] > obv.iloc[-10]

            if price_higher and not obv_higher:
                divergence = "bearish"
            elif not price_higher and obv_higher:
                divergence = "bullish"

        # Confirmation check
        confirmation = False
        if abs(price_change) > price_change_threshold:
            # Price moved significantly
            if current_rvol > volume_threshold:
                # High volume
                if (price_change > 0 and obv_change > 0) or (price_change < 0 and obv_change < 0):
                    confirmation = True

        return VolumeSignal(
            indicator="Volume Analysis",
            value=current_rvol,
            trend=trend,
            divergence=divergence,
            confirmation=confirmation,
        )
