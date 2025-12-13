"""
Trend Indicators.

Identify and measure trend direction, strength, and potential reversals.

Indicators:
- ADX: Average Directional Index (trend strength)
- Parabolic SAR: Stop and Reverse (dynamic support/resistance)
- Ichimoku Cloud: Multi-line trend and momentum framework
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ADXSignal:
    """ADX trend strength signal."""

    adx: float  # Trend strength (0-100)
    plus_di: float  # Positive directional indicator
    minus_di: float  # Negative directional indicator
    trend_strength: str  # "weak", "moderate", "strong", "very_strong"
    trend_direction: str  # "bullish", "bearish", "neutral"


@dataclass
class ParabolicSARSignal:
    """Parabolic SAR signal."""

    sar: float  # SAR value
    trend: str  # "bullish" (SAR below price) or "bearish" (SAR above price)
    reversal: bool  # True if trend just reversed
    acceleration: float  # Current acceleration factor


@dataclass
class IchimokuCloudValues:
    """Raw Ichimoku line values at the latest bar."""

    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float


@dataclass
class IchimokuSignal:
    """Ichimoku Cloud signal summary."""

    trend: str  # "bullish", "bearish", "neutral"
    position: str  # "above_cloud", "in_cloud", "below_cloud"
    cloud_bias: str  # "bullish", "bearish"
    baseline_cross: str | None  # "bullish", "bearish", or None
    lagging_confirmation: bool
    values: IchimokuCloudValues


class TrendIndicators:
    """
    Trend indicator calculations and analysis.

    Provides tools for identifying and measuring market trends.
    """

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength on a 0-100 scale:
        - 0-25: Weak or absent trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default 14)

        Returns:
            (ADX, +DI, -DI) tuple

        Reference:
            Wilder, J. Welles (1978). New Concepts in Technical Trading Systems
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM (Directional Movement)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        # +DM when up move is greater than down move and positive
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        # -DM when down move is greater than up move and positive
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smooth DM and TR using Wilder's smoothing (EMA-like)
        # First, use simple sum for initial value
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()
        tr_smooth = tr.rolling(window=period).sum()

        # Then apply Wilder's smoothing for subsequent values
        for i in range(period, len(plus_dm)):
            plus_dm_smooth.iloc[i] = (
                plus_dm_smooth.iloc[i - 1] - (plus_dm_smooth.iloc[i - 1] / period) + plus_dm.iloc[i]
            )
            minus_dm_smooth.iloc[i] = (
                minus_dm_smooth.iloc[i - 1]
                - (minus_dm_smooth.iloc[i - 1] / period)
                + minus_dm.iloc[i]
            )
            tr_smooth.iloc[i] = (
                tr_smooth.iloc[i - 1] - (tr_smooth.iloc[i - 1] / period) + tr.iloc[i]
            )

        # Calculate +DI and -DI (Directional Indicators)
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)  # Handle division by zero

        # Calculate ADX (Average Directional Index)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def adx_signal(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> ADXSignal:
        """
        Generate ADX trend strength signal.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default 14)

        Returns:
            ADXSignal with trend strength and direction
        """
        adx, plus_di, minus_di = TrendIndicators.adx(high, low, close, period)

        current_adx = float(adx.iloc[-1])
        current_plus_di = float(plus_di.iloc[-1])
        current_minus_di = float(minus_di.iloc[-1])

        # Determine trend strength
        if current_adx < 20:
            strength = "weak"
        elif current_adx < 40:
            strength = "moderate"
        elif current_adx < 60:
            strength = "strong"
        else:
            strength = "very_strong"

        # Determine trend direction
        if current_plus_di > current_minus_di:
            direction = "bullish"
        elif current_minus_di > current_plus_di:
            direction = "bearish"
        else:
            direction = "neutral"

        return ADXSignal(
            adx=current_adx,
            plus_di=current_plus_di,
            minus_di=current_minus_di,
            trend_strength=strength,
            trend_direction=direction,
        )

    @staticmethod
    def parabolic_sar(
        high: pd.Series,
        low: pd.Series,
        _close: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).

        Provides dynamic support/resistance levels that follow price trends.
        When price crosses SAR, the trend reverses.

        Args:
            high: High prices
            low: Low prices
            _close: Close prices (not used in SAR calculation)
            acceleration: Acceleration factor increment (default 0.02)
            maximum: Maximum acceleration factor (default 0.2)

        Returns:
            Series of SAR values

        Reference:
            Wilder, J. Welles (1978). New Concepts in Technical Trading Systems

        Usage:
            - SAR below price: Bullish trend, SAR acts as support
            - SAR above price: Bearish trend, SAR acts as resistance
            - Price crosses SAR: Trend reversal signal
        """
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)  # 1 for up, -1 for down
        af = acceleration
        ep = 0.0  # Extreme point

        # Initialize with first bar
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # Assume initial uptrend
        ep = high.iloc[0]

        for i in range(1, len(high)):
            prev_sar = sar.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]

            # Calculate SAR
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)

            # Check for reversal
            if prev_trend == 1:  # Uptrend
                # Ensure SAR is not above prior lows
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i - 1])
                if i > 1:
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i - 2])

                if low.iloc[i] < sar.iloc[i]:
                    # Reversal to downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep  # SAR becomes the extreme point
                    ep = low.iloc[i]  # New EP is current low
                    af = acceleration  # Reset AF
                else:
                    # Continue uptrend
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]  # Update extreme point
                        af = min(af + acceleration, maximum)  # Increase AF
            else:  # Downtrend
                # Ensure SAR is not below prior highs
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i - 1])
                if i > 1:
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i - 2])

                if high.iloc[i] > sar.iloc[i]:
                    # Reversal to uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep  # SAR becomes the extreme point
                    ep = high.iloc[i]  # New EP is current high
                    af = acceleration  # Reset AF
                else:
                    # Continue downtrend
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]  # Update extreme point
                        af = min(af + acceleration, maximum)  # Increase AF

        return sar

    @staticmethod
    def parabolic_sar_signal(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> ParabolicSARSignal:
        """
        Generate Parabolic SAR signal.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            acceleration: Acceleration factor increment (default 0.02)
            maximum: Maximum acceleration factor (default 0.2)

        Returns:
            ParabolicSARSignal with trend and reversal info
        """
        sar = TrendIndicators.parabolic_sar(high, low, close, acceleration, maximum)

        current_sar = float(sar.iloc[-1])
        current_price = float(close.iloc[-1])
        prev_sar = float(sar.iloc[-2]) if len(sar) > 1 else current_sar

        # Determine trend
        if current_price > current_sar:
            trend = "bullish"
        else:
            trend = "bearish"

        # Check for reversal (SAR crossed)
        prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
        prev_trend = "bullish" if prev_price > prev_sar else "bearish"
        reversal = trend != prev_trend

        return ParabolicSARSignal(
            sar=current_sar,
            trend=trend,
            reversal=reversal,
            acceleration=acceleration,  # Simplified - actual AF varies
        )

    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26,
    ) -> tuple[IchimokuCloudValues, IchimokuSignal]:
        """
        Calculate Ichimoku Cloud lines and derive a trend signal.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            conversion_period: Lookback for Tenkan-sen (default 9)
            base_period: Lookback for Kijun-sen (default 26)
            span_b_period: Lookback for Senkou Span B (default 52)
            displacement: Forward shift for leading spans and backward for Chikou (default 26)

        Returns:
            (IchimokuCloudValues, IchimokuSignal)
        """
        conversion_line = (
            high.rolling(window=conversion_period).max()
            + low.rolling(window=conversion_period).min()
        ) / 2
        base_line = (
            high.rolling(window=base_period).max() + low.rolling(window=base_period).min()
        ) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        leading_span_b = (
            (high.rolling(window=span_b_period).max() + low.rolling(window=span_b_period).min()) / 2
        ).shift(displacement)
        lagging_span = close.shift(-displacement)
        chikou_series = lagging_span.dropna()
        chikou_value = float(chikou_series.iloc[-1]) if not chikou_series.empty else float("nan")

        values = IchimokuCloudValues(
            tenkan_sen=float(conversion_line.iloc[-1]),
            kijun_sen=float(base_line.iloc[-1]),
            senkou_span_a=float(leading_span_a.iloc[-1]),
            senkou_span_b=float(leading_span_b.iloc[-1]),
            chikou_span=chikou_value,
        )

        price = float(close.iloc[-1])
        cloud_top = max(values.senkou_span_a, values.senkou_span_b)
        cloud_bottom = min(values.senkou_span_a, values.senkou_span_b)

        if price > cloud_top:
            position = "above_cloud"
        elif price < cloud_bottom:
            position = "below_cloud"
        else:
            position = "in_cloud"

        cloud_bias = "bullish" if values.senkou_span_a >= values.senkou_span_b else "bearish"

        # Detect price crossing baseline (Kijun)
        baseline_cross: str | None = None
        if len(close) > 1 and len(base_line.dropna()) >= 2:
            prev_price = float(close.iloc[-2])
            prev_base = float(base_line.iloc[-2])
            if prev_price <= prev_base < price:
                baseline_cross = "bullish"
            elif prev_price >= prev_base > price:
                baseline_cross = "bearish"

        # Lagging span confirmation: Chikou above/below price at displacement offset
        lag_confirm = False
        if len(close) > displacement and not pd.isna(values.chikou_span):
            lookback_price = float(close.iloc[-displacement - 1])
            if (
                values.chikou_span > lookback_price
                and cloud_bias == "bullish"
                or values.chikou_span < lookback_price
                and cloud_bias == "bearish"
            ):
                lag_confirm = True

        # Overall trend assessment
        if (
            position == "above_cloud"
            and cloud_bias == "bullish"
            and values.tenkan_sen > values.kijun_sen
        ):
            trend = "bullish"
        elif (
            position == "below_cloud"
            and cloud_bias == "bearish"
            and values.tenkan_sen < values.kijun_sen
        ):
            trend = "bearish"
        else:
            trend = "neutral"

        signal = IchimokuSignal(
            trend=trend,
            position=position,
            cloud_bias=cloud_bias,
            baseline_cross=baseline_cross,
            lagging_confirmation=lag_confirm,
            values=values,
        )

        return values, signal
