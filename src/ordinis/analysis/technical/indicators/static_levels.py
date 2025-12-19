"""
Static Levels.

Calculate mathematical support and resistance levels that don't change with each bar.

Indicators:
- Fibonacci Retracement: Key levels during pullbacks (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Fibonacci Extension: Target levels beyond swing high/low (161.8%, 261.8%, 423.6%)
- Pivot Points: Classic, Woodie's, Camarilla, Fibonacci pivots
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class FibonacciLevels:
    """Fibonacci retracement or extension levels."""

    swing_high: float
    swing_low: float
    current_price: float
    levels: dict[str, float]  # Level name -> price
    direction: str  # "retracement" or "extension"
    nearest_level: str  # Name of nearest level
    nearest_distance: float  # Distance to nearest level (%)


@dataclass
class PivotLevels:
    """Pivot point levels."""

    pivot: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    support_1: float
    support_2: float
    support_3: float
    pivot_type: str  # "classic", "woodie", "camarilla", "fibonacci"


class StaticLevels:
    """
    Static level calculations for support and resistance.

    These levels are calculated from historical price data and remain
    constant until recalculated with new swing highs/lows.
    """

    @staticmethod
    def fibonacci_retracement(
        swing_high: float, swing_low: float, current_price: float | None = None
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement levels.

        Used to identify potential support levels during a pullback in an uptrend,
        or resistance levels during a rally in a downtrend.

        Args:
            swing_high: Recent swing high price
            swing_low: Recent swing low price
            current_price: Current market price (optional, for analysis)

        Returns:
            FibonacciLevels with retracement levels

        Common Usage:
            - In uptrend: Buy at 38.2%, 50%, or 61.8% retracement of the swing
            - In downtrend: Sell at 38.2%, 50%, or 61.8% retracement of the rally
            - 50% is not a Fibonacci ratio but commonly used

        Reference:
            Fibonacci ratios derive from the golden ratio (φ ≈ 1.618)
            23.6% = 1 - (1/1.618^3)
            38.2% = 1 - (1/1.618)
            61.8% = 1/1.618
            78.6% = √(1/1.618)
        """
        diff = swing_high - swing_low

        levels = {
            "0.0%": swing_high,
            "23.6%": swing_high - (diff * 0.236),
            "38.2%": swing_high - (diff * 0.382),
            "50.0%": swing_high - (diff * 0.500),
            "61.8%": swing_high - (diff * 0.618),
            "78.6%": swing_high - (diff * 0.786),
            "100.0%": swing_low,
        }

        # Find nearest level if current price provided
        if current_price is not None:
            distances = {
                name: abs(price - current_price) / current_price for name, price in levels.items()
            }
            nearest = min(distances, key=distances.get)  # type: ignore[arg-type]
            nearest_dist = distances[nearest]
        else:
            nearest = "unknown"
            nearest_dist = 0.0

        return FibonacciLevels(
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=current_price if current_price is not None else 0.0,
            levels=levels,
            direction="retracement",
            nearest_level=nearest,
            nearest_distance=nearest_dist,
        )

    @staticmethod
    def fibonacci_extension(
        swing_high: float, swing_low: float, current_price: float | None = None
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci extension levels.

        Used to identify potential profit targets beyond the original swing high/low.

        Args:
            swing_high: Recent swing high price
            swing_low: Recent swing low price
            current_price: Current market price (optional, for analysis)

        Returns:
            FibonacciLevels with extension levels

        Common Usage:
            - In uptrend: Profit targets at 161.8%, 261.8% extension above swing low
            - In downtrend: Profit targets at 161.8%, 261.8% extension below swing high
            - 161.8% is the primary extension target (φ = 1.618)

        Reference:
            Extension levels project the move beyond the original range
            161.8% = 1.618 (golden ratio φ)
            261.8% = φ^2
            423.6% = φ^3
        """
        diff = swing_high - swing_low

        levels = {
            "0.0%": swing_low,
            "100.0%": swing_high,
            "161.8%": swing_low + (diff * 1.618),
            "200.0%": swing_low + (diff * 2.000),
            "261.8%": swing_low + (diff * 2.618),
            "361.8%": swing_low + (diff * 3.618),
            "423.6%": swing_low + (diff * 4.236),
        }

        # Find nearest level if current price provided
        if current_price is not None:
            distances = {
                name: abs(price - current_price) / current_price for name, price in levels.items()
            }
            nearest = min(distances, key=distances.get)  # type: ignore[arg-type]
            nearest_dist = distances[nearest]
        else:
            nearest = "unknown"
            nearest_dist = 0.0

        return FibonacciLevels(
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=current_price if current_price is not None else 0.0,
            levels=levels,
            direction="extension",
            nearest_level=nearest,
            nearest_distance=nearest_dist,
        )

    @staticmethod
    def pivot_points_classic(high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate classic pivot points.

        Most widely used pivot point calculation. Provides intraday support and
        resistance levels based on previous period's high, low, and close.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            PivotLevels with support and resistance

        Calculation:
            Pivot = (High + Low + Close) / 3
            R1 = (2 * Pivot) - Low
            R2 = Pivot + (High - Low)
            R3 = High + 2 * (Pivot - Low)
            S1 = (2 * Pivot) - High
            S2 = Pivot - (High - Low)
            S3 = Low - 2 * (High - Pivot)

        Usage:
            - Price above pivot: Bullish bias
            - Price below pivot: Bearish bias
            - R1, R2, R3: Resistance levels for profit targets
            - S1, S2, S3: Support levels for entry points
        """
        pivot = (high + low + close) / 3

        resistance_1 = (2 * pivot) - low
        resistance_2 = pivot + (high - low)
        resistance_3 = high + 2 * (pivot - low)

        support_1 = (2 * pivot) - high
        support_2 = pivot - (high - low)
        support_3 = low - 2 * (high - pivot)

        return PivotLevels(
            pivot=pivot,
            resistance_1=resistance_1,
            resistance_2=resistance_2,
            resistance_3=resistance_3,
            support_1=support_1,
            support_2=support_2,
            support_3=support_3,
            pivot_type="classic",
        )

    @staticmethod
    def pivot_points_fibonacci(high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate Fibonacci pivot points.

        Uses Fibonacci ratios (38.2%, 61.8%, 100%) to calculate support and resistance.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            PivotLevels with Fibonacci-based support and resistance

        Calculation:
            Pivot = (High + Low + Close) / 3
            R1 = Pivot + 0.382 * (High - Low)
            R2 = Pivot + 0.618 * (High - Low)
            R3 = Pivot + 1.000 * (High - Low)
            S1 = Pivot - 0.382 * (High - Low)
            S2 = Pivot - 0.618 * (High - Low)
            S3 = Pivot - 1.000 * (High - Low)
        """
        pivot = (high + low + close) / 3
        range_hl = high - low

        resistance_1 = pivot + (0.382 * range_hl)
        resistance_2 = pivot + (0.618 * range_hl)
        resistance_3 = pivot + (1.000 * range_hl)

        support_1 = pivot - (0.382 * range_hl)
        support_2 = pivot - (0.618 * range_hl)
        support_3 = pivot - (1.000 * range_hl)

        return PivotLevels(
            pivot=pivot,
            resistance_1=resistance_1,
            resistance_2=resistance_2,
            resistance_3=resistance_3,
            support_1=support_1,
            support_2=support_2,
            support_3=support_3,
            pivot_type="fibonacci",
        )

    @staticmethod
    def find_swing_high_low(
        data: pd.DataFrame, lookback: int = 20, current_index: int | None = None
    ) -> tuple[float, float, int, int]:
        """
        Identify recent swing high and swing low.

        A swing high is a peak where price is higher than N bars before and after.
        A swing low is a trough where price is lower than N bars before and after.

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to look back (default 20)
            current_index: Index to calculate from (default: last bar)

        Returns:
            (swing_high, swing_low, high_index, low_index) tuple

        Usage:
            Use these swings as input for Fibonacci calculations:
            ```python
            swing_high, swing_low, _, _ = StaticLevels.find_swing_high_low(data)
            fib = StaticLevels.fibonacci_retracement(swing_high, swing_low)
            ```
        """
        if current_index is None:
            current_index = len(data) - 1

        start_index = max(0, current_index - lookback)
        window = data.iloc[start_index : current_index + 1]

        swing_high = window["high"].max()
        swing_low = window["low"].min()

        high_index = window["high"].idxmax()
        low_index = window["low"].idxmin()

        return float(swing_high), float(swing_low), int(high_index), int(low_index)
