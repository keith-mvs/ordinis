"""
Moving Average Crossover Strategy.

Classic trend-following strategy using moving average crossovers.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

from .base import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Generates buy signals when fast MA crosses above slow MA (golden cross)
    and sell signals when fast MA crosses below slow MA (death cross).

    Default Parameters:
    - fast_period: 50 (fast moving average period)
    - slow_period: 200 (slow moving average period)
    - ma_type: 'SMA' (Simple Moving Average)
    - min_bars: 200 (minimum bars for signal generation)
    """

    def configure(self):
        """Configure moving average crossover parameters."""
        # Set default parameters
        self.params.setdefault("fast_period", 50)
        self.params.setdefault("slow_period", 200)
        self.params.setdefault("ma_type", "SMA")
        self.params.setdefault("min_bars", self.params["slow_period"] + 10)

    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate moving average crossover signal.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal object or None
        """
        # Validate data
        is_valid, _msg = self.validate_data(data)
        if not is_valid:
            return None

        try:
            close = data["close"]

            # Calculate moving averages
            if self.params["ma_type"] == "EMA":
                fast_ma = close.ewm(span=self.params["fast_period"], adjust=False).mean()
                slow_ma = close.ewm(span=self.params["slow_period"], adjust=False).mean()
            else:  # SMA
                fast_ma = close.rolling(window=self.params["fast_period"]).mean()
                slow_ma = close.rolling(window=self.params["slow_period"]).mean()

            # Get current and previous values
            fast_current = fast_ma.iloc[-1]
            fast_prev = fast_ma.iloc[-2]
            slow_current = slow_ma.iloc[-1]
            slow_prev = slow_ma.iloc[-2]

            # Detect crossovers
            golden_cross = fast_prev <= slow_prev and fast_current > slow_current
            death_cross = fast_prev >= slow_prev and fast_current < slow_current

            # Calculate signal strength
            ma_distance_pct = (fast_current - slow_current) / slow_current

            if golden_cross:
                # Buy signal - golden cross
                return Signal(
                    symbol=data.get("symbol", ["UNKNOWN"])[0] if "symbol" in data else "UNKNOWN",
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,
                    direction=Direction.LONG,
                    probability=0.65,
                    expected_return=0.05,
                    confidence_interval=(-0.02, 0.12),
                    score=abs(ma_distance_pct) * 100,
                    model_id=self.name,
                    model_version="1.0.0",
                    metadata={
                        "strategy": self.name,
                        "crossover_type": "golden_cross",
                        "fast_ma": fast_current,
                        "slow_ma": slow_current,
                        "ma_distance_pct": ma_distance_pct,
                    },
                )
            if death_cross:
                # Sell signal - death cross
                return Signal(
                    symbol=data.get("symbol", ["UNKNOWN"])[0] if "symbol" in data else "UNKNOWN",
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT,
                    direction=Direction.SHORT,
                    probability=0.65,
                    expected_return=-0.05,
                    confidence_interval=(-0.12, 0.02),
                    score=abs(ma_distance_pct) * 100,
                    model_id=self.name,
                    model_version="1.0.0",
                    metadata={
                        "strategy": self.name,
                        "crossover_type": "death_cross",
                        "fast_ma": fast_current,
                        "slow_ma": slow_current,
                        "ma_distance_pct": ma_distance_pct,
                    },
                )

            # No crossover - check if we're in a trend
            if fast_current > slow_current * 1.02:
                # Strong uptrend - hold signal
                return Signal(
                    symbol=data.get("symbol", ["UNKNOWN"])[0] if "symbol" in data else "UNKNOWN",
                    timestamp=timestamp,
                    signal_type=SignalType.HOLD,
                    direction=Direction.LONG,
                    probability=0.55,
                    expected_return=0.02,
                    confidence_interval=(-0.01, 0.05),
                    score=abs(ma_distance_pct) * 50,
                    model_id=self.name,
                    model_version="1.0.0",
                    metadata={
                        "strategy": self.name,
                        "trend": "uptrend",
                        "fast_ma": fast_current,
                        "slow_ma": slow_current,
                        "ma_distance_pct": ma_distance_pct,
                    },
                )

            return None

        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""Moving Average Crossover Strategy

Classic trend-following strategy using moving average crossovers.

Entry Rules:
- BUY when {self.params['fast_period']}-period MA crosses above {self.params['slow_period']}-period MA (Golden Cross)
- Indicates start of uptrend

Exit Rules:
- SELL when {self.params['fast_period']}-period MA crosses below {self.params['slow_period']}-period MA (Death Cross)
- Indicates start of downtrend

Hold Conditions:
- Maintain position when fast MA > slow MA by 2%+ (strong uptrend)

Parameters:
- Fast MA Period: {self.params['fast_period']} bars
- Slow MA Period: {self.params['slow_period']} bars
- MA Type: {self.params['ma_type']}

Best For:
- Trending markets
- Longer-term trading (swing/position)
- Low-frequency signals
- Clear directional moves

Risk Considerations:
- Lagging indicator (late entries/exits)
- Many false signals in choppy markets
- Can give back significant profits in trend reversals
- Works best with trend confirmation
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get("min_bars", self.params["slow_period"] + 10)
