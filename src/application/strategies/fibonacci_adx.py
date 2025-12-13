"""
Fibonacci + ADX Combined Strategy.

Combines Fibonacci retracement levels with ADX trend filter for high-probability entries.
"""

from datetime import datetime

import pandas as pd

from engines.signalcore import ModelConfig
from engines.signalcore.core.signal import Direction, Signal, SignalType
from engines.signalcore.models import ADXTrendModel, FibonacciRetracementModel

from .base import BaseStrategy


class FibonacciADXStrategy(BaseStrategy):
    """
    Fibonacci Retracement + ADX Filter Strategy.

    Combines Fibonacci levels with ADX trend confirmation:
    - ADX confirms strong trend (ADX > 25)
    - Fibonacci identifies entry levels on pullbacks
    - Enters at key levels (38.2%, 50%, 61.8%)

    Parameters:
        adx_period: ADX calculation period (default 14)
        adx_threshold: Minimum ADX for trend (default 25)
        swing_lookback: Bars for swing identification (default 50)
        fib_levels: Key Fibonacci levels (default [0.382, 0.5, 0.618])
        tolerance: Price tolerance near level (default 0.01 = 1%)

    Best Markets:
        - Trending markets with clear swings
        - Medium to high volatility
        - Avoid in choppy markets

    Risk Management:
        - Stop below next Fibonacci level
        - Target at swing high/low
        - Maximum 3% position size
    """

    def __init__(
        self,
        name: str,
        adx_period: int = 14,
        adx_threshold: float = 25,
        swing_lookback: int = 50,
        fib_levels: list[float] | None = None,
        tolerance: float = 0.01,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            adx_period: ADX calculation period
            adx_threshold: Minimum ADX for trend confirmation
            swing_lookback: Bars to look back for swings
            fib_levels: Key Fibonacci levels to watch
            tolerance: Price tolerance near level (as decimal)
        """
        super().__init__(name)

        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.swing_lookback = swing_lookback
        self.fib_levels = fib_levels if fib_levels is not None else [0.382, 0.5, 0.618]
        self.tolerance = tolerance

        # Initialize ADX model
        adx_config = ModelConfig(
            model_id=f"{name}-adx",
            model_type="trend",
            version="1.0.0",
            parameters={
                "adx_period": adx_period,
                "adx_threshold": adx_threshold,
            },
        )
        self.adx_model = ADXTrendModel(adx_config)

        # Initialize Fibonacci model
        fib_config = ModelConfig(
            model_id=f"{name}-fibonacci",
            model_type="static_level",
            version="1.0.0",
            parameters={
                "swing_lookback": swing_lookback,
                "key_levels": self.fib_levels,
                "tolerance": tolerance,
            },
        )
        self.fib_model = FibonacciRetracementModel(fib_config)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal if conditions met, None otherwise
        """
        # Check minimum data requirements
        min_required = max(
            self.adx_model.config.min_data_points, self.fib_model.config.min_data_points
        )
        if len(data) < min_required:
            return None

        # Get ADX signal to check trend strength
        adx_signal = self.adx_model.generate(data, timestamp)

        # Only proceed if ADX shows strong trend
        if adx_signal.metadata["adx"] < self.adx_threshold:
            return None

        # Get Fibonacci signal for entry level
        fib_signal = self.fib_model.generate(data, timestamp)

        # Check if Fibonacci and ADX agree on direction
        if (
            adx_signal.direction == fib_signal.direction
            and fib_signal.signal_type == SignalType.ENTRY
        ):
            # Both indicators agree - generate combined signal

            # Combine scores (ADX weighted 40%, Fibonacci 60%)
            combined_score = (adx_signal.score * 0.4) + (fib_signal.score * 0.6)
            combined_prob = (adx_signal.probability * 0.4) + (fib_signal.probability * 0.6)

            # Calculate stop loss and take profit
            swing_high = fib_signal.metadata["swing_high"]
            swing_low = fib_signal.metadata["swing_low"]
            current_price = fib_signal.metadata["current_price"]
            nearest_level = fib_signal.metadata["nearest_level"]

            if fib_signal.direction == Direction.LONG:
                # For long: stop below next lower Fib level
                stop_loss = swing_low * 0.98  # 2% below swing low
                take_profit = swing_high  # Target swing high
            else:
                # For short: stop above next higher Fib level
                stop_loss = swing_high * 1.02  # 2% above swing high
                take_profit = swing_low  # Target swing low

            # Combine metadata
            metadata = {
                **fib_signal.metadata,
                "adx": adx_signal.metadata["adx"],
                "plus_di": adx_signal.metadata["plus_di"],
                "minus_di": adx_signal.metadata["minus_di"],
                "strategy": self.name,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": abs(take_profit - current_price)
                / abs(current_price - stop_loss),
            }

            return Signal(
                symbol=fib_signal.symbol,
                timestamp=timestamp,
                signal_type=SignalType.ENTRY,
                direction=fib_signal.direction,
                probability=combined_prob,
                expected_return=fib_signal.expected_return * 1.3,  # Boost for dual confirmation
                confidence_interval=fib_signal.confidence_interval,
                score=combined_score,
                model_id=f"{self.name}-combined",
                model_version="1.0.0",
                feature_contributions={
                    **adx_signal.feature_contributions,
                    **fib_signal.feature_contributions,
                },
                regime=f"{adx_signal.regime}_{fib_signal.regime}",
                data_quality=min(adx_signal.data_quality, fib_signal.data_quality),
                staleness=max(adx_signal.staleness, fib_signal.staleness),
                metadata=metadata,
            )

        # No qualifying signal
        return None

    @property
    def required_bars(self) -> int:
        """Minimum bars required for signal generation."""
        return max(
            self.adx_model.config.min_data_points,
            self.fib_model.config.min_data_points,
        )
