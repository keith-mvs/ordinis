"""Moving Average Crossover Strategy."""

import logging

from .base import BaseStrategy, Signal, SignalStrength

logger = logging.getLogger(__name__)


class MACrossoverStrategy(BaseStrategy):
    """
    Moving average crossover strategy.

    Generates BUY when fast MA crosses above slow MA.
    Generates SELL when fast MA crosses below slow MA.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        name: str = "MA_Crossover",
    ):
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []
        self.prev_signal_direction: str | None = None

    def update(self, price: float, **kwargs) -> Signal | None:
        """Update with new price and generate signal."""
        self.prices.append(price)

        # Keep only necessary history
        if len(self.prices) > self.slow_period * 2:
            self.prices = self.prices[-self.slow_period * 2 :]

        # Need enough data
        if len(self.prices) < self.slow_period:
            if len(self.prices) % 10 == 0:
                logger.debug(f"[{self.name}] Warmup: {len(self.prices)}/{self.slow_period}")
            return None

        if not self.initialized:
            self.initialized = True
            logger.info(f"[{self.name}] Initialized with {len(self.prices)} bars")

        # Calculate MAs
        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        # Detect crossover
        if fast_ma > slow_ma and self.prev_signal_direction != "buy":
            # Bullish crossover
            self.prev_signal_direction = "buy"
            spread = fast_ma - slow_ma
            confidence = min(abs(spread / slow_ma) * 100, 1.0)  # Normalize to 0-1

            signal = Signal(
                direction="buy",
                strength=SignalStrength.BUY,
                confidence=confidence,
                reason=f"Fast MA ({fast_ma:.2f}) crossed above Slow MA ({slow_ma:.2f})",
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "spread": spread,
                    "price": price,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        if fast_ma < slow_ma and self.prev_signal_direction != "sell":
            # Bearish crossover
            self.prev_signal_direction = "sell"
            spread = slow_ma - fast_ma
            confidence = min(abs(spread / slow_ma) * 100, 1.0)

            signal = Signal(
                direction="sell",
                strength=SignalStrength.SELL,
                confidence=confidence,
                reason=f"Fast MA ({fast_ma:.2f}) crossed below Slow MA ({slow_ma:.2f})",
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "spread": spread,
                    "price": price,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        return None

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.prices = []
        self.prev_signal_direction = None
