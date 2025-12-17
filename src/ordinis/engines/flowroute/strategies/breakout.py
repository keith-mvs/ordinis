"""Breakout/Momentum Strategy."""

import logging

from .base import BaseStrategy, Signal, SignalStrength

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy based on recent high/low.

    Generates BUY when price breaks above recent high.
    Generates SELL when price breaks below recent low.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        breakout_threshold: float = 0.01,  # 1% breakout
        name: str = "Breakout",
    ):
        super().__init__(name)
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.prices: list[float] = []
        self.prev_signal_direction: str | None = None

    def update(self, price: float, **kwargs) -> Signal | None:
        """Update with new price and generate signal."""
        self.prices.append(price)

        # Keep only necessary history
        if len(self.prices) > self.lookback_period * 2:
            self.prices = self.prices[-self.lookback_period * 2 :]

        # Need enough data
        if len(self.prices) < self.lookback_period:
            if len(self.prices) % 5 == 0:
                logger.debug(f"[{self.name}] Warmup: {len(self.prices)}/{self.lookback_period}")
            return None

        if not self.initialized:
            self.initialized = True
            logger.info(f"[{self.name}] Initialized with {len(self.prices)} bars")

        # Calculate recent high/low (excluding current price)
        lookback_prices = self.prices[-self.lookback_period - 1 : -1]
        recent_high = max(lookback_prices)
        recent_low = min(lookback_prices)
        breakout_high = recent_high * (1 + self.breakout_threshold)
        breakout_low = recent_low * (1 - self.breakout_threshold)

        # Detect breakout
        if price > breakout_high and self.prev_signal_direction != "buy":
            # Upside breakout
            self.prev_signal_direction = "buy"
            breakout_pct = ((price - recent_high) / recent_high) * 100
            confidence = min(breakout_pct / (self.breakout_threshold * 100), 1.0)

            signal = Signal(
                direction="buy",
                strength=SignalStrength.STRONG_BUY if breakout_pct > 2.0 else SignalStrength.BUY,
                confidence=confidence,
                reason=f"Breakout above {recent_high:.2f} (+{breakout_pct:.2f}%)",
                metadata={
                    "price": price,
                    "recent_high": recent_high,
                    "breakout_pct": breakout_pct,
                    "lookback_period": self.lookback_period,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        if price < breakout_low and self.prev_signal_direction != "sell":
            # Downside breakdown
            self.prev_signal_direction = "sell"
            breakdown_pct = ((recent_low - price) / recent_low) * 100
            confidence = min(breakdown_pct / (self.breakout_threshold * 100), 1.0)

            signal = Signal(
                direction="sell",
                strength=SignalStrength.STRONG_SELL if breakdown_pct > 2.0 else SignalStrength.SELL,
                confidence=confidence,
                reason=f"Breakdown below {recent_low:.2f} (-{breakdown_pct:.2f}%)",
                metadata={
                    "price": price,
                    "recent_low": recent_low,
                    "breakdown_pct": breakdown_pct,
                    "lookback_period": self.lookback_period,
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
