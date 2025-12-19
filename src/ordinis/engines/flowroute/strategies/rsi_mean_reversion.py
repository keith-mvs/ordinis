"""RSI Mean Reversion Strategy."""

import logging

from .base import BaseStrategy, Signal, SignalStrength

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.

    Generates BUY when RSI < oversold threshold (default 30).
    Generates SELL when RSI > overbought threshold (default 70).
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: str = "RSI_MeanReversion",
    ):
        super().__init__(name)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.prices: list[float] = []
        self.prev_signal_direction: str | None = None

    def _calculate_rsi(self) -> float | None:
        """Calculate RSI indicator."""
        if len(self.prices) < self.period + 1:
            return None

        # Calculate price changes
        deltas = [self.prices[i] - self.prices[i - 1] for i in range(-self.period, 0)]

        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        # Average gain and loss
        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period

        if avg_loss == 0:
            return 100.0  # No losses = max RSI

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def update(self, price: float, **kwargs) -> Signal | None:
        """Update with new price and generate signal."""
        self.prices.append(price)

        # Keep only necessary history
        if len(self.prices) > self.period * 3:
            self.prices = self.prices[-self.period * 3 :]

        # Need enough data
        if len(self.prices) < self.period + 1:
            if len(self.prices) % 5 == 0:
                logger.debug(f"[{self.name}] Warmup: {len(self.prices)}/{self.period + 1}")
            return None

        if not self.initialized:
            self.initialized = True
            logger.info(f"[{self.name}] Initialized with {len(self.prices)} bars")

        # Calculate RSI
        rsi = self._calculate_rsi()
        if rsi is None:
            return None

        # Generate signals
        if rsi < self.oversold and self.prev_signal_direction != "buy":
            # Oversold - mean reversion buy
            self.prev_signal_direction = "buy"
            distance = self.oversold - rsi
            confidence = min(distance / self.oversold, 1.0)

            signal = Signal(
                direction="buy",
                strength=SignalStrength.STRONG_BUY if rsi < 20 else SignalStrength.BUY,
                confidence=confidence,
                reason=f"RSI oversold at {rsi:.1f} (threshold: {self.oversold})",
                metadata={
                    "rsi": rsi,
                    "threshold": self.oversold,
                    "price": price,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        if rsi > self.overbought and self.prev_signal_direction != "sell":
            # Overbought - mean reversion sell
            self.prev_signal_direction = "sell"
            distance = rsi - self.overbought
            confidence = min(distance / (100 - self.overbought), 1.0)

            signal = Signal(
                direction="sell",
                strength=SignalStrength.STRONG_SELL if rsi > 80 else SignalStrength.SELL,
                confidence=confidence,
                reason=f"RSI overbought at {rsi:.1f} (threshold: {self.overbought})",
                metadata={
                    "rsi": rsi,
                    "threshold": self.overbought,
                    "price": price,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        # Log current RSI periodically
        if len(self.prices) % 10 == 0:
            logger.debug(f"[{self.name}] RSI: {rsi:.1f}")

        return None

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.prices = []
        self.prev_signal_direction = None
