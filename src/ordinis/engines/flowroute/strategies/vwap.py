"""VWAP (Volume Weighted Average Price) Strategy."""

import logging

from .base import BaseStrategy, Signal, SignalStrength

logger = logging.getLogger(__name__)


class VWAPStrategy(BaseStrategy):
    """
    VWAP-based strategy.

    Generates BUY when price crosses above VWAP.
    Generates SELL when price crosses below VWAP.

    VWAP is calculated from market open each day.
    """

    def __init__(
        self,
        deviation_threshold: float = 0.005,  # 0.5% deviation to confirm signal
        name: str = "VWAP",
    ):
        super().__init__(name)
        self.deviation_threshold = deviation_threshold
        self.prices: list[float] = []
        self.volumes: list[float] = []
        self.prev_signal_direction: str | None = None
        self.cumulative_tpv: float = 0.0  # Typical Price * Volume
        self.cumulative_volume: float = 0.0

    def update(self, price: float, **kwargs) -> Signal | None:
        """
        Update with new price and volume.

        Args:
            price: Current price
            **kwargs: Must include 'volume'
        """
        volume = kwargs.get("volume", 1.0)  # Default to 1 if no volume provided

        self.prices.append(price)
        self.volumes.append(volume)

        # Update cumulative values
        typical_price = price  # For simplicity, using price directly
        self.cumulative_tpv += typical_price * volume
        self.cumulative_volume += volume

        # Keep history reasonable
        if len(self.prices) > 500:
            self.prices = self.prices[-500:]
            self.volumes = self.volumes[-500:]

        # Need at least a few data points
        if len(self.prices) < 5:
            return None

        if not self.initialized:
            self.initialized = True
            logger.info(f"[{self.name}] Initialized with {len(self.prices)} bars")

        # Calculate VWAP
        if self.cumulative_volume == 0:
            return None

        vwap = self.cumulative_tpv / self.cumulative_volume
        deviation = (price - vwap) / vwap

        # Generate signals on crossover with sufficient deviation
        if deviation > self.deviation_threshold and self.prev_signal_direction != "buy":
            # Price above VWAP - bullish
            self.prev_signal_direction = "buy"
            confidence = min(abs(deviation) / (self.deviation_threshold * 2), 1.0)

            signal = Signal(
                direction="buy",
                strength=SignalStrength.BUY,
                confidence=confidence,
                reason=f"Price above VWAP: ${price:.2f} vs ${vwap:.2f} ({deviation * 100:.2f}%)",
                metadata={
                    "price": price,
                    "vwap": vwap,
                    "deviation": deviation,
                    "volume": volume,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        if deviation < -self.deviation_threshold and self.prev_signal_direction != "sell":
            # Price below VWAP - bearish
            self.prev_signal_direction = "sell"
            confidence = min(abs(deviation) / (self.deviation_threshold * 2), 1.0)

            signal = Signal(
                direction="sell",
                strength=SignalStrength.SELL,
                confidence=confidence,
                reason=f"Price below VWAP: ${price:.2f} vs ${vwap:.2f} ({deviation * 100:.2f}%)",
                metadata={
                    "price": price,
                    "vwap": vwap,
                    "deviation": deviation,
                    "volume": volume,
                },
            )
            self.last_signal = signal
            logger.info(
                f"[{self.name}] {signal.direction.upper()} - {signal.reason} "
                f"(confidence: {signal.confidence:.2%})"
            )
            return signal

        # Log VWAP periodically
        if len(self.prices) % 20 == 0:
            logger.debug(
                f"[{self.name}] Price: ${price:.2f}, VWAP: ${vwap:.2f}, "
                f"Dev: {deviation * 100:.2f}%"
            )

        return None

    def reset(self) -> None:
        """Reset strategy state (call this at market open)."""
        super().reset()
        self.prices = []
        self.volumes = []
        self.cumulative_tpv = 0.0
        self.cumulative_volume = 0.0
        self.prev_signal_direction = None
        logger.info(f"[{self.name}] Reset for new trading day")
