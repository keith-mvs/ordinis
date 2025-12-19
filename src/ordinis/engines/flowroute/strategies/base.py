"""Base strategy class for all trading strategies."""

from dataclasses import dataclass
from enum import Enum


class SignalStrength(Enum):
    """Signal strength levels."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Signal:
    """Trading signal with metadata."""

    direction: str  # 'buy', 'sell', 'neutral'
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    reason: str
    metadata: dict


class BaseStrategy:
    """Base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.last_signal: Signal | None = None

    def update(self, price: float, **kwargs) -> Signal | None:
        """
        Update strategy with new data and generate signal.

        Args:
            price: Current price
            **kwargs: Additional data (volume, timestamp, etc.)

        Returns:
            Signal if conditions met, None otherwise
        """
        raise NotImplementedError("Strategy must implement update()")

    def reset(self) -> None:
        """Reset strategy state."""
        self.initialized = False
        self.last_signal = None

    def get_status(self) -> dict:
        """Get current strategy status."""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "last_signal": self.last_signal.direction if self.last_signal else None,
        }

    def is_ready(self) -> bool:
        """Check if strategy has enough data to generate signals."""
        return self.initialized
