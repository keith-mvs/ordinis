"""
Signal types and data structures for SignalCore.

Signals are quantitative assessments, not direct trading orders.
They represent probabilistic predictions that RiskGuard validates.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Type of trading signal."""

    ENTRY = "entry"  # Open a new position
    EXIT = "exit"  # Close existing position
    SCALE = "scale"  # Adjust position size
    HOLD = "hold"  # Maintain current position


class Direction(Enum):
    """Trade direction."""

    LONG = "long"  # Buy/bullish
    SHORT = "short"  # Sell/bearish
    NEUTRAL = "neutral"  # No directional bias


@dataclass
class Signal:
    """
    Quantitative signal output from SignalCore.

    Signals are NOT direct orders - they are probabilistic assessments
    that must pass through RiskGuard before becoming executable orders.

    Attributes:
        symbol: Stock ticker symbol
        timestamp: When signal was generated
        signal_type: Entry, exit, scale, or hold
        direction: Long, short, or neutral

        # Quantitative outputs
        probability: Probability of favorable outcome (0.0 to 1.0)
        expected_return: Point estimate of expected return
        confidence_interval: (lower, upper) bounds on expected return
        score: Composite signal strength (-1.0 to +1.0)

        # Model attribution
        model_id: Identifier of generating model
        model_version: Model version string
        feature_contributions: Feature importance for explainability

        # Metadata
        regime: Current detected market regime
        data_quality: Input data quality score (0.0 to 1.0)
        staleness: Age of underlying data
        metadata: Additional model-specific data
    """

    symbol: str
    timestamp: datetime
    signal_type: SignalType
    direction: Direction

    # Quantitative outputs
    probability: float
    expected_return: float
    confidence_interval: tuple[float, float]
    score: float

    # Model attribution
    model_id: str
    model_version: str
    feature_contributions: dict[str, float] = field(default_factory=dict)

    # Metadata
    regime: str = "unknown"
    data_quality: float = 1.0
    staleness: timedelta = timedelta(seconds=0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")

        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [-1, 1], got {self.score}")

        if not 0.0 <= self.data_quality <= 1.0:
            raise ValueError(f"Data quality must be in [0, 1], got {self.data_quality}")

        lower, upper = self.confidence_interval
        if lower > upper:
            raise ValueError(f"Invalid confidence interval: ({lower}, {upper})")

    def is_actionable(self, min_probability: float = 0.6, min_score: float = 0.3) -> bool:
        """
        Check if signal meets minimum thresholds for action.

        Args:
            min_probability: Minimum probability threshold
            min_score: Minimum absolute score threshold

        Returns:
            True if signal is actionable
        """
        return (
            self.probability >= min_probability
            and abs(self.score) >= min_score
            and self.signal_type != SignalType.HOLD
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "probability": self.probability,
            "expected_return": self.expected_return,
            "confidence_interval": self.confidence_interval,
            "score": self.score,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "feature_contributions": self.feature_contributions,
            "regime": self.regime,
            "data_quality": self.data_quality,
            "staleness_seconds": self.staleness.total_seconds(),
            "metadata": self.metadata,
        }


@dataclass
class SignalBatch:
    """
    Collection of signals for portfolio-level decisions.

    Attributes:
        timestamp: Batch generation time
        signals: List of individual signals
        universe: Symbols in current trading universe
        regime_state: Current market regime information
        portfolio_context: Portfolio state and constraints
    """

    timestamp: datetime
    signals: list[Signal]
    universe: list[str]
    regime_state: dict[str, Any] = field(default_factory=dict)
    portfolio_context: dict[str, Any] = field(default_factory=dict)

    def filter_actionable(
        self, min_probability: float = 0.6, min_score: float = 0.3
    ) -> list[Signal]:
        """
        Filter to only actionable signals.

        Args:
            min_probability: Minimum probability threshold
            min_score: Minimum absolute score threshold

        Returns:
            List of actionable signals
        """
        return [
            signal for signal in self.signals if signal.is_actionable(min_probability, min_score)
        ]

    def get_by_symbol(self, symbol: str) -> Signal | None:
        """Get signal for specific symbol."""
        for signal in self.signals:
            if signal.symbol == symbol:
                return signal
        return None

    def get_entry_signals(self) -> list[Signal]:
        """Get all entry signals."""
        return [s for s in self.signals if s.signal_type == SignalType.ENTRY]

    def get_exit_signals(self) -> list[Signal]:
        """Get all exit signals."""
        return [s for s in self.signals if s.signal_type == SignalType.EXIT]

    def to_dict(self) -> dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "universe": self.universe,
            "regime_state": self.regime_state,
            "portfolio_context": self.portfolio_context,
        }
