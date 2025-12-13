"""
Signal-Driven Rebalancing Strategy.

Allocates portfolio weights based on trading signals from technical indicators,
fundamental analysis, or other signal sources.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pandas as pd


class SignalMethod(Enum):
    """Methods for converting signals to weights."""

    PROPORTIONAL = "proportional"  # Weight proportional to signal strength
    BINARY = "binary"  # Full weight to positive signals, zero to negative
    RANKED = "ranked"  # Weight based on signal ranking


@dataclass
class SignalInput:
    """Trading signal for a symbol.

    Attributes:
        symbol: Ticker symbol
        signal: Signal value (typically -1 to +1, where +1 = strong buy, -1 = strong sell)
        confidence: Optional confidence level (0 to 1)
        source: Optional source identifier (e.g., "RSI", "MACD", "composite")
    """

    symbol: str
    signal: float
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate signal and confidence ranges."""
        if not -2.0 <= self.signal <= 2.0:
            raise ValueError(f"signal should be in [-2, 2], got {self.signal}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class SignalDrivenWeights:
    """Calculated signal-driven weights for symbols.

    Attributes:
        weights: Dictionary of {symbol: weight}
        signals: Dictionary of {symbol: signal_value}
        confidences: Dictionary of {symbol: confidence}
        method: Method used to convert signals to weights
        timestamp: When weights were calculated
    """

    weights: dict[str, float]
    signals: dict[str, float]
    confidences: dict[str, float]
    method: SignalMethod
    timestamp: datetime


@dataclass
class SignalDrivenDecision:
    """Rebalancing action for signal-driven strategy.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current allocation as decimal (0.0 to 1.0)
        target_weight: Signal-based weight as decimal (0.0 to 1.0)
        signal: Signal value that drove the allocation
        confidence: Confidence in the signal
        adjustment_shares: Number of shares to buy (+) or sell (-)
        adjustment_value: Dollar value of adjustment
        timestamp: When the decision was generated
    """

    symbol: str
    current_weight: float
    target_weight: float
    signal: float
    confidence: float
    adjustment_shares: float
    adjustment_value: float
    timestamp: datetime


class SignalDrivenRebalancer:
    """Rebalances portfolio based on trading signals.

    Converts trading signals (e.g., from technical indicators) into portfolio
    weights and generates rebalancing orders.

    Supports:
    - Multiple signal conversion methods (proportional, binary, ranked)
    - Confidence weighting
    - Min/max weight constraints
    - Cash buffer for risk management

    Example:
        >>> signals = [
        ...     SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
        ...     SignalInput("MSFT", signal=0.3, confidence=0.7, source="MACD"),
        ...     SignalInput("GOOGL", signal=-0.2, confidence=0.6, source="RSI"),
        ... ]
        >>> rebalancer = SignalDrivenRebalancer(
        ...     method=SignalMethod.PROPORTIONAL,
        ...     min_weight=0.0,
        ...     max_weight=0.50,
        ... )
        >>> weights = rebalancer.calculate_weights(signals)
        >>> positions = {"AAPL": 10, "MSFT": 20, "GOOGL": 15}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> decisions = rebalancer.generate_rebalance_orders(signals, positions, prices)
    """

    def __init__(
        self,
        method: SignalMethod = SignalMethod.PROPORTIONAL,
        min_weight: float = 0.0,
        max_weight: float = 0.50,
        min_signal_threshold: float = 0.0,
        cash_buffer: float = 0.0,
        drift_threshold: float = 0.10,
    ) -> None:
        """Initialize signal-driven rebalancer.

        Args:
            method: Method for converting signals to weights
            min_weight: Minimum weight per asset (default: 0.0)
            max_weight: Maximum weight per asset (default: 0.50 = 50%)
            min_signal_threshold: Minimum signal value to consider (default: 0.0)
            cash_buffer: Percentage of portfolio to keep as cash (default: 0.0)
            drift_threshold: Maximum acceptable drift from target (default: 0.10 = 10%)

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= min_weight < max_weight <= 1.0:
            raise ValueError(
                f"Must have 0 <= min_weight < max_weight <= 1, got {min_weight}, {max_weight}"
            )
        if not -2.0 <= min_signal_threshold <= 2.0:
            raise ValueError(f"min_signal_threshold must be in [-2, 2], got {min_signal_threshold}")
        if not 0.0 <= cash_buffer < 1.0:
            raise ValueError(f"cash_buffer must be in [0, 1), got {cash_buffer}")
        if not 0.0 < drift_threshold <= 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1], got {drift_threshold}")

        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_signal_threshold = min_signal_threshold
        self.cash_buffer = cash_buffer
        self.drift_threshold = drift_threshold

    def calculate_weights(
        self,
        signals: list[SignalInput],
    ) -> SignalDrivenWeights:
        """Calculate portfolio weights from trading signals.

        Args:
            signals: List of SignalInput objects

        Returns:
            SignalDrivenWeights object with calculated weights

        Raises:
            ValueError: If signals list is empty or has no positive signals
        """
        if not signals:
            raise ValueError("Signals list cannot be empty")

        # Filter signals by threshold
        filtered_signals = [s for s in signals if s.signal >= self.min_signal_threshold]

        if not filtered_signals:
            raise ValueError(f"No signals above threshold {self.min_signal_threshold}")

        # Convert signals to weights based on method
        if self.method == SignalMethod.PROPORTIONAL:
            weights_dict = self._proportional_weights(filtered_signals)
        elif self.method == SignalMethod.BINARY:
            weights_dict = self._binary_weights(filtered_signals)
        elif self.method == SignalMethod.RANKED:
            weights_dict = self._ranked_weights(filtered_signals)
        else:
            raise ValueError(f"Unknown signal method: {self.method}")

        # Apply cash buffer (reduce all weights proportionally)
        if self.cash_buffer > 0:
            equity_allocation = 1.0 - self.cash_buffer
            weights_dict = {sym: w * equity_allocation for sym, w in weights_dict.items()}

        # Apply min/max constraints
        constrained = {
            sym: max(self.min_weight, min(self.max_weight, w)) for sym, w in weights_dict.items()
        }

        # Renormalize to sum to (1 - cash_buffer)
        target_sum = 1.0 - self.cash_buffer
        current_sum = sum(constrained.values())
        if current_sum > 0:
            final_weights = {sym: (w / current_sum) * target_sum for sym, w in constrained.items()}
        else:
            final_weights = constrained

        # Extract signals and confidences
        signals_dict = {s.symbol: s.signal for s in filtered_signals}
        confidences_dict = {s.symbol: s.confidence for s in filtered_signals}

        return SignalDrivenWeights(
            weights=final_weights,
            signals=signals_dict,
            confidences=confidences_dict,
            method=self.method,
            timestamp=datetime.now(tz=UTC),
        )

    def _proportional_weights(self, signals: list[SignalInput]) -> dict[str, float]:
        """Calculate weights proportional to signal strength * confidence."""
        # Weight = signal * confidence
        raw_weights = {s.symbol: max(0, s.signal * s.confidence) for s in signals}

        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total == 0:
            return {s.symbol: 1.0 / len(signals) for s in signals}

        return {sym: w / total for sym, w in raw_weights.items()}

    def _binary_weights(self, signals: list[SignalInput]) -> dict[str, float]:
        """Equal weight to all positive signals, zero to negative."""
        positive_symbols = [s.symbol for s in signals if s.signal > 0]

        if not positive_symbols:
            raise ValueError("No positive signals for binary weighting")

        equal_weight = 1.0 / len(positive_symbols)
        return {sym: equal_weight for sym in positive_symbols}

    def _ranked_weights(self, signals: list[SignalInput]) -> dict[str, float]:
        """Weight based on signal ranking (best signal gets most weight)."""
        # Sort by signal * confidence (descending)
        sorted_signals = sorted(signals, key=lambda s: s.signal * s.confidence, reverse=True)

        # Assign linearly decreasing weights (rank-based)
        n = len(sorted_signals)
        rank_weights = {sorted_signals[i].symbol: (n - i) / sum(range(1, n + 1)) for i in range(n)}

        return rank_weights

    def should_rebalance(
        self,
        signals: list[SignalInput],
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> bool:
        """Check if portfolio exceeds drift threshold from signal-driven weights.

        Args:
            signals: Current trading signals
            positions: Current shares held per symbol
            prices: Current price per symbol

        Returns:
            True if rebalancing is needed, False otherwise
        """
        # Calculate current weights
        total_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in prices)

        if total_value == 0.0:
            return True

        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value for sym in prices
        }

        # Calculate target weights
        target_weights = self.calculate_weights(signals)

        # Calculate max drift
        all_symbols = set(current_weights.keys()) | set(target_weights.weights.keys())
        max_drift = max(
            abs(current_weights.get(sym, 0.0) - target_weights.weights.get(sym, 0.0))
            for sym in all_symbols
        )

        return max_drift > self.drift_threshold

    def generate_rebalance_orders(
        self,
        signals: list[SignalInput],
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float = 0.0,
    ) -> list[SignalDrivenDecision]:
        """Generate rebalancing orders based on trading signals.

        Args:
            signals: Current trading signals
            positions: Current shares held per symbol
            prices: Current price per symbol
            cash: Available cash for rebalancing

        Returns:
            List of SignalDrivenDecision objects with buy/sell instructions
        """
        # Calculate signal-driven weights
        target_weights_obj = self.calculate_weights(signals)
        target_weights = target_weights_obj.weights

        # Get all symbols (union of current positions and signals)
        all_symbols = set(positions.keys()) | set(target_weights.keys())

        # Calculate total portfolio value
        equity_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in all_symbols)
        total_value = equity_value + cash

        if total_value == 0.0:
            return []

        # Calculate current weights
        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in all_symbols
        }

        # Generate rebalancing decisions
        decisions: list[SignalDrivenDecision] = []
        timestamp = datetime.now(tz=UTC)

        for sym in all_symbols:
            target_weight = target_weights.get(sym, 0.0)
            current_weight = current_weights.get(sym, 0.0)

            # Calculate target dollar value
            target_value = total_value * target_weight
            current_value = positions.get(sym, 0.0) * prices.get(sym, 0.0)

            # Calculate adjustment
            adjustment_value = target_value - current_value
            adjustment_shares = adjustment_value / prices.get(sym, 1.0) if sym in prices else 0.0

            decision = SignalDrivenDecision(
                symbol=sym,
                current_weight=current_weight,
                target_weight=target_weight,
                signal=target_weights_obj.signals.get(sym, 0.0),
                confidence=target_weights_obj.confidences.get(sym, 0.0),
                adjustment_shares=adjustment_shares,
                adjustment_value=adjustment_value,
                timestamp=timestamp,
            )
            decisions.append(decision)

        return decisions

    def get_rebalance_summary(
        self,
        decisions: list[SignalDrivenDecision],
    ) -> pd.DataFrame:
        """Convert rebalancing decisions to a summary DataFrame.

        Args:
            decisions: List of SignalDrivenDecision objects

        Returns:
            DataFrame with columns: symbol, current_weight, target_weight, signal,
                                   confidence, drift, adjustment_shares, adjustment_value
        """
        data = [
            {
                "symbol": d.symbol,
                "current_weight": d.current_weight,
                "target_weight": d.target_weight,
                "signal": d.signal,
                "confidence": d.confidence,
                "drift": d.current_weight - d.target_weight,
                "adjustment_shares": d.adjustment_shares,
                "adjustment_value": d.adjustment_value,
            }
            for d in decisions
        ]

        return pd.DataFrame(data)
