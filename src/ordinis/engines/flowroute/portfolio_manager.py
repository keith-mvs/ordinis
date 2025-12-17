"""Portfolio Manager for multi-strategy trading."""

from dataclasses import dataclass
import logging

from .strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSignal:
    """Aggregated signal from multiple strategies."""

    direction: str  # 'buy', 'sell', 'neutral'
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    consensus: float  # 0.0 to 1.0 (% of strategies agreeing)
    contributing_strategies: list[str]
    reasons: list[str]


class PortfolioManager:
    """
    Manages multiple strategies and aggregates their signals.

    Supports different aggregation modes:
    - consensus: All strategies must agree
    - majority: >50% of strategies must agree
    - weighted: Weight strategies by their confidence
    - any: Execute if any strategy signals
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        mode: str = "weighted",
        min_strategies_ready: int = 1,
    ):
        """
        Initialize portfolio manager.

        Args:
            strategies: List of strategy instances
            mode: Aggregation mode ('consensus', 'majority', 'weighted', 'any')
            min_strategies_ready: Minimum strategies that must be initialized
        """
        self.strategies = strategies
        self.mode = mode
        self.min_strategies_ready = min_strategies_ready
        self.signal_history: list[PortfolioSignal] = []

        logger.info(f"[PORTFOLIO] Initialized with {len(strategies)} strategies (mode: {mode})")
        for strategy in strategies:
            logger.info(f"[PORTFOLIO]   - {strategy.name}")

    def update(self, price: float, **kwargs) -> PortfolioSignal | None:
        """
        Update all strategies and aggregate signals.

        Args:
            price: Current price
            **kwargs: Additional data (volume, etc.)

        Returns:
            Aggregated portfolio signal or None
        """
        # Update all strategies
        signals: list[tuple[BaseStrategy, Signal | None]] = []
        for strategy in self.strategies:
            signal = strategy.update(price, **kwargs)
            signals.append((strategy, signal))

        # Check if enough strategies are ready
        ready_count = sum(1 for s in self.strategies if s.is_ready())
        if ready_count < self.min_strategies_ready:
            if ready_count % 5 == 0:  # Log periodically
                logger.debug(
                    f"[PORTFOLIO] Waiting for strategies: {ready_count}/{self.min_strategies_ready} ready"
                )
            return None

        # Aggregate signals based on mode
        portfolio_signal = self._aggregate_signals(signals, price)

        if portfolio_signal:
            self.signal_history.append(portfolio_signal)
            # Keep history manageable
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]

        return portfolio_signal

    def _aggregate_signals(
        self,
        signals: list[tuple[BaseStrategy, Signal | None]],
        price: float,
    ) -> PortfolioSignal | None:
        """Aggregate signals from multiple strategies."""

        # Count signals by direction
        buy_signals = []
        sell_signals = []

        for strategy, signal in signals:
            if signal is None:
                continue

            if signal.direction == "buy":
                buy_signals.append((strategy, signal))
            elif signal.direction == "sell":
                sell_signals.append((strategy, signal))

        # No signals
        if not buy_signals and not sell_signals:
            return None

        # Apply aggregation logic
        if self.mode == "consensus":
            return self._consensus_aggregation(buy_signals, sell_signals, price)
        if self.mode == "majority":
            return self._majority_aggregation(buy_signals, sell_signals, price)
        if self.mode == "weighted":
            return self._weighted_aggregation(buy_signals, sell_signals, price)
        if self.mode == "any":
            return self._any_aggregation(buy_signals, sell_signals, price)
        logger.error(f"[PORTFOLIO] Unknown aggregation mode: {self.mode}")
        return None

    def _consensus_aggregation(
        self,
        buy_signals: list,
        sell_signals: list,
        price: float,
    ) -> PortfolioSignal | None:
        """All strategies must agree."""
        ready_strategies = [s for s in self.strategies if s.is_ready()]

        if buy_signals and len(buy_signals) == len(ready_strategies):
            # All agree on BUY
            avg_confidence = sum(s.confidence for _, s in buy_signals) / len(buy_signals)
            return PortfolioSignal(
                direction="buy",
                strength=1.0,
                confidence=avg_confidence,
                consensus=1.0,
                contributing_strategies=[s.name for s, _ in buy_signals],
                reasons=[sig.reason for _, sig in buy_signals],
            )
        if sell_signals and len(sell_signals) == len(ready_strategies):
            # All agree on SELL
            avg_confidence = sum(s.confidence for _, s in sell_signals) / len(sell_signals)
            return PortfolioSignal(
                direction="sell",
                strength=-1.0,
                confidence=avg_confidence,
                consensus=1.0,
                contributing_strategies=[s.name for s, _ in sell_signals],
                reasons=[sig.reason for _, sig in sell_signals],
            )

        return None

    def _majority_aggregation(
        self,
        buy_signals: list,
        sell_signals: list,
        price: float,
    ) -> PortfolioSignal | None:
        """More than 50% of strategies must agree."""
        ready_strategies = [s for s in self.strategies if s.is_ready()]
        majority = len(ready_strategies) / 2

        if len(buy_signals) > majority:
            avg_confidence = sum(s.confidence for _, s in buy_signals) / len(buy_signals)
            consensus = len(buy_signals) / len(ready_strategies)
            return PortfolioSignal(
                direction="buy",
                strength=consensus,
                confidence=avg_confidence,
                consensus=consensus,
                contributing_strategies=[s.name for s, _ in buy_signals],
                reasons=[sig.reason for _, sig in buy_signals],
            )
        if len(sell_signals) > majority:
            avg_confidence = sum(s.confidence for _, s in sell_signals) / len(sell_signals)
            consensus = len(sell_signals) / len(ready_strategies)
            return PortfolioSignal(
                direction="sell",
                strength=-consensus,
                confidence=avg_confidence,
                consensus=consensus,
                contributing_strategies=[s.name for s, _ in sell_signals],
                reasons=[sig.reason for _, sig in sell_signals],
            )

        return None

    def _weighted_aggregation(
        self,
        buy_signals: list,
        sell_signals: list,
        price: float,
    ) -> PortfolioSignal | None:
        """Weight strategies by their confidence."""
        # Calculate weighted score
        buy_score = sum(s.confidence * s.strength.value for _, s in buy_signals)
        sell_score = sum(abs(s.confidence * s.strength.value) for _, s in sell_signals)

        total_score = buy_score + sell_score
        if total_score == 0:
            return None

        # Net score
        net_score = buy_score - sell_score

        # Need significant score to generate signal (>5% weighted agreement - very low for testing)
        threshold = 0.05 * len([s for s in self.strategies if s.is_ready()])

        if net_score > threshold:
            # Net BUY signal
            avg_confidence = (
                sum(s.confidence for _, s in buy_signals) / len(buy_signals) if buy_signals else 0.0
            )
            consensus = len(buy_signals) / len([s for s in self.strategies if s.is_ready()])

            return PortfolioSignal(
                direction="buy",
                strength=min(net_score / len(self.strategies), 1.0),
                confidence=avg_confidence,
                consensus=consensus,
                contributing_strategies=[s.name for s, _ in buy_signals],
                reasons=[sig.reason for _, sig in buy_signals],
            )
        if net_score < -threshold:
            # Net SELL signal
            avg_confidence = (
                sum(s.confidence for _, s in sell_signals) / len(sell_signals)
                if sell_signals
                else 0.0
            )
            consensus = len(sell_signals) / len([s for s in self.strategies if s.is_ready()])

            return PortfolioSignal(
                direction="sell",
                strength=max(net_score / len(self.strategies), -1.0),
                confidence=avg_confidence,
                consensus=consensus,
                contributing_strategies=[s.name for s, _ in sell_signals],
                reasons=[sig.reason for _, sig in sell_signals],
            )

        return None

    def _any_aggregation(
        self,
        buy_signals: list,
        sell_signals: list,
        price: float,
    ) -> PortfolioSignal | None:
        """Execute if any strategy signals (most aggressive)."""
        if buy_signals:
            # Take highest confidence BUY
            best_signal = max(buy_signals, key=lambda x: x[1].confidence)
            strategy, signal = best_signal

            return PortfolioSignal(
                direction="buy",
                strength=signal.strength.value / 2,  # Reduce strength in 'any' mode
                confidence=signal.confidence,
                consensus=len(buy_signals) / len([s for s in self.strategies if s.is_ready()]),
                contributing_strategies=[s.name for s, _ in buy_signals],
                reasons=[sig.reason for _, sig in buy_signals],
            )
        if sell_signals:
            # Take highest confidence SELL
            best_signal = max(sell_signals, key=lambda x: x[1].confidence)
            _strategy, signal = best_signal

            return PortfolioSignal(
                direction="sell",
                strength=signal.strength.value / 2,
                confidence=signal.confidence,
                consensus=len(sell_signals) / len([s for s in self.strategies if s.is_ready()]),
                contributing_strategies=[s.name for s, _ in sell_signals],
                reasons=[sig.reason for _, sig in sell_signals],
            )

        return None

    def get_status(self) -> dict:
        """Get current portfolio status."""
        return {
            "total_strategies": len(self.strategies),
            "ready_strategies": sum(1 for s in self.strategies if s.is_ready()),
            "aggregation_mode": self.mode,
            "strategy_status": [s.get_status() for s in self.strategies],
            "signal_count": len(self.signal_history),
        }

    def reset(self) -> None:
        """Reset all strategies."""
        for strategy in self.strategies:
            strategy.reset()
        self.signal_history = []
        logger.info("[PORTFOLIO] Reset all strategies")
