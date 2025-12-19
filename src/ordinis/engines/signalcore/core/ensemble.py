"""
Signal Ensemble Logic.

Combines multiple signals into a single consensus signal.
"""

from enum import Enum

from .signal import Direction, Signal, SignalType


class EnsembleStrategy(Enum):
    """Strategy for combining signals."""

    VOTING = "voting"  # Majority vote
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by model score/confidence
    HIGHEST_CONFIDENCE = "highest_confidence"  # Take signal with highest probability
    IC_WEIGHTED = "ic_weighted"  # Information Coefficient weighted (Phase 4)
    VOL_ADJUSTED = "vol_adjusted"  # Volatility-adjusted weighting (Phase 4)
    REGRESSION = "regression"  # Regression-based optimization (Phase 4)


class SignalEnsemble:
    """
    Combines multiple signals into a consensus signal.
    """

    @staticmethod
    def combine(
        signals: list[Signal], strategy: EnsembleStrategy = EnsembleStrategy.VOTING
    ) -> Signal | None:
        """
        Combine a list of signals into a single signal.

        Args:
            signals: List of signals to combine
            strategy: Ensemble strategy to use

        Returns:
            Consensus Signal or None if no consensus
        """
        if not signals:
            return None

        if len(signals) == 1:
            return signals[0]

        # Filter out HOLD/NEUTRAL signals for voting if we want active signals
        active_signals = [s for s in signals if s.direction != Direction.NEUTRAL]

        if not active_signals:
            # If all are neutral, return the first one (or a neutral signal)
            return signals[0]

        if strategy == EnsembleStrategy.HIGHEST_CONFIDENCE:
            return max(active_signals, key=lambda s: s.probability)

        if strategy == EnsembleStrategy.VOTING:
            return SignalEnsemble._voting_ensemble(active_signals)

        if strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return SignalEnsemble._weighted_ensemble(active_signals)

        if strategy == EnsembleStrategy.IC_WEIGHTED:
            return SignalEnsemble._ic_weighted_ensemble(active_signals)

        if strategy == EnsembleStrategy.VOL_ADJUSTED:
            return SignalEnsemble._vol_adjusted_ensemble(active_signals)

        if strategy == EnsembleStrategy.REGRESSION:
            return SignalEnsemble._regression_ensemble(active_signals)

        return signals[0]

    @staticmethod
    def _voting_ensemble(signals: list[Signal]) -> Signal:
        """Majority vote ensemble."""
        long_votes = sum(1 for s in signals if s.direction == Direction.LONG)
        short_votes = sum(1 for s in signals if s.direction == Direction.SHORT)

        base_signal = signals[0]  # Use metadata from first signal as base

        if long_votes > short_votes:
            direction = Direction.LONG
            consensus_strength = long_votes / len(signals)
        elif short_votes > long_votes:
            direction = Direction.SHORT
            consensus_strength = short_votes / len(signals)
        else:
            direction = Direction.NEUTRAL
            consensus_strength = 0.0

        # Average probability of the winning side
        winning_signals = [s for s in signals if s.direction == direction]
        avg_prob = (
            sum(s.probability for s in winning_signals) / len(winning_signals)
            if winning_signals
            else 0.0
        )

        # Create consensus signal
        # Note: We are creating a new Signal object.
        # We need to be careful about required fields.
        return Signal(
            symbol=base_signal.symbol,
            timestamp=base_signal.timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=avg_prob,
            expected_return=sum(s.expected_return for s in winning_signals) / len(winning_signals)
            if winning_signals
            else 0.0,
            confidence_interval=(0.0, 0.0),  # Simplified
            score=consensus_strength if direction == Direction.LONG else -consensus_strength,
            model_id="ensemble_voting",
            model_version="1.0.0",
            metadata={"vote_count": len(signals), "consensus_strength": consensus_strength},
        )

    @staticmethod
    def _weighted_ensemble(signals: list[Signal]) -> Signal:
        """Weighted average ensemble based on probability."""
        # Simple implementation: Weighted score
        total_score = 0.0
        total_weight = 0.0

        for s in signals:
            weight = s.probability
            score = s.score  # -1 to 1
            if s.direction == Direction.SHORT:
                score = -abs(score)
            elif s.direction == Direction.LONG:
                score = abs(score)

            total_score += score * weight
            total_weight += weight

        avg_score = total_score / total_weight if total_weight > 0 else 0.0

        direction = (
            Direction.LONG
            if avg_score > 0.1
            else (Direction.SHORT if avg_score < -0.1 else Direction.NEUTRAL)
        )

        base_signal = signals[0]

        return Signal(
            symbol=base_signal.symbol,
            timestamp=base_signal.timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=abs(avg_score),  # Proxy for probability
            expected_return=0.0,  # Hard to average
            confidence_interval=(0.0, 0.0),
            score=avg_score,
            model_id="ensemble_weighted",
            model_version="1.0.0",
            metadata={"signal_count": len(signals)},
        )

    @staticmethod
    def _ic_weighted_ensemble(signals: list[Signal]) -> Signal:
        """
        IC-Weighted ensemble (Phase 4).

        Weights models by historical Information Coefficient (IC).
        Uses placeholder uniform weights for now.
        """
        # Placeholder: Uniform weights (would use historical IC in production)
        model_weights = {s.model_id: 1.0 / len(signals) for s in signals}

        total_score = 0.0
        for s in signals:
            weight = model_weights.get(s.model_id, 1.0 / len(signals))
            score = s.score
            total_score += score * weight

        direction = (
            Direction.LONG
            if total_score > 0.1
            else (Direction.SHORT if total_score < -0.1 else Direction.NEUTRAL)
        )
        base_signal = signals[0]

        return Signal(
            symbol=base_signal.symbol,
            timestamp=base_signal.timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=min(0.95, abs(total_score)),
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            score=total_score,
            model_id="ensemble_ic_weighted",
            model_version="1.0.0",
            metadata={"signal_count": len(signals), "weights": model_weights},
        )

    @staticmethod
    def _vol_adjusted_ensemble(signals: list[Signal]) -> Signal:
        """
        Volatility-Adjusted ensemble (Phase 4).

        Downweights signals from volatile/unreliable models.
        Uses inverse probability as proxy for volatility.
        """
        total_score = 0.0
        total_weight = 0.0

        for s in signals:
            # Use probability as signal quality (higher prob = more reliable)
            weight = s.probability
            score = s.score

            total_score += score * weight
            total_weight += weight

        avg_score = total_score / total_weight if total_weight > 0 else 0.0
        direction = (
            Direction.LONG
            if avg_score > 0.1
            else (Direction.SHORT if avg_score < -0.1 else Direction.NEUTRAL)
        )
        base_signal = signals[0]

        return Signal(
            symbol=base_signal.symbol,
            timestamp=base_signal.timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=min(0.95, abs(avg_score)),
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            score=avg_score,
            model_id="ensemble_vol_adjusted",
            model_version="1.0.0",
            metadata={"signal_count": len(signals), "total_weight": total_weight},
        )

    @staticmethod
    def _regression_ensemble(signals: list[Signal]) -> Signal:
        """
        Regression-Based ensemble (Phase 4).

        Optimizes weights using regression (placeholder: equal weights).
        """
        # Placeholder: Simple average (would use Ridge/Lasso in production)
        avg_score = sum(s.score for s in signals) / len(signals)

        direction = (
            Direction.LONG
            if avg_score > 0.1
            else (Direction.SHORT if avg_score < -0.1 else Direction.NEUTRAL)
        )
        base_signal = signals[0]

        return Signal(
            symbol=base_signal.symbol,
            timestamp=base_signal.timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=min(0.95, abs(avg_score)),
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            score=avg_score,
            model_id="ensemble_regression",
            model_version="1.0.0",
            metadata={"signal_count": len(signals), "method": "ridge_placeholder"},
        )
