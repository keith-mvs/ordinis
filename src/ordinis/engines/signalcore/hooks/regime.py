"""
Market Regime governance hook for SignalCore.

Detects market regime and adjusts strategy parameters accordingly.
Integrates with agent-provided context and technical indicators.

Market Regimes:
- TRENDING_UP: Clear upward momentum, favor trend-following
- TRENDING_DOWN: Sustained selling, reduce exposure
- RANGING: Sideways action, favor mean-reversion
- HIGH_VOLATILITY: Elevated VIX, reduce position sizes
- RISK_OFF: Major negative catalyst, defensive positioning
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any

from ordinis.engines.base.hooks import (
    BaseGovernanceHook,
    Decision,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.base.models import AuditRecord, EngineError

_logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state and metadata.

    Attributes:
        regime: Current market regime
        confidence: Confidence in classification (0-1)
        detected_at: When this regime was detected
        key_factors: Factors that led to classification
        position_size_modifier: Suggested position sizing factor
        strategy_hints: Strategy-specific parameter hints
    """

    regime: MarketRegime
    confidence: float
    detected_at: datetime
    key_factors: list[str]
    position_size_modifier: float = 1.0
    strategy_hints: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "key_factors": self.key_factors,
            "position_size_modifier": self.position_size_modifier,
            "strategy_hints": self.strategy_hints or {},
        }


# Default parameter adjustments by regime
REGIME_ADJUSTMENTS: dict[MarketRegime, dict[str, Any]] = {
    MarketRegime.TRENDING_UP: {
        "position_size_modifier": 1.1,
        "rsi_oversold": 25,  # Lower threshold to catch dips
        "atr_stop_mult": 2.0,  # Wider stops for trend riding
        "favor_strategies": ["trend_following", "momentum_breakout"],
    },
    MarketRegime.TRENDING_DOWN: {
        "position_size_modifier": 0.5,
        "rsi_oversold": 35,  # Higher threshold, less aggressive
        "atr_stop_mult": 1.5,  # Tighter stops
        "avoid_strategies": ["momentum_breakout"],
    },
    MarketRegime.RANGING: {
        "position_size_modifier": 1.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "favor_strategies": ["rsi_mean_reversion", "bollinger_bands"],
    },
    MarketRegime.HIGH_VOLATILITY: {
        "position_size_modifier": 0.5,
        "atr_stop_mult": 2.5,  # Much wider stops
        "max_concurrent_positions": 3,  # Reduce exposure
    },
    MarketRegime.RISK_OFF: {
        "position_size_modifier": 0.25,
        "block_new_entries": True,
        "tighten_existing_stops": True,
    },
}


class RegimeHook(BaseGovernanceHook):
    """Market regime detection and strategy adjustment hook.

    This hook:
    1. Detects current market regime from indicators and context
    2. Adjusts strategy parameters based on regime
    3. Can block signals in unfavorable regimes

    Example:
        >>> hook = RegimeHook()
        >>> hook.set_regime(MarketRegime.HIGH_VOLATILITY, confidence=0.8)
        >>> # Signals will now have position sizes reduced by 50%
    """

    def __init__(
        self,
        initial_regime: MarketRegime = MarketRegime.UNKNOWN,
        vix_threshold: float = 20.0,
        regime_persistence_minutes: int = 60,
    ) -> None:
        """Initialize RegimeHook.

        Args:
            initial_regime: Starting regime assumption
            vix_threshold: VIX level above which is HIGH_VOLATILITY
            regime_persistence_minutes: How long a regime persists
        """
        super().__init__("signalcore")
        self._vix_threshold = vix_threshold
        self._regime_persistence = timedelta(minutes=regime_persistence_minutes)

        # Current state
        self._current_state = RegimeState(
            regime=initial_regime,
            confidence=0.5,
            detected_at=datetime.now(UTC),
            key_factors=["initial_assumption"],
        )

        # Context from agents/external sources
        self._agent_context: str = ""
        self._agent_regime: MarketRegime | None = None
        self._agent_update_time: datetime | None = None

    @property
    def current_regime(self) -> MarketRegime:
        """Get current market regime."""
        return self._current_state.regime

    @property
    def regime_state(self) -> RegimeState:
        """Get full regime state."""
        return self._current_state

    def set_regime(
        self,
        regime: MarketRegime,
        confidence: float = 0.8,
        key_factors: list[str] | None = None,
        strategy_hints: dict[str, Any] | None = None,
    ) -> None:
        """Set market regime manually or from agent.

        Args:
            regime: New market regime
            confidence: Confidence level (0-1)
            key_factors: Factors supporting this classification
            strategy_hints: Optional strategy-specific hints
        """
        adjustments = REGIME_ADJUSTMENTS.get(regime, {})
        position_modifier = adjustments.get("position_size_modifier", 1.0)

        self._current_state = RegimeState(
            regime=regime,
            confidence=confidence,
            detected_at=datetime.now(UTC),
            key_factors=key_factors or [],
            position_size_modifier=position_modifier,
            strategy_hints=strategy_hints,
        )

        _logger.info(
            "Regime set to %s (confidence=%.2f, position_modifier=%.2f)",
            regime.value,
            confidence,
            position_modifier,
        )

    def set_from_agent_context(
        self,
        context: str,
        regime_str: str | None = None,
        confidence: float = 0.7,
    ) -> None:
        """Update regime from agent-provided context.

        Args:
            context: Market context description from agent
            regime_str: Regime classification if provided
            confidence: Agent's confidence in classification
        """
        self._agent_context = context
        self._agent_update_time = datetime.now(UTC)

        if regime_str:
            try:
                regime = MarketRegime(regime_str.lower())
                self._agent_regime = regime
                self.set_regime(
                    regime=regime,
                    confidence=confidence,
                    key_factors=[f"Agent context: {context[:100]}"],
                )
            except ValueError:
                _logger.warning(f"Unknown regime from agent: {regime_str}")

    def detect_from_indicators(
        self,
        vix: float | None = None,
        spy_return_5d: float | None = None,
        market_breadth: float | None = None,
    ) -> MarketRegime:
        """Detect regime from technical indicators.

        Args:
            vix: Current VIX level
            spy_return_5d: S&P 500 5-day return
            market_breadth: Advance/decline ratio

        Returns:
            Detected market regime
        """
        key_factors = []
        regime = MarketRegime.UNKNOWN

        # VIX check
        if vix is not None:
            if vix > self._vix_threshold * 1.5:
                regime = MarketRegime.HIGH_VOLATILITY
                key_factors.append(f"VIX elevated at {vix:.1f}")
            elif vix > self._vix_threshold:
                key_factors.append(f"VIX moderately high at {vix:.1f}")

        # Trend check from SPY return
        if spy_return_5d is not None:
            if spy_return_5d > 2.0:
                if regime == MarketRegime.UNKNOWN:
                    regime = MarketRegime.TRENDING_UP
                key_factors.append(f"SPY +{spy_return_5d:.1f}% in 5 days")
            elif spy_return_5d < -2.0:
                if regime == MarketRegime.UNKNOWN:
                    regime = MarketRegime.TRENDING_DOWN
                key_factors.append(f"SPY {spy_return_5d:.1f}% in 5 days")
            elif abs(spy_return_5d) < 1.0:
                if regime == MarketRegime.UNKNOWN:
                    regime = MarketRegime.RANGING
                key_factors.append(f"SPY flat at {spy_return_5d:.1f}%")

        # Market breadth check
        if market_breadth is not None:
            if market_breadth > 2.0:
                key_factors.append(f"Strong breadth: {market_breadth:.1f}")
            elif market_breadth < 0.5:
                key_factors.append(f"Weak breadth: {market_breadth:.1f}")
                if regime == MarketRegime.TRENDING_UP:
                    regime = MarketRegime.RANGING  # Divergence

        if regime != MarketRegime.UNKNOWN:
            self.set_regime(regime, confidence=0.6, key_factors=key_factors)

        return regime

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Apply regime-based adjustments before signal generation.

        Args:
            context: Preflight context with action and parameters

        Returns:
            PreflightResult with decision and adjustments
        """
        # Only check for signal generation actions
        if context.action not in ["generate_signal", "generate_batch", "evaluate_signal"]:
            return PreflightResult(decision=Decision.ALLOW)

        regime = self._current_state.regime
        adjustments = REGIME_ADJUSTMENTS.get(regime, {})

        # Check for blocking conditions
        if adjustments.get("block_new_entries", False):
            return PreflightResult(
                decision=Decision.DENY,
                reason=f"New entries blocked in {regime.value} regime",
                policy_id="regime_hook",
                policy_version=self.policy_version,
            )

        # Check strategy compatibility
        strategy = context.inputs.get("strategy", "")
        avoid_strategies = adjustments.get("avoid_strategies", [])
        if strategy in avoid_strategies:
            return PreflightResult(
                decision=Decision.WARN,
                reason=f"Strategy {strategy} not recommended in {regime.value}",
                policy_id="regime_hook",
                policy_version=self.policy_version,
                warnings=[f"Consider using {adjustments.get('favor_strategies', ['other'])}"],
            )

        # Apply position size modifier
        position_modifier = self._current_state.position_size_modifier
        result_adjustments: dict[str, Any] = {
            "regime": regime.value,
            "regime_confidence": self._current_state.confidence,
            "position_size_modifier": position_modifier,
        }

        # Add strategy hints if available
        if self._current_state.strategy_hints:
            result_adjustments["strategy_hints"] = self._current_state.strategy_hints

        # Add any regime-specific parameter adjustments
        for key in ["rsi_oversold", "rsi_overbought", "atr_stop_mult"]:
            if key in adjustments:
                result_adjustments[key] = adjustments[key]

        decision = Decision.ALLOW
        warnings = []

        if position_modifier < 1.0:
            warnings.append(
                f"Position size reduced to {position_modifier * 100:.0f}% due to {regime.value}"
            )
            decision = Decision.WARN

        return PreflightResult(
            decision=decision,
            reason=f"Regime: {regime.value} (confidence: {self._current_state.confidence:.2f})",
            policy_id="regime_hook",
            policy_version=self.policy_version,
            adjustments=result_adjustments,
            warnings=warnings,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Log regime-related audit events.

        Args:
            record: Audit record
        """
        if record.action in ["generate_signal", "generate_batch"]:
            _logger.debug(
                "RegimeHook audit: action=%s regime=%s modifier=%.2f",
                record.action,
                self._current_state.regime.value,
                self._current_state.position_size_modifier,
            )

    async def on_error(self, error: EngineError) -> None:
        """Handle errors.

        Args:
            error: Engine error
        """
        _logger.warning(
            "RegimeHook error: %s - %s",
            error.code,
            error.message,
        )

    def get_strategy_adjustments(self, strategy: str) -> dict[str, Any]:
        """Get recommended parameter adjustments for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dictionary of parameter adjustments
        """
        base_adjustments = REGIME_ADJUSTMENTS.get(self._current_state.regime, {}).copy()

        # Remove non-parameter keys
        for key in ["favor_strategies", "avoid_strategies", "block_new_entries"]:
            base_adjustments.pop(key, None)

        # Add strategy hints if available
        if self._current_state.strategy_hints:
            strategy_specific = self._current_state.strategy_hints.get(strategy, {})
            base_adjustments.update(strategy_specific)

        return base_adjustments
