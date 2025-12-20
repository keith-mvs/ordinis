"""
Regime-Aware Sizing Hook - Adapts position sizing based on market regime.

Integrates SignalCore's RegimeDetector with PortfolioEngine sizing decisions.
Adjusts position sizes based on detected market conditions (trending, choppy,
volatile, etc.) to improve risk-adjusted returns.

Gap Addressed: Previously no regime-aware allocation despite SignalCore
having RegimeDetector capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

from ordinis.engines.base import (
    AuditRecord,
    Decision,
    GovernanceHook,
    PreflightContext,
    PreflightResult,
)

if TYPE_CHECKING:
    from ordinis.engines.signalcore.regime_detector import (
        RegimeDetector,
    )

logger = logging.getLogger(__name__)


# Regime-based sizing multipliers
# Informed by empirical analysis: trending regimes are more profitable
REGIME_SIZING_MULTIPLIERS = {
    "TRENDING": 1.2,  # Increase size in trends
    "VOLATILE_TRENDING": 1.1,  # Slightly increase in volatile trends
    "MEAN_REVERTING": 0.9,  # Slight reduction for mean-reversion
    "CHOPPY": 0.7,  # Reduce size in choppy markets
    "QUIET_CHOPPY": 0.5,  # Significantly reduce in worst regime
    "UNKNOWN": 1.0,  # Neutral when unknown
}

# Regime-based volatility multipliers for risk scaling
REGIME_VOLATILITY_SCALING = {
    "TRENDING": 0.9,  # Lower vol target in trends (let winners run)
    "VOLATILE_TRENDING": 1.1,  # Higher vol due to increased volatility
    "MEAN_REVERTING": 1.0,  # Normal vol target
    "CHOPPY": 1.2,  # Higher vol target = smaller positions
    "QUIET_CHOPPY": 1.3,  # Most conservative
    "UNKNOWN": 1.0,
}


@dataclass
class RegimeAdjustment:
    """Adjustment recommendation based on detected regime.

    Attributes:
        symbol: Asset symbol
        regime: Detected market regime
        confidence: Regime detection confidence
        sizing_multiplier: Recommended sizing multiplier
        volatility_scaling: Volatility target scaling
        recommended_strategies: Strategies suited for this regime
        avoid_strategies: Strategies to avoid
        trade_recommendation: Overall recommendation
    """

    symbol: str
    regime: str
    confidence: float
    sizing_multiplier: float
    volatility_scaling: float
    recommended_strategies: list[str] = field(default_factory=list)
    avoid_strategies: list[str] = field(default_factory=list)
    trade_recommendation: str = "TRADE"  # TRADE, CAUTION, AVOID
    reasoning: str = ""


@dataclass
class RegimeSizingConfig:
    """Configuration for regime-aware sizing.

    Attributes:
        enable_regime_adjustment: Whether to apply regime adjustments
        min_confidence: Minimum regime confidence to apply adjustment
        max_sizing_multiplier: Maximum allowed sizing increase
        min_sizing_multiplier: Minimum allowed sizing decrease
        choppy_regime_threshold: ADX threshold for choppy detection
        avoid_quiet_choppy: Whether to completely avoid quiet_choppy regime
    """

    enable_regime_adjustment: bool = True
    min_confidence: float = 0.6
    max_sizing_multiplier: float = 1.5
    min_sizing_multiplier: float = 0.3
    choppy_regime_threshold: float = 20.0  # ADX < 20 = choppy
    avoid_quiet_choppy: bool = True


class RegimeSizingHook(GovernanceHook):
    """
    Governance hook that adjusts sizing based on market regime.

    Integrates with RegimeDetector to:
    1. Detect current market regime for each symbol
    2. Apply regime-appropriate sizing multipliers
    3. Block trades in unfavorable regimes (optional)
    4. Audit regime-based decisions

    Example:
        >>> from ordinis.engines.signalcore.regime_detector import RegimeDetector
        >>> from ordinis.engines.portfolio.hooks import RegimeSizingHook

        >>> detector = RegimeDetector()
        >>> hook = RegimeSizingHook(
        ...     regime_detector=detector,
        ...     config=RegimeSizingConfig(avoid_quiet_choppy=True),
        ... )

        >>> engine = PortfolioEngine(config, governance_hook=hook)
    """

    def __init__(
        self,
        regime_detector: RegimeDetector | None = None,
        config: RegimeSizingConfig | None = None,
        custom_multipliers: dict[str, float] | None = None,
    ) -> None:
        """Initialize regime sizing hook.

        Args:
            regime_detector: RegimeDetector instance (lazy-loaded if None)
            config: Regime sizing configuration
            custom_multipliers: Custom regime multipliers to override defaults
        """
        self._detector = regime_detector
        self.config = config or RegimeSizingConfig()
        self._multipliers = {**REGIME_SIZING_MULTIPLIERS}
        if custom_multipliers:
            self._multipliers.update(custom_multipliers)

        # Cache of recent regime analyses
        self._regime_cache: dict[str, RegimeAdjustment] = {}
        self._audit_log: list[AuditRecord] = []

    def _get_detector(self) -> RegimeDetector:
        """Lazy-load regime detector."""
        if self._detector is None:
            from ordinis.engines.signalcore.regime_detector import RegimeDetector

            self._detector = RegimeDetector()
        return self._detector

    def analyze_regime(
        self,
        symbol: str,
        price_data: Any,  # pd.DataFrame
        timeframe: str = "5min",
    ) -> RegimeAdjustment:
        """Analyze market regime for a symbol.

        Args:
            symbol: Asset symbol
            price_data: OHLCV price data
            timeframe: Data timeframe

        Returns:
            RegimeAdjustment with sizing recommendations
        """
        detector = self._get_detector()

        try:
            analysis = detector.analyze(price_data, symbol=symbol, timeframe=timeframe)

            regime_name = analysis.regime.value.upper()
            sizing_mult = self._multipliers.get(regime_name, 1.0)
            vol_scaling = REGIME_VOLATILITY_SCALING.get(regime_name, 1.0)

            # Apply config limits
            sizing_mult = max(
                self.config.min_sizing_multiplier,
                min(self.config.max_sizing_multiplier, sizing_mult),
            )

            # Determine trade recommendation
            if regime_name == "QUIET_CHOPPY" and self.config.avoid_quiet_choppy:
                trade_rec = "AVOID"
            elif regime_name in ("CHOPPY", "QUIET_CHOPPY"):
                trade_rec = "CAUTION"
            else:
                trade_rec = "TRADE"

            adjustment = RegimeAdjustment(
                symbol=symbol,
                regime=regime_name,
                confidence=analysis.confidence,
                sizing_multiplier=sizing_mult,
                volatility_scaling=vol_scaling,
                recommended_strategies=analysis.recommended_strategies,
                avoid_strategies=analysis.avoid_strategies,
                trade_recommendation=trade_rec,
                reasoning=analysis.reasoning,
            )

            # Cache the result
            self._regime_cache[symbol] = adjustment

            return adjustment

        except Exception as e:
            logger.warning(f"Regime analysis failed for {symbol}: {e}")
            return RegimeAdjustment(
                symbol=symbol,
                regime="UNKNOWN",
                confidence=0.0,
                sizing_multiplier=1.0,
                volatility_scaling=1.0,
                trade_recommendation="CAUTION",
                reasoning=f"Analysis failed: {e}",
            )

    def get_cached_regime(self, symbol: str) -> RegimeAdjustment | None:
        """Get cached regime analysis for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Cached RegimeAdjustment or None
        """
        return self._regime_cache.get(symbol)

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Apply regime-based preflight checks.

        Checks:
        1. If regime analysis is available for symbols
        2. If sizing adjustments should be applied
        3. If trades should be blocked based on regime

        Args:
            context: Preflight context with operation details

        Returns:
            PreflightResult with regime-based decision
        """
        if not self.config.enable_regime_adjustment:
            return PreflightResult(
                decision=Decision.ALLOW,
                allowed=True,
                reason="Regime adjustment disabled",
            )

        params = context.parameters
        symbol = params.get("symbol")
        symbols = params.get("symbols", [symbol] if symbol else [])

        modifications: dict[str, Any] = {}
        warnings: list[str] = []
        blocked_symbols: list[str] = []

        for sym in symbols:
            if not sym:
                continue

            adjustment = self._regime_cache.get(sym)
            if not adjustment:
                # No cached regime, allow with neutral adjustment
                continue

            # Check confidence threshold
            if adjustment.confidence < self.config.min_confidence:
                warnings.append(f"{sym}: Low regime confidence ({adjustment.confidence:.1%})")
                continue

            # Check if trade should be blocked
            if adjustment.trade_recommendation == "AVOID":
                blocked_symbols.append(sym)
                continue

            # Apply sizing modification
            if adjustment.sizing_multiplier != 1.0:
                if "sizing_adjustments" not in modifications:
                    modifications["sizing_adjustments"] = {}
                modifications["sizing_adjustments"][sym] = {
                    "multiplier": adjustment.sizing_multiplier,
                    "regime": adjustment.regime,
                    "reasoning": adjustment.reasoning,
                }

            if adjustment.trade_recommendation == "CAUTION":
                warnings.append(f"{sym}: {adjustment.regime} regime - reduced sizing recommended")

        # Block if any symbols should be avoided
        if blocked_symbols:
            return PreflightResult(
                decision=Decision.DENY,
                allowed=False,
                reason=f"Unfavorable regime for: {', '.join(blocked_symbols)}. "
                f"Detected: {[self._regime_cache.get(s).regime for s in blocked_symbols if self._regime_cache.get(s)]}",
            )

        # Allow with modifications
        reason = "Regime checks passed"
        if warnings:
            reason += f" (warnings: {'; '.join(warnings)})"

        return PreflightResult(
            decision=Decision.ALLOW,
            allowed=True,
            reason=reason,
            modifications=modifications if modifications else None,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Record audit entry with regime context.

        Args:
            record: Audit record to log
        """
        # Enrich with regime information
        if hasattr(record, "inputs") and isinstance(record.inputs, dict):
            symbol = record.inputs.get("symbol")
            if symbol and symbol in self._regime_cache:
                adjustment = self._regime_cache[symbol]
                if not hasattr(record, "details"):
                    record.details = {}
                record.details["regime"] = adjustment.regime
                record.details["regime_confidence"] = adjustment.confidence
                record.details["sizing_multiplier"] = adjustment.sizing_multiplier

        self._audit_log.append(record)
        logger.debug(f"Regime audit: {record.operation} - {record.status}")

    def get_audit_log(self, limit: int | None = None) -> list[AuditRecord]:
        """Get audit log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of audit records
        """
        if limit:
            return self._audit_log[-limit:]
        return list(self._audit_log)

    def clear_cache(self) -> None:
        """Clear regime analysis cache."""
        self._regime_cache.clear()

    def get_regime_summary(self) -> dict[str, Any]:
        """Get summary of cached regime analyses.

        Returns:
            Summary statistics by regime type
        """
        if not self._regime_cache:
            return {"n_symbols": 0, "by_regime": {}}

        by_regime: dict[str, list[str]] = {}
        for sym, adj in self._regime_cache.items():
            if adj.regime not in by_regime:
                by_regime[adj.regime] = []
            by_regime[adj.regime].append(sym)

        return {
            "n_symbols": len(self._regime_cache),
            "by_regime": {k: len(v) for k, v in by_regime.items()},
            "symbols_by_regime": by_regime,
            "avg_confidence": sum(a.confidence for a in self._regime_cache.values())
            / len(self._regime_cache),
        }


class CombinedRegimeGovernanceHook(GovernanceHook):
    """
    Combined governance hook that integrates regime sizing with other rules.

    Chains RegimeSizingHook with standard portfolio governance rules
    for comprehensive preflight validation.
    """

    def __init__(
        self,
        regime_hook: RegimeSizingHook,
        additional_hooks: list[GovernanceHook] | None = None,
    ) -> None:
        """Initialize combined hook.

        Args:
            regime_hook: Regime sizing hook
            additional_hooks: Other governance hooks to chain
        """
        self._regime_hook = regime_hook
        self._additional_hooks = additional_hooks or []

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Run all hooks in sequence.

        Args:
            context: Preflight context

        Returns:
            Combined preflight result
        """
        # Run regime hook first
        result = await self._regime_hook.preflight(context)
        if not result.allowed:
            return result

        # Apply modifications to context for downstream hooks
        if result.modifications:
            updated_params = {**context.parameters, **result.modifications}
            context = PreflightContext(
                operation=context.operation,
                parameters=updated_params,
                timestamp=context.timestamp,
            )

        # Run additional hooks
        all_modifications = result.modifications or {}
        warnings = []

        for hook in self._additional_hooks:
            hook_result = await hook.preflight(context)
            if not hook_result.allowed:
                return hook_result
            if hook_result.modifications:
                all_modifications.update(hook_result.modifications)
            if "warning" in str(hook_result.reason).lower():
                warnings.append(hook_result.reason)

        reason = "All governance checks passed"
        if warnings:
            reason += f" ({len(warnings)} warnings)"

        return PreflightResult(
            decision=Decision.ALLOW,
            allowed=True,
            reason=reason,
            modifications=all_modifications if all_modifications else None,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Audit to all hooks.

        Args:
            record: Audit record
        """
        await self._regime_hook.audit(record)
        for hook in self._additional_hooks:
            await hook.audit(record)
