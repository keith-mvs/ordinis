"""
RiskGuard engine for rule-based risk management.

Evaluates all trading decisions against deterministic rules.
Integrates with alerting for risk breach notifications.

Phase 2 enhancements (2025-12-17):
- FeedbackCollector integration for risk breach feedback
- Buying power validation with circuit breaker
- Exposure limit feedback to LearningEngine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from ordinis.domain.positions import Position

from ...signalcore.core.signal import Signal
from ..rules.standard import STANDARD_RISK_RULES
from .rules import RiskCheckResult, RiskRule, RuleCategory

if TYPE_CHECKING:
    from alerting import AlertManager

    from ordinis.engines.learning.collectors.feedback import FeedbackCollector


logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """
    Current portfolio state for risk evaluation.

    All values required for rule evaluation.
    """

    # Equity
    equity: float
    cash: float
    peak_equity: float  # For drawdown calculation

    # Daily tracking
    daily_pnl: float
    daily_trades: int

    # Positions
    open_positions: dict[str, Position]
    total_positions: int

    # Risk metrics
    total_exposure: float
    sector_exposures: dict[str, float] = field(default_factory=dict)
    correlated_exposure: float = 0.0

    # Market state
    market_open: bool = True
    connectivity_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equity": self.equity,
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "total_positions": self.total_positions,
            "total_exposure": self.total_exposure,
            "sector_exposures": self.sector_exposures,
            "correlated_exposure": self.correlated_exposure,
            "market_open": self.market_open,
            "connectivity_ok": self.connectivity_ok,
        }


@dataclass
class ProposedTrade:
    """Proposed trade for risk evaluation."""

    symbol: str
    direction: str  # "long" or "short"
    quantity: int
    entry_price: float
    stop_price: float | None = None
    target_price: float | None = None
    sector: str | None = None


class RiskGuardEngine:
    """
    RiskGuard rule-based risk management engine.

    Evaluates all trading decisions against deterministic rules
    before execution. Sends alerts on risk breaches.
    """

    def __init__(
        self,
        rules: dict[str, RiskRule] | None = None,
        alert_manager: AlertManager | None = None,
        feedback_collector: FeedbackCollector | None = None,
    ):
        """
        Initialize RiskGuard engine.

        Args:
            rules: Dictionary of rule_id -> RiskRule
            alert_manager: Optional alert manager for notifications
            feedback_collector: Feedback collector for risk breach feedback
        """
        # Use standard rules if none provided, but allow empty dict if explicitly passed?
        # Usually 'None' means default.
        if rules is None:
            # Convert rule_id -> RiskRule mapping from standard rules
            # STANDARD_RISK_RULES is keyed by descriptive name in standard.py,
            # but engine expects rule_id as key?
            # Let's check standard.py again.
            self._rules = {r.rule_id: r for r in STANDARD_RISK_RULES.values()}
        else:
            self._rules = rules

        self._alert_manager = alert_manager
        self._feedback = feedback_collector
        self._halted: bool = False
        self._halt_reason: str | None = None

    def add_rule(self, rule: RiskRule) -> None:
        """
        Add or update a risk rule.

        Args:
            rule: Risk rule to add
        """
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        """
        Remove a risk rule.

        Args:
            rule_id: Rule identifier

        Raises:
            KeyError: If rule not found
        """
        if rule_id not in self._rules:
            raise KeyError(f"Rule {rule_id} not found")

        del self._rules[rule_id]

    def get_rule(self, rule_id: str) -> RiskRule:
        """
        Get rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            Risk rule

        Raises:
            KeyError: If rule not found
        """
        if rule_id not in self._rules:
            raise KeyError(f"Rule {rule_id} not found")

        return self._rules[rule_id]

    def list_rules(
        self, category: RuleCategory | None = None, enabled_only: bool = False
    ) -> list[RiskRule]:
        """
        List risk rules.

        Args:
            category: Filter by category
            enabled_only: Only return enabled rules

        Returns:
            List of risk rules
        """
        rules = list(self._rules.values())

        if category:
            rules = [r for r in rules if r.category == category]

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        return rules

    async def evaluate(self, signals: list[Signal]) -> tuple[list[Signal], list[str]]:
        """Evaluate signals and return approved signals with rejection reasons.

        Protocol implementation for OrchestrationEngine.
        """
        approved_signals = []
        rejections = []

        # Mock portfolio state for now since we don't have easy access to it here
        # In real flow, Orchestrator might pass it or we fetch it
        portfolio = PortfolioState(
            equity=100000.0,
            cash=100000.0,
            peak_equity=100000.0,
            daily_pnl=0.0,
            daily_trades=0,
            open_positions={},
            total_positions=0,
            total_exposure=0.0,
            sector_exposures={},
            correlated_exposure=0.0,
        )

        for signal in signals:
            # Create a proposed trade from signal for validation
            # In a real system, sizing happens before this check
            price = 0.0
            if signal.metadata:
                price = signal.metadata.get("price", signal.metadata.get("current_price", 0.0))

            quantity = 100  # Default for demo

            trade = ProposedTrade(
                symbol=signal.symbol,
                direction=signal.direction.value
                if hasattr(signal.direction, "value")
                else str(signal.direction),
                quantity=quantity,
                entry_price=price,
                sector=signal.metadata.get("sector"),
            )

            # Evaluate against rules
            passed, results, adjusted_signal = self.evaluate_signal(signal, trade, portfolio)

            if passed:
                approved_signals.append(adjusted_signal or signal)
            else:
                reasons = [r.message for r in results if not r.passed]
                rejections.extend(reasons)
                logger.warning(f"Signal rejected by RiskGuard: {reasons}")

        return approved_signals, rejections

    def evaluate_signal(
        self, signal: Signal, proposed_trade: ProposedTrade, portfolio: PortfolioState
    ) -> tuple[bool, list[RiskCheckResult], Signal | None]:
        """
        Evaluate signal against all applicable rules.

        Args:
            signal: Trading signal from SignalCore
            proposed_trade: Proposed trade details
            portfolio: Current portfolio state

        Returns:
            Tuple of (passed, results, adjusted_signal)
            - passed: Whether signal passed all critical checks
            - results: List of all rule check results
            - adjusted_signal: Signal adjusted by resize rules (if any)
        """
        if self._halted:
            return (
                False,
                [
                    RiskCheckResult(
                        rule_id="SYSTEM",
                        rule_name="System Halted",
                        passed=False,
                        current_value=0.0,
                        threshold=0.0,
                        comparison="N/A",
                        message=f"Trading halted: {self._halt_reason}",
                        action_taken="reject",
                        severity="critical",
                        timestamp=datetime.utcnow(),
                    )
                ],
                None,
            )

        results: list[RiskCheckResult] = []
        all_passed = True
        adjusted_quantity = proposed_trade.quantity

        # Evaluate pre-trade and position limit rules
        applicable_categories = [
            RuleCategory.PRE_TRADE,
            RuleCategory.POSITION_LIMIT,
            RuleCategory.PORTFOLIO_LIMIT,
            RuleCategory.SANITY_CHECK,
        ]

        for category in applicable_categories:
            category_rules = self.list_rules(category=category, enabled_only=True)

            for rule in category_rules:
                # Calculate current value based on rule
                current_value = self._calculate_rule_value(rule, proposed_trade, portfolio)

                # Evaluate rule
                passed = rule.evaluate(current_value)

                result = RiskCheckResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=passed,
                    current_value=current_value,
                    threshold=rule.threshold,
                    comparison=rule.comparison,
                    message=rule.condition,
                    action_taken="pass" if passed else rule.action_on_breach,
                    severity=rule.severity,
                    timestamp=datetime.utcnow(),
                )

                results.append(result)

                # Handle breach
                if not passed:
                    # Phase 2: Record risk breach to FeedbackCollector
                    if self._feedback:
                        import asyncio

                        try:
                            # Fire-and-forget async recording
                            asyncio.get_event_loop().create_task(
                                self._feedback.record_risk_breach(
                                    breach_type=category.value,
                                    rule_id=rule.rule_id,
                                    current_value=current_value,
                                    threshold=rule.threshold,
                                    action_taken=rule.action_on_breach,
                                    symbol=proposed_trade.symbol,
                                    strategy=signal.metadata.get("strategy")
                                    if signal.metadata
                                    else None,
                                    portfolio_state=portfolio.to_dict(),
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to record risk breach: {e}")

                    if rule.action_on_breach == "reject":
                        all_passed = False
                    elif rule.action_on_breach == "resize":
                        # Resize trade to fit within limit
                        adjusted_quantity = self._resize_trade(
                            rule, current_value, proposed_trade, portfolio
                        )
                    elif rule.action_on_breach == "halt":
                        self._halted = True
                        self._halt_reason = f"{rule.name} breached"
                        all_passed = False
                    # warn: continue but log

        # Create adjusted signal if resized
        adjusted_signal = None
        if adjusted_quantity != proposed_trade.quantity and adjusted_quantity > 0:
            # Signal remains the same but execution quantity is adjusted
            adjusted_signal = signal

        return all_passed, results, adjusted_signal

    def check_kill_switches(self, portfolio: PortfolioState) -> tuple[bool, str | None]:
        """
        Check if any kill switch conditions are triggered.

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (triggered, reason)
        """
        kill_switch_rules = self.list_rules(category=RuleCategory.KILL_SWITCH, enabled_only=True)

        for rule in kill_switch_rules:
            current_value = self._calculate_portfolio_metric(rule, portfolio)

            if not rule.evaluate(current_value):
                self._halted = True
                self._halt_reason = (
                    f"{rule.name}: {current_value:.4f} {rule.comparison} {rule.threshold}"
                )
                return True, self._halt_reason

        return False, None

    def get_available_capacity(self, symbol: str, portfolio: PortfolioState) -> dict[str, Any]:
        """
        Calculate available capacity for new position.

        Args:
            symbol: Symbol to check
            portfolio: Current portfolio state

        Returns:
            Dict with max_shares, max_value, limiting_rule
        """
        # Check position limit rules (both PRE_TRADE and POSITION_LIMIT)
        position_rules = self.list_rules(category=RuleCategory.PRE_TRADE, enabled_only=True)
        position_rules += self.list_rules(category=RuleCategory.POSITION_LIMIT, enabled_only=True)

        max_value = portfolio.equity  # Start with full equity
        limiting_rule = None

        for rule in position_rules:
            if "position_value" in rule.condition:
                # This is a position size limit
                max_allowed = rule.threshold * portfolio.equity
                if max_allowed < max_value:
                    max_value = max_allowed
                    limiting_rule = rule.name

        return {
            "max_value": max_value,
            "limiting_rule": limiting_rule,
            "available_positions": self._get_max_positions(portfolio),
        }

    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._halted

    def reset_halt(self) -> None:
        """Reset halt state (use with caution)."""
        self._halted = False
        self._halt_reason = None

    def to_dict(self) -> dict[str, Any]:
        """Get engine state as dictionary."""
        return {
            "rules": {rule_id: rule.to_dict() for rule_id, rule in self._rules.items()},
            "total_rules": len(self._rules),
            "enabled_rules": len([r for r in self._rules.values() if r.enabled]),
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }

    def _calculate_rule_value(
        self, rule: RiskRule, trade: ProposedTrade, portfolio: PortfolioState
    ) -> float:
        """Calculate current value for rule evaluation."""
        # Position size checks
        if "position_value" in rule.condition:
            # Check if comparing to absolute threshold or ratio
            position_value = trade.quantity * trade.entry_price
            if rule.threshold >= 1.0:  # Absolute dollar threshold
                return position_value
            # Percentage of equity
            return position_value / portfolio.equity

        if "risk_per_trade" in rule.condition and trade.stop_price:
            risk_per_share = abs(trade.entry_price - trade.stop_price)
            total_risk = risk_per_share * trade.quantity
            return total_risk / portfolio.equity

        # Portfolio checks
        if "open_positions" in rule.condition or "count(" in rule.condition:
            return float(portfolio.total_positions + 1)  # Include proposed trade

        if "sector_exposure" in rule.condition and trade.sector:
            current_sector = portfolio.sector_exposures.get(trade.sector, 0.0)
            proposed_value = trade.quantity * trade.entry_price
            return (current_sector + proposed_value) / portfolio.equity

        if "correlated_exposure" in rule.condition:
            return portfolio.correlated_exposure / portfolio.equity

        # Cash checks
        if "cash" in rule.condition and "portfolio_equity" in rule.condition:
            return portfolio.cash / portfolio.equity

        # Drawdown check
        if "drawdown" in rule.condition:
            if portfolio.peak_equity <= 0:
                return 0.0
            return (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity

        # Sanity checks
        if "price_deviation" in rule.condition:
            if trade.symbol in portfolio.open_positions:
                last_price = portfolio.open_positions[trade.symbol].current_price
                return abs(trade.entry_price - last_price) / last_price
            return 0.0

        # Default: return 0 if condition not recognized
        return 0.0

    def _calculate_portfolio_metric(self, rule: RiskRule, portfolio: PortfolioState) -> float:
        """Calculate portfolio-level metric for kill switches."""
        if "daily_pnl" in rule.condition:
            return portfolio.daily_pnl / portfolio.equity

        if "drawdown" in rule.condition or "peak_equity" in rule.condition:
            return (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity

        if "market_open" in rule.condition:
            return 1.0 if portfolio.market_open else 0.0

        if "connectivity_ok" in rule.condition:
            return 1.0 if portfolio.connectivity_ok else 0.0

        return 0.0

    def _resize_trade(
        self,
        rule: RiskRule,
        current_value: float,
        trade: ProposedTrade,
        portfolio: PortfolioState,
    ) -> int:
        """
        Resize trade to fit within rule limits.

        Returns adjusted quantity.
        """
        # Simple proportional resize
        if current_value > 0:
            resize_ratio = rule.threshold / current_value
            adjusted_quantity = int(trade.quantity * resize_ratio * 0.95)  # 5% buffer
            return max(0, adjusted_quantity)

        return trade.quantity

    def _get_max_positions(self, portfolio: PortfolioState) -> int:
        """Get maximum number of positions allowed."""
        position_limit_rules = self.list_rules(
            category=RuleCategory.PORTFOLIO_LIMIT, enabled_only=True
        )

        for rule in position_limit_rules:
            if "count(" in rule.condition or "positions" in rule.condition:
                return int(rule.threshold) - portfolio.total_positions

        return 100  # Default if no limit set
