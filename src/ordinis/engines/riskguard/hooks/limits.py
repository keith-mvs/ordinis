"""
Position limit governance hook for RiskGuard.

Validates position sizes against maximum allocation limits.
"""

from __future__ import annotations

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


class PositionLimitHook(BaseGovernanceHook):
    """Validates position sizes against configurable limits.

    This hook checks proposed trades to ensure they don't exceed
    maximum position size or concentration limits.

    Example:
        >>> hook = PositionLimitHook(
        ...     max_position_pct=0.10,  # 10% max per position
        ...     max_sector_pct=0.30,    # 30% max per sector
        ... )
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        max_concurrent_positions: int = 10,
        portfolio_value: float = 100000.0,
    ) -> None:
        """Initialize PositionLimitHook.

        Args:
            max_position_pct: Maximum position size as fraction of portfolio (0-1)
            max_sector_pct: Maximum sector concentration as fraction (0-1)
            max_concurrent_positions: Maximum number of open positions
            portfolio_value: Current portfolio value for calculations
        """
        super().__init__("riskguard")
        self._max_position_pct = max_position_pct
        self._max_sector_pct = max_sector_pct
        self._max_concurrent = max_concurrent_positions
        self._portfolio_value = portfolio_value
        # Track current positions for concentration checks
        self._current_positions: dict[str, dict[str, Any]] = {}

    def update_portfolio_value(self, value: float) -> None:
        """Update the portfolio value for limit calculations.

        Args:
            value: Current portfolio value
        """
        self._portfolio_value = value

    def update_positions(self, positions: dict[str, dict[str, Any]]) -> None:
        """Update current positions for concentration checks.

        Args:
            positions: Dict of symbol -> position data
        """
        self._current_positions = positions

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Validate position size against limits.

        Args:
            context: Preflight context with proposed trade details

        Returns:
            PreflightResult with decision
        """
        # Only check position-related actions
        if context.action not in ["calculate_position_size", "submit_order", "evaluate_trade"]:
            return PreflightResult(decision=Decision.ALLOW)

        inputs = context.inputs
        position_value = inputs.get("position_value", 0)
        symbol = inputs.get("symbol", "")
        sector = inputs.get("sector", "unknown")

        # Check position size limit
        if self._portfolio_value > 0:
            position_pct = position_value / self._portfolio_value
            if position_pct > self._max_position_pct:
                max_allowed = self._portfolio_value * self._max_position_pct
                return PreflightResult(
                    decision=Decision.DENY,
                    reason=(
                        f"Position ${position_value:,.0f} ({position_pct:.1%}) exceeds "
                        f"max {self._max_position_pct:.0%} (${max_allowed:,.0f})"
                    ),
                    policy_id="position_limit_hook",
                    policy_version=self.policy_version,
                )

        # Check concurrent positions limit
        if len(self._current_positions) >= self._max_concurrent:
            if symbol not in self._current_positions:  # New position
                return PreflightResult(
                    decision=Decision.DENY,
                    reason=(
                        f"Already at max {self._max_concurrent} positions; "
                        f"close existing before opening new"
                    ),
                    policy_id="position_limit_hook",
                    policy_version=self.policy_version,
                )

        # Check sector concentration
        if sector != "unknown" and self._portfolio_value > 0:
            sector_value = sum(
                p.get("market_value", 0)
                for p in self._current_positions.values()
                if p.get("sector") == sector
            )
            new_sector_value = sector_value + position_value
            sector_pct = new_sector_value / self._portfolio_value
            if sector_pct > self._max_sector_pct:
                return PreflightResult(
                    decision=Decision.WARN,
                    reason=(
                        f"Sector '{sector}' concentration {sector_pct:.1%} "
                        f"exceeds {self._max_sector_pct:.0%} limit"
                    ),
                    policy_id="position_limit_hook",
                    policy_version=self.policy_version,
                    warnings=[f"High sector concentration: {sector} at {sector_pct:.1%}"],
                )

        return PreflightResult(
            decision=Decision.ALLOW,
            reason="Position within limits",
            policy_id="position_limit_hook",
            policy_version=self.policy_version,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Log position limit audit events.

        Args:
            record: Audit record
        """
        _logger.debug(
            "PositionLimitHook audit: action=%s decision=%s",
            record.action,
            record.decision,
        )

    async def on_error(self, error: EngineError) -> None:
        """Handle errors.

        Args:
            error: Engine error
        """
        _logger.error(
            "PositionLimitHook error: %s - %s",
            error.code,
            error.message,
        )
