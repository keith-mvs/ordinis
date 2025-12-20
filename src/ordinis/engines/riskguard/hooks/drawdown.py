"""
Drawdown governance hook for RiskGuard.

Provides graduated exposure reduction based on current drawdown levels.
Integrates with KillSwitch for emergency halts.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from ordinis.engines.base.hooks import (
    BaseGovernanceHook,
    Decision,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.base.models import AuditRecord, EngineError

if TYPE_CHECKING:
    from ordinis.safety.kill_switch import KillSwitch

_logger = logging.getLogger(__name__)


@dataclass
class DrawdownThreshold:
    """Defines a drawdown threshold and corresponding exposure factor.

    Attributes:
        drawdown_pct: Drawdown percentage that triggers this threshold
        exposure_factor: Factor to multiply position sizes by (0.0-1.0)
        action: Description of the action taken
    """

    drawdown_pct: float
    exposure_factor: float
    action: str

    def __post_init__(self) -> None:
        if not 0 <= self.drawdown_pct <= 100:
            raise ValueError("drawdown_pct must be between 0 and 100")
        if not 0 <= self.exposure_factor <= 1:
            raise ValueError("exposure_factor must be between 0 and 1")


# Default graduated thresholds
DEFAULT_THRESHOLDS = [
    DrawdownThreshold(5.0, 0.75, "Reduce exposure 25%"),
    DrawdownThreshold(10.0, 0.50, "Reduce exposure 50%"),
    DrawdownThreshold(15.0, 0.0, "Halt trading"),
]


class DrawdownHook(BaseGovernanceHook):
    """Graduated exposure reduction based on drawdown.

    This hook checks the current drawdown before position sizing operations
    and either reduces exposure or blocks new entries entirely.

    Example:
        >>> from ordinis.safety.kill_switch import KillSwitch
        >>> from ordinis.engines.riskguard.hooks import DrawdownHook
        >>>
        >>> kill_switch = KillSwitch()
        >>> hook = DrawdownHook(kill_switch)
        >>>
        >>> # Use in RiskGuard engine
        >>> engine = RiskGuardEngine(config, governance_hook=hook)
    """

    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        thresholds: list[DrawdownThreshold] | None = None,
        tracked_actions: list[str] | None = None,
    ) -> None:
        """Initialize DrawdownHook.

        Args:
            kill_switch: KillSwitch instance for drawdown data and emergency halt
            thresholds: List of DrawdownThreshold defining graduated response
            tracked_actions: Actions to apply drawdown checks to
        """
        super().__init__("riskguard")
        self._kill_switch = kill_switch
        self._thresholds = sorted(
            thresholds or DEFAULT_THRESHOLDS,
            key=lambda t: t.drawdown_pct,
            reverse=True,  # Check highest thresholds first
        )
        self._tracked_actions = tracked_actions or [
            "calculate_position_size",
            "evaluate_trade",
            "submit_order",
        ]
        # Track current exposure factor
        self._current_exposure_factor = 1.0

    @property
    def exposure_factor(self) -> float:
        """Get current exposure factor."""
        return self._current_exposure_factor

    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage from KillSwitch or estimate.

        Returns:
            Current drawdown as percentage (0-100)
        """
        if self._kill_switch:
            status = self._kill_switch.get_status()
            # Calculate from peak/current equity if available
            if status.get("limits"):
                # KillSwitch tracks this internally
                peak = getattr(self._kill_switch, "_peak_equity", 0)
                current = getattr(self._kill_switch, "_current_equity", 0)
                if peak > 0:
                    return ((peak - current) / peak) * 100
        return 0.0

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Check drawdown and apply graduated exposure reduction.

        Args:
            context: Preflight context with action and parameters

        Returns:
            PreflightResult with decision and any exposure adjustments
        """
        # Only check for tracked actions
        if context.action not in self._tracked_actions:
            return PreflightResult(decision=Decision.ALLOW)

        # Check if kill switch is already active
        if self._kill_switch and self._kill_switch.is_active:
            return PreflightResult(
                decision=Decision.DENY,
                reason=f"Kill switch active: {self._kill_switch.state.message}",
                policy_id="drawdown_hook",
                policy_version=self.policy_version,
            )

        # Get current drawdown
        drawdown = self._get_current_drawdown()

        # Check against thresholds (highest first)
        for threshold in self._thresholds:
            if drawdown >= threshold.drawdown_pct:
                self._current_exposure_factor = threshold.exposure_factor

                if threshold.exposure_factor == 0.0:
                    # Full halt
                    _logger.warning(
                        "DrawdownHook: Blocking operation due to %.1f%% drawdown",
                        drawdown,
                    )
                    return PreflightResult(
                        decision=Decision.DENY,
                        reason=f"Drawdown {drawdown:.1f}% exceeds halt threshold ({threshold.drawdown_pct}%)",
                        policy_id="drawdown_hook",
                        policy_version=self.policy_version,
                    )
                # Reduce exposure
                _logger.info(
                    "DrawdownHook: Reducing exposure to %.0f%% due to %.1f%% drawdown",
                    threshold.exposure_factor * 100,
                    drawdown,
                )
                return PreflightResult(
                    decision=Decision.WARN,
                    reason=f"Drawdown {drawdown:.1f}%: {threshold.action}",
                    policy_id="drawdown_hook",
                    policy_version=self.policy_version,
                    adjustments={"exposure_factor": threshold.exposure_factor},
                    warnings=[f"Position sizing reduced to {threshold.exposure_factor * 100:.0f}%"],
                )

        # No threshold hit, normal operation
        self._current_exposure_factor = 1.0
        return PreflightResult(
            decision=Decision.ALLOW,
            reason=f"Drawdown {drawdown:.1f}% within acceptable limits",
            policy_id="drawdown_hook",
            policy_version=self.policy_version,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Log drawdown-related audit events.

        Args:
            record: Audit record from operation
        """
        if record.action in self._tracked_actions:
            _logger.debug(
                "DrawdownHook audit: action=%s decision=%s exposure_factor=%.2f",
                record.action,
                record.decision,
                self._current_exposure_factor,
            )

    async def on_error(self, error: EngineError) -> None:
        """Handle errors related to drawdown checks.

        Args:
            error: Error from engine operation
        """
        _logger.error(
            "DrawdownHook error: %s - %s (recoverable=%s)",
            error.code,
            error.message,
            error.recoverable,
        )
