"""
Position reconciliation between local state and broker.

Ensures consistency between:
- Local database positions
- Broker account positions

Handles discrepancies by:
- Alerting on mismatches
- Optionally auto-correcting
- Logging for audit trail
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.adapters.storage.repositories.position import PositionRepository

logger = logging.getLogger(__name__)


class ReconciliationAction(Enum):
    """Actions to take on discrepancy."""

    ALERT_ONLY = "alert_only"
    AUTO_CORRECT = "auto_correct"
    HALT_TRADING = "halt_trading"


class DiscrepancyType(Enum):
    """Types of position discrepancies."""

    QUANTITY_MISMATCH = "quantity_mismatch"
    SIDE_MISMATCH = "side_mismatch"
    MISSING_LOCAL = "missing_local"
    MISSING_BROKER = "missing_broker"
    PRICE_MISMATCH = "price_mismatch"


@dataclass
class PositionDiscrepancy:
    """Details of a position discrepancy."""

    symbol: str
    discrepancy_type: DiscrepancyType
    local_value: Any
    broker_value: Any
    difference: float | None = None
    severity: str = "medium"
    message: str = ""


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    success: bool
    timestamp: datetime
    local_positions: int
    broker_positions: int
    discrepancies: list[PositionDiscrepancy] = field(default_factory=list)
    corrections_made: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies found."""
        return len(self.discrepancies) > 0

    @property
    def has_critical_discrepancies(self) -> bool:
        """Check if any critical discrepancies found."""
        return any(d.severity == "critical" for d in self.discrepancies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "local_positions": self.local_positions,
            "broker_positions": self.broker_positions,
            "discrepancy_count": len(self.discrepancies),
            "critical_discrepancies": self.has_critical_discrepancies,
            "corrections_made": self.corrections_made,
            "errors": self.errors,
            "discrepancies": [
                {
                    "symbol": d.symbol,
                    "type": d.discrepancy_type.value,
                    "local": d.local_value,
                    "broker": d.broker_value,
                    "severity": d.severity,
                    "message": d.message,
                }
                for d in self.discrepancies
            ],
        }


class PositionReconciliation:
    """
    Reconciles local positions with broker positions.

    Compares:
    - Quantity (critical)
    - Side (critical)
    - Average cost (warning)

    Actions:
    - Alert only: Log and notify, no changes
    - Auto correct: Update local to match broker
    - Halt trading: Trigger kill switch on critical mismatch
    """

    def __init__(
        self,
        position_repo: "PositionRepository",
        action: ReconciliationAction = ReconciliationAction.ALERT_ONLY,
        quantity_tolerance: int = 0,
        price_tolerance_pct: float = 1.0,
    ):
        """
        Initialize reconciliation.

        Args:
            position_repo: Position repository
            action: Action to take on discrepancy
            quantity_tolerance: Acceptable quantity difference
            price_tolerance_pct: Acceptable price difference percentage
        """
        self.position_repo = position_repo
        self.action = action
        self.quantity_tolerance = quantity_tolerance
        self.price_tolerance_pct = price_tolerance_pct

    async def reconcile(
        self,
        broker_positions: list[dict[str, Any]],
    ) -> ReconciliationResult:
        """
        Reconcile local positions with broker positions.

        Args:
            broker_positions: List of position dicts from broker
                Expected format: [{"symbol": str, "qty": int, "side": str, "avg_cost": float}]

        Returns:
            ReconciliationResult with discrepancies and actions taken
        """
        timestamp = datetime.utcnow()
        discrepancies: list[PositionDiscrepancy] = []
        errors: list[str] = []
        corrections = 0

        try:
            # Get local positions
            local_positions = await self.position_repo.get_all()
            local_by_symbol = {p.symbol: p for p in local_positions}

            # Build broker position map
            broker_by_symbol = {}
            for bp in broker_positions:
                symbol = bp.get("symbol")
                if symbol:
                    broker_by_symbol[symbol] = bp

            # Check each local position against broker
            for symbol, local_pos in local_by_symbol.items():
                if symbol not in broker_by_symbol:
                    if local_pos.quantity > 0:
                        discrepancies.append(
                            PositionDiscrepancy(
                                symbol=symbol,
                                discrepancy_type=DiscrepancyType.MISSING_BROKER,
                                local_value=local_pos.quantity,
                                broker_value=0,
                                severity="critical",
                                message=f"Local has {local_pos.quantity} shares, broker has none",
                            )
                        )
                else:
                    broker_pos = broker_by_symbol[symbol]
                    disc = self._compare_position(local_pos, broker_pos)
                    if disc:
                        discrepancies.extend(disc)

            # Check for positions in broker not in local
            for symbol, broker_pos in broker_by_symbol.items():
                if symbol not in local_by_symbol:
                    qty = broker_pos.get("qty", 0)
                    if qty != 0:
                        discrepancies.append(
                            PositionDiscrepancy(
                                symbol=symbol,
                                discrepancy_type=DiscrepancyType.MISSING_LOCAL,
                                local_value=0,
                                broker_value=qty,
                                severity="critical",
                                message=f"Broker has {qty} shares, local has none",
                            )
                        )

            # Handle discrepancies based on action
            if discrepancies and self.action == ReconciliationAction.AUTO_CORRECT:
                corrections = await self._auto_correct(broker_positions, errors)

            return ReconciliationResult(
                success=not errors and not any(d.severity == "critical" for d in discrepancies),
                timestamp=timestamp,
                local_positions=len(local_positions),
                broker_positions=len(broker_positions),
                discrepancies=discrepancies,
                corrections_made=corrections,
                errors=errors,
            )

        except Exception as e:
            logger.exception("Reconciliation failed")
            return ReconciliationResult(
                success=False,
                timestamp=timestamp,
                local_positions=0,
                broker_positions=len(broker_positions),
                errors=[str(e)],
            )

    def _compare_position(
        self,
        local: Any,
        broker: dict[str, Any],
    ) -> list[PositionDiscrepancy]:
        """Compare local and broker positions."""
        discrepancies = []
        symbol = local.symbol

        # Compare quantity
        local_qty = local.quantity
        broker_qty = broker.get("qty", 0)
        if abs(local_qty - broker_qty) > self.quantity_tolerance:
            discrepancies.append(
                PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    local_value=local_qty,
                    broker_value=broker_qty,
                    difference=broker_qty - local_qty,
                    severity="critical",
                    message=f"Quantity mismatch: local={local_qty}, broker={broker_qty}",
                )
            )

        # Compare side
        local_side = local.side
        broker_side = broker.get("side", "").upper()
        if broker_qty != 0:
            expected_side = "LONG" if broker_qty > 0 else "SHORT"
            if local_side != expected_side and local_qty != 0:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.SIDE_MISMATCH,
                        local_value=local_side,
                        broker_value=expected_side,
                        severity="critical",
                        message=f"Side mismatch: local={local_side}, broker={expected_side}",
                    )
                )

        # Compare average cost (warning only)
        local_cost = local.avg_cost
        broker_cost = broker.get("avg_cost", 0)
        if local_cost > 0 and broker_cost > 0:
            pct_diff = abs(local_cost - broker_cost) / local_cost * 100
            if pct_diff > self.price_tolerance_pct:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=symbol,
                        discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                        local_value=local_cost,
                        broker_value=broker_cost,
                        difference=broker_cost - local_cost,
                        severity="low",
                        message=f"Avg cost diff: local=${local_cost:.2f}, broker=${broker_cost:.2f} ({pct_diff:.1f}%)",
                    )
                )

        return discrepancies

    async def _auto_correct(
        self,
        broker_positions: list[dict[str, Any]],
        errors: list[str],
    ) -> int:
        """
        Auto-correct local positions to match broker.

        Args:
            broker_positions: Broker position list
            errors: Error list to append to

        Returns:
            Number of corrections made
        """
        corrections = 0

        try:
            from ordinis.adapters.storage.models import PositionRow

            for bp in broker_positions:
                symbol = bp.get("symbol")
                if not symbol:
                    continue

                qty = bp.get("qty", 0)
                avg_cost = bp.get("avg_cost", 0)
                current_price = bp.get("current_price", avg_cost)

                if qty == 0:
                    # Delete position if broker shows zero
                    await self.position_repo.delete(symbol)
                    corrections += 1
                    logger.info(f"Auto-corrected: Deleted {symbol} (broker shows 0)")
                else:
                    # Upsert position to match broker
                    side = "LONG" if qty > 0 else "SHORT"
                    unrealized_pnl = (
                        (current_price - avg_cost) * abs(qty)
                        if side == "LONG"
                        else (avg_cost - current_price) * abs(qty)
                    )

                    position = PositionRow(
                        symbol=symbol,
                        side=side,
                        quantity=abs(qty),
                        avg_cost=avg_cost,
                        current_price=current_price,
                        unrealized_pnl=unrealized_pnl,
                        last_update=datetime.utcnow().isoformat(),
                    )
                    await self.position_repo.upsert(position)
                    corrections += 1
                    logger.info(f"Auto-corrected: Updated {symbol} to {qty} @ ${avg_cost:.2f}")

        except Exception as e:
            errors.append(f"Auto-correct failed: {e}")
            logger.exception("Auto-correct error")

        return corrections

    async def check_single_position(
        self,
        symbol: str,
        broker_qty: int,
        broker_avg_cost: float,
    ) -> PositionDiscrepancy | None:
        """
        Quick check for a single position.

        Args:
            symbol: Stock symbol
            broker_qty: Broker quantity
            broker_avg_cost: Broker average cost

        Returns:
            PositionDiscrepancy if mismatch found, None otherwise
        """
        local = await self.position_repo.get_by_symbol(symbol)

        if local is None:
            if broker_qty != 0:
                return PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=DiscrepancyType.MISSING_LOCAL,
                    local_value=0,
                    broker_value=broker_qty,
                    severity="critical",
                    message=f"Position missing locally: broker has {broker_qty}",
                )
            return None

        if abs(local.quantity - abs(broker_qty)) > self.quantity_tolerance:
            return PositionDiscrepancy(
                symbol=symbol,
                discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                local_value=local.quantity,
                broker_value=broker_qty,
                severity="critical",
                message=f"Quantity mismatch: local={local.quantity}, broker={broker_qty}",
            )

        return None
