"""
Order types and lifecycle management for FlowRoute.

All order states and transitions are tracked for auditability.
"""

from dataclasses import dataclass, field
from typing import Any

from ordinis.domain.enums import OrderSide, OrderType, TimeInForce


@dataclass
class OrderIntent:
    """
    Trading intent from RiskGuard.

    This is what RiskGuard approves - FlowRoute translates to Order.
    """

    intent_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Source tracking
    signal_id: str | None = None
    strategy_id: str | None = None

    # Risk parameters (from RiskGuard)
    max_slippage_pct: float = 0.01
    max_fill_time_seconds: int = 60

    metadata: dict[str, Any] = field(default_factory=dict)
