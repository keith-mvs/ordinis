"""
Pydantic models for database rows.

These models provide:
- Type-safe representation of database rows
- Serialization to/from database format
- Conversion to/from domain objects
"""

import json
from typing import Any

from pydantic import BaseModel


class PositionRow(BaseModel):
    """Database row for positions table."""

    id: int | None = None
    symbol: str
    side: str = "FLAT"
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: str | None = None
    last_update: str
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "PositionRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            symbol=row[1],
            side=row[2],
            quantity=row[3],
            avg_cost=row[4],
            current_price=row[5],
            realized_pnl=row[6],
            unrealized_pnl=row[7],
            entry_time=row[8],
            last_update=row[9],
            created_at=row[10],
            updated_at=row[11],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.symbol,
            self.side,
            self.quantity,
            self.avg_cost,
            self.current_price,
            self.realized_pnl,
            self.unrealized_pnl,
            self.entry_time,
            self.last_update,
        )

    def to_update_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for UPDATE (includes symbol for WHERE)."""
        return (
            self.side,
            self.quantity,
            self.avg_cost,
            self.current_price,
            self.realized_pnl,
            self.unrealized_pnl,
            self.entry_time,
            self.last_update,
            self.symbol,
        )


class OrderRow(BaseModel):
    """Database row for orders table."""

    id: int | None = None
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"
    status: str = "created"
    filled_quantity: int = 0
    remaining_quantity: int
    avg_fill_price: float = 0.0
    created_at: str
    submitted_at: str | None = None
    filled_at: str | None = None
    intent_id: str | None = None
    signal_id: str | None = None
    strategy_id: str | None = None
    broker_order_id: str | None = None
    broker_response: str | None = None  # JSON
    error_message: str | None = None
    retry_count: int = 0
    metadata: str | None = None  # JSON
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "OrderRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            order_id=row[1],
            symbol=row[2],
            side=row[3],
            quantity=row[4],
            order_type=row[5],
            limit_price=row[6],
            stop_price=row[7],
            time_in_force=row[8],
            status=row[9],
            filled_quantity=row[10],
            remaining_quantity=row[11],
            avg_fill_price=row[12],
            created_at=row[13],
            submitted_at=row[14],
            filled_at=row[15],
            intent_id=row[16],
            signal_id=row[17],
            strategy_id=row[18],
            broker_order_id=row[19],
            broker_response=row[20],
            error_message=row[21],
            retry_count=row[22],
            metadata=row[23],
            updated_at=row[24],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.order_id,
            self.symbol,
            self.side,
            self.quantity,
            self.order_type,
            self.limit_price,
            self.stop_price,
            self.time_in_force,
            self.status,
            self.filled_quantity,
            self.remaining_quantity,
            self.avg_fill_price,
            self.created_at,
            self.submitted_at,
            self.filled_at,
            self.intent_id,
            self.signal_id,
            self.strategy_id,
            self.broker_order_id,
            self.broker_response,
            self.error_message,
            self.retry_count,
            self.metadata,
        )

    def get_broker_response_dict(self) -> dict[str, Any]:
        """Parse broker_response JSON."""
        if self.broker_response:
            return json.loads(self.broker_response)
        return {}

    def get_metadata_dict(self) -> dict[str, Any]:
        """Parse metadata JSON."""
        if self.metadata:
            return json.loads(self.metadata)
        return {}


class FillRow(BaseModel):
    """Database row for fills table."""

    id: int | None = None
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: str
    latency_ms: float = 0.0
    slippage_bps: float = 0.0
    vs_arrival_bps: float = 0.0
    metadata: str | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "FillRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            fill_id=row[1],
            order_id=row[2],
            symbol=row[3],
            side=row[4],
            quantity=row[5],
            price=row[6],
            commission=row[7],
            timestamp=row[8],
            latency_ms=row[9],
            slippage_bps=row[10],
            vs_arrival_bps=row[11],
            metadata=row[12],
            created_at=row[13],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.fill_id,
            self.order_id,
            self.symbol,
            self.side,
            self.quantity,
            self.price,
            self.commission,
            self.timestamp,
            self.latency_ms,
            self.slippage_bps,
            self.vs_arrival_bps,
            self.metadata,
        )


class TradeRow(BaseModel):
    """Database row for trades table."""

    id: int | None = None
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    duration_seconds: float
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    strategy_id: str | None = None
    metadata: str | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "TradeRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            trade_id=row[1],
            symbol=row[2],
            side=row[3],
            entry_time=row[4],
            exit_time=row[5],
            entry_price=row[6],
            exit_price=row[7],
            quantity=row[8],
            pnl=row[9],
            pnl_pct=row[10],
            commission=row[11],
            duration_seconds=row[12],
            entry_order_id=row[13],
            exit_order_id=row[14],
            strategy_id=row[15],
            metadata=row[16],
            created_at=row[17],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.trade_id,
            self.symbol,
            self.side,
            self.entry_time,
            self.exit_time,
            self.entry_price,
            self.exit_price,
            self.quantity,
            self.pnl,
            self.pnl_pct,
            self.commission,
            self.duration_seconds,
            self.entry_order_id,
            self.exit_order_id,
            self.strategy_id,
            self.metadata,
        )


class SystemStateRow(BaseModel):
    """Database row for system_state table."""

    id: int | None = None
    key: str
    value: str
    value_type: str = "string"
    description: str | None = None
    updated_at: str | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "SystemStateRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            key=row[1],
            value=row[2],
            value_type=row[3],
            description=row[4],
            updated_at=row[5],
            created_at=row[6],
        )

    def get_typed_value(self) -> Any:
        """Get value converted to its declared type."""
        if self.value_type == "bool":
            return self.value.lower() in ("true", "1", "yes")
        if self.value_type == "int":
            return int(self.value)
        if self.value_type == "float":
            return float(self.value)
        if self.value_type == "json":
            return json.loads(self.value)
        return self.value


class PortfolioSnapshotRow(BaseModel):
    """Database row for portfolio_snapshots table."""

    id: int | None = None
    snapshot_date: str
    cash: float
    total_equity: float
    total_position_value: float
    positions_json: str
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "PortfolioSnapshotRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            snapshot_date=row[1],
            cash=row[2],
            total_equity=row[3],
            total_position_value=row[4],
            positions_json=row[5],
            created_at=row[6],
        )

    def get_positions(self) -> list[dict[str, Any]]:
        """Parse positions JSON."""
        return json.loads(self.positions_json)

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.snapshot_date,
            self.cash,
            self.total_equity,
            self.total_position_value,
            self.positions_json,
        )
