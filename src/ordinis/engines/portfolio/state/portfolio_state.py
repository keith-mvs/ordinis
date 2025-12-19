"""
Portfolio State - Transactional state management for PortfolioEngine.

Provides atomic state updates with snapshot/restore capability to address
Gap 2: Inconsistent State Management.

Features:
- Immutable state snapshots
- Transactional updates with commit/rollback
- Optimistic locking for concurrent access
- State validation and integrity checks
- Audit trail for state changes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
import hashlib
import json
import logging
import threading

if TYPE_CHECKING:
    from ordinis.domain.positions import Position

logger = logging.getLogger(__name__)


class StateValidationError(Exception):
    """Raised when state validation fails."""

    pass


class OptimisticLockError(Exception):
    """Raised when concurrent modification detected."""

    pass


class TransactionState(Enum):
    """Transaction lifecycle states."""

    IDLE = auto()  # No active transaction
    PENDING = auto()  # Transaction started, not yet committed
    COMMITTED = auto()  # Transaction committed successfully
    ROLLED_BACK = auto()  # Transaction rolled back


@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable snapshot of a position.

    Frozen dataclass ensures immutability.
    """

    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    side: str  # 'LONG', 'SHORT', 'FLAT'
    market_value: Decimal
    unrealized_pnl: Decimal
    initial_margin: Decimal = Decimal("0")
    multiplier: Decimal = Decimal("1")

    @classmethod
    def from_position(cls, position: "Position") -> PositionSnapshot:
        """Create snapshot from Position object.

        Args:
            position: Position to snapshot

        Returns:
            Immutable PositionSnapshot
        """
        return cls(
            symbol=position.symbol,
            quantity=Decimal(str(position.quantity)),
            avg_entry_price=Decimal(str(position.avg_entry_price)),
            current_price=Decimal(str(position.current_price)),
            side=position.side.value if hasattr(position.side, "value") else str(position.side),
            market_value=Decimal(str(position.market_value)),
            unrealized_pnl=Decimal(str(position.unrealized_pnl)),
            initial_margin=Decimal(str(getattr(position, "initial_margin", 0))),
            multiplier=Decimal(str(getattr(position, "multiplier", 1))),
        )


@dataclass(frozen=True)
class PortfolioStateSnapshot:
    """Immutable snapshot of complete portfolio state.

    Captures all state required for transaction rollback.
    """

    # Identifiers
    snapshot_id: str
    timestamp: datetime
    version: int

    # Core state
    cash: Decimal
    equity: Decimal
    margin_used: Decimal
    positions: tuple[PositionSnapshot, ...]  # Immutable tuple

    # Derived metrics
    total_market_value: Decimal
    total_unrealized_pnl: Decimal
    position_count: int

    # Validation hash
    state_hash: str

    @classmethod
    def create(
        cls,
        cash: Decimal,
        equity: Decimal,
        margin_used: Decimal,
        positions: dict[str, "Position"],
        version: int = 0,
    ) -> PortfolioStateSnapshot:
        """Create a new state snapshot.

        Args:
            cash: Current cash balance
            equity: Total equity
            margin_used: Margin in use
            positions: Dictionary of positions
            version: State version number

        Returns:
            Immutable PortfolioStateSnapshot
        """
        # Convert positions to immutable snapshots
        pos_snapshots = tuple(
            PositionSnapshot.from_position(p) for p in positions.values()
        )

        # Calculate derived metrics
        total_market_value = sum((p.market_value for p in pos_snapshots), Decimal("0"))
        total_unrealized_pnl = sum((p.unrealized_pnl for p in pos_snapshots), Decimal("0"))

        # Generate unique snapshot ID
        snapshot_id = f"snap_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"

        # Compute state hash for integrity verification
        state_data = {
            "cash": str(cash),
            "equity": str(equity),
            "margin_used": str(margin_used),
            "positions": [
                {"symbol": p.symbol, "qty": str(p.quantity), "price": str(p.current_price)}
                for p in pos_snapshots
            ],
            "version": version,
        }
        state_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return cls(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(UTC),
            version=version,
            cash=cash,
            equity=equity,
            margin_used=margin_used,
            positions=pos_snapshots,
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            position_count=len(pos_snapshots),
            state_hash=state_hash,
        )

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        """Get position snapshot by symbol.

        Args:
            symbol: Symbol to lookup

        Returns:
            PositionSnapshot or None if not found
        """
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "cash": str(self.cash),
            "equity": str(self.equity),
            "margin_used": str(self.margin_used),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": str(p.quantity),
                    "avg_entry_price": str(p.avg_entry_price),
                    "current_price": str(p.current_price),
                    "side": p.side,
                    "market_value": str(p.market_value),
                    "unrealized_pnl": str(p.unrealized_pnl),
                }
                for p in self.positions
            ],
            "total_market_value": str(self.total_market_value),
            "total_unrealized_pnl": str(self.total_unrealized_pnl),
            "position_count": self.position_count,
            "state_hash": self.state_hash,
        }


@dataclass
class StateChange:
    """Record of a state change within a transaction."""

    timestamp: datetime
    change_type: str  # 'cash', 'position_add', 'position_remove', 'position_update'
    details: dict[str, Any]
    old_value: Any = None
    new_value: Any = None


class PortfolioStateManager:
    """Manages portfolio state with transactional semantics.

    Provides:
    - Atomic updates via begin_transaction/commit/rollback
    - Optimistic locking to detect concurrent modifications
    - State snapshots for recovery
    - Audit trail of all changes

    Example:
        >>> manager = PortfolioStateManager(initial_cash=100000.0)
        >>> manager.begin_transaction()
        >>> try:
        ...     manager.update_cash(-5000.0)  # Buy something
        ...     manager.add_position("AAPL", 100, 150.0)
        ...     manager.commit()
        ... except Exception:
        ...     manager.rollback()
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        initial_equity: float | None = None,
    ) -> None:
        """Initialize portfolio state manager.

        Args:
            initial_cash: Starting cash balance
            initial_equity: Starting equity (defaults to cash)
        """
        self._lock = threading.RLock()

        # Core state
        self._cash = Decimal(str(initial_cash))
        self._equity = Decimal(str(initial_equity if initial_equity is not None else initial_cash))
        self._margin_used = Decimal("0")
        self._positions: dict[str, dict[str, Any]] = {}

        # Version for optimistic locking
        self._version = 0

        # Transaction state
        self._transaction_state = TransactionState.IDLE
        self._transaction_snapshot: PortfolioStateSnapshot | None = None
        self._pending_changes: list[StateChange] = []

        # History
        self._snapshot_history: list[PortfolioStateSnapshot] = []
        self._max_history = 100

    @property
    def cash(self) -> Decimal:
        """Current cash balance."""
        return self._cash

    @property
    def equity(self) -> Decimal:
        """Current total equity."""
        return self._equity

    @property
    def margin_used(self) -> Decimal:
        """Current margin in use."""
        return self._margin_used

    @property
    def version(self) -> int:
        """Current state version."""
        return self._version

    @property
    def in_transaction(self) -> bool:
        """Whether a transaction is active."""
        return self._transaction_state == TransactionState.PENDING

    def get_snapshot(self) -> PortfolioStateSnapshot:
        """Get current state as immutable snapshot.

        Returns:
            Current PortfolioStateSnapshot
        """
        # Convert internal position dicts to mock Position objects
        from ordinis.domain.positions import Position, PositionSide

        positions = {}
        for symbol, data in self._positions.items():
            pos = Position(symbol=symbol)
            pos.quantity = float(data.get("quantity", 0))
            pos.avg_entry_price = float(data.get("avg_entry_price", 0))
            pos.current_price = float(data.get("current_price", 0))
            pos.side = PositionSide(data.get("side", "FLAT"))
            pos.multiplier = float(data.get("multiplier", 1))
            positions[symbol] = pos

        return PortfolioStateSnapshot.create(
            cash=self._cash,
            equity=self._equity,
            margin_used=self._margin_used,
            positions=positions,
            version=self._version,
        )

    def begin_transaction(self) -> None:
        """Begin a new transaction.

        Captures current state snapshot for potential rollback.

        Raises:
            RuntimeError: If transaction already in progress
        """
        with self._lock:
            if self._transaction_state == TransactionState.PENDING:
                raise RuntimeError("Transaction already in progress")

            self._transaction_snapshot = self.get_snapshot()
            self._pending_changes = []
            self._transaction_state = TransactionState.PENDING

            logger.debug(f"Transaction started at version {self._version}")

    def commit(self) -> PortfolioStateSnapshot:
        """Commit the current transaction.

        Increments version and saves snapshot to history.

        Returns:
            New state snapshot after commit

        Raises:
            RuntimeError: If no transaction in progress
        """
        with self._lock:
            if self._transaction_state != TransactionState.PENDING:
                raise RuntimeError("No transaction in progress")

            # Validate state before commit
            self._validate_state()

            # Increment version
            self._version += 1

            # Save snapshot to history
            snapshot = self.get_snapshot()
            self._snapshot_history.append(snapshot)
            if len(self._snapshot_history) > self._max_history:
                self._snapshot_history = self._snapshot_history[-self._max_history :]

            # Clear transaction state
            self._transaction_state = TransactionState.COMMITTED
            self._transaction_snapshot = None
            self._pending_changes = []

            logger.debug(f"Transaction committed, new version {self._version}")
            return snapshot

    def rollback(self) -> None:
        """Rollback the current transaction.

        Restores state from transaction start snapshot.

        Raises:
            RuntimeError: If no transaction in progress
        """
        with self._lock:
            if self._transaction_state != TransactionState.PENDING:
                raise RuntimeError("No transaction in progress")

            if self._transaction_snapshot is None:
                raise RuntimeError("No snapshot available for rollback")

            # Restore from snapshot
            snap = self._transaction_snapshot
            self._cash = snap.cash
            self._equity = snap.equity
            self._margin_used = snap.margin_used

            # Restore positions
            self._positions.clear()
            for pos in snap.positions:
                self._positions[pos.symbol] = {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "side": pos.side,
                    "multiplier": pos.multiplier,
                    "initial_margin": pos.initial_margin,
                }

            # Clear transaction state
            self._transaction_state = TransactionState.ROLLED_BACK
            self._transaction_snapshot = None
            changes_rolled_back = len(self._pending_changes)
            self._pending_changes = []

            logger.info(f"Transaction rolled back, {changes_rolled_back} changes reverted")

    def update_cash(self, delta: float) -> None:
        """Update cash balance.

        Args:
            delta: Amount to add (positive) or subtract (negative)

        Raises:
            StateValidationError: If result would be negative
        """
        with self._lock:
            old_value = self._cash
            new_value = self._cash + Decimal(str(delta))

            if new_value < 0:
                raise StateValidationError(
                    f"Cash update would result in negative balance: {new_value}"
                )

            self._cash = new_value
            self._record_change("cash", {"delta": delta}, old_value, new_value)

    def set_cash(self, value: float) -> None:
        """Set cash balance to specific value.

        Args:
            value: New cash balance
        """
        with self._lock:
            old_value = self._cash
            self._cash = Decimal(str(value))
            self._record_change("cash_set", {"value": value}, old_value, self._cash)

    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str = "LONG",
        multiplier: float = 1.0,
    ) -> None:
        """Add or update a position.

        Args:
            symbol: Position symbol
            quantity: Number of shares/contracts
            price: Entry/current price
            side: 'LONG' or 'SHORT'
            multiplier: Contract multiplier
        """
        with self._lock:
            old_value = self._positions.get(symbol)

            if symbol in self._positions:
                # Update existing position (average in)
                existing = self._positions[symbol]
                old_qty = float(existing["quantity"])
                old_price = float(existing["avg_entry_price"])
                new_qty = old_qty + quantity

                if new_qty != 0:
                    new_avg = (old_qty * old_price + quantity * price) / new_qty
                else:
                    new_avg = 0.0

                self._positions[symbol] = {
                    "quantity": Decimal(str(new_qty)),
                    "avg_entry_price": Decimal(str(new_avg)),
                    "current_price": Decimal(str(price)),
                    "side": side,
                    "multiplier": Decimal(str(multiplier)),
                    "initial_margin": Decimal("0"),
                }
            else:
                # New position
                self._positions[symbol] = {
                    "quantity": Decimal(str(quantity)),
                    "avg_entry_price": Decimal(str(price)),
                    "current_price": Decimal(str(price)),
                    "side": side,
                    "multiplier": Decimal(str(multiplier)),
                    "initial_margin": Decimal("0"),
                }

            self._record_change(
                "position_add" if old_value is None else "position_update",
                {"symbol": symbol, "quantity": quantity, "price": price},
                old_value,
                self._positions[symbol],
            )

    def remove_position(self, symbol: str) -> dict[str, Any] | None:
        """Remove a position.

        Args:
            symbol: Position symbol to remove

        Returns:
            Removed position data or None if not found
        """
        with self._lock:
            old_value = self._positions.pop(symbol, None)
            if old_value:
                self._record_change(
                    "position_remove", {"symbol": symbol}, old_value, None
                )
            return old_value

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for positions.

        Args:
            prices: Dictionary of symbol -> current price
        """
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol]["current_price"] = Decimal(str(price))

            self._recalculate_equity()

    def _recalculate_equity(self) -> None:
        """Recalculate equity from cash and position values."""
        long_value = Decimal("0")
        short_value = Decimal("0")

        for pos in self._positions.values():
            qty = pos["quantity"]
            price = pos["current_price"]
            multiplier = pos.get("multiplier", Decimal("1"))
            market_value = qty * price * multiplier

            if pos["side"] == "LONG":
                long_value += market_value
            else:
                short_value += market_value

        self._equity = self._cash + long_value - short_value

    def _validate_state(self) -> None:
        """Validate current state for consistency.

        Raises:
            StateValidationError: If state is invalid
        """
        if self._cash < 0:
            raise StateValidationError(f"Negative cash balance: {self._cash}")

        for symbol, pos in self._positions.items():
            if pos["quantity"] < 0:
                raise StateValidationError(f"Negative quantity for {symbol}")

    def _record_change(
        self,
        change_type: str,
        details: dict[str, Any],
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Record a state change for audit.

        Args:
            change_type: Type of change
            details: Change details
            old_value: Previous value
            new_value: New value
        """
        if self._transaction_state == TransactionState.PENDING:
            self._pending_changes.append(
                StateChange(
                    timestamp=datetime.now(UTC),
                    change_type=change_type,
                    details=details,
                    old_value=old_value,
                    new_value=new_value,
                )
            )

    def get_pending_changes(self) -> list[StateChange]:
        """Get pending changes in current transaction.

        Returns:
            List of StateChange objects
        """
        return list(self._pending_changes)

    def get_history(self, limit: int | None = None) -> list[PortfolioStateSnapshot]:
        """Get state snapshot history.

        Args:
            limit: Maximum snapshots to return

        Returns:
            List of historical snapshots
        """
        if limit:
            return self._snapshot_history[-limit:]
        return list(self._snapshot_history)

    def restore_from_snapshot(
        self, snapshot: PortfolioStateSnapshot, expected_version: int | None = None
    ) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: Snapshot to restore from
            expected_version: If provided, verify current version matches

        Raises:
            OptimisticLockError: If expected_version doesn't match
        """
        with self._lock:
            if expected_version is not None and self._version != expected_version:
                raise OptimisticLockError(
                    f"Version mismatch: expected {expected_version}, got {self._version}"
                )

            self._cash = snapshot.cash
            self._equity = snapshot.equity
            self._margin_used = snapshot.margin_used

            self._positions.clear()
            for pos in snapshot.positions:
                self._positions[pos.symbol] = {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "side": pos.side,
                    "multiplier": pos.multiplier,
                    "initial_margin": pos.initial_margin,
                }

            self._version = snapshot.version

            logger.info(f"State restored from snapshot {snapshot.snapshot_id}")
