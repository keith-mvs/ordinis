"""
Circuit breaker for FlowRoute execution engine.

Automatically halts trading when error rates or equity drops exceed
configurable thresholds. Supports manual override and auto-recovery.

Phase 4 implementation (2025-12-17).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


class TripReason(Enum):
    """Reason for circuit breaker trip."""

    ERROR_RATE = "error_rate"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    EQUITY_DROP = "equity_drop"
    BUYING_POWER_LOW = "buying_power_low"
    MANUAL = "manual"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker thresholds."""

    # Error rate thresholds
    error_rate_threshold: float = 0.10  # 10% error rate triggers trip
    error_rate_window_seconds: float = 300.0  # 5 minute window

    # Consecutive error thresholds
    consecutive_error_threshold: int = 5  # 5 consecutive errors

    # Equity thresholds
    equity_drop_threshold: float = 0.05  # 5% drop in session

    # Buying power thresholds
    min_buying_power: Decimal = Decimal("1000")

    # Recovery settings
    cooldown_seconds: float = 300.0  # 5 minute cooldown before half-open
    half_open_success_threshold: int = 3  # Successful orders to close

    # Auto-recovery
    auto_recovery: bool = True


@dataclass
class OrderOutcome:
    """Record of order outcome for error rate calculation."""

    timestamp: datetime
    success: bool
    error_type: str | None = None
    symbol: str | None = None


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker operations."""

    total_trips: int = 0
    trips_by_reason: dict[str, int] = field(default_factory=dict)
    total_blocked_orders: int = 0
    last_trip_time: datetime | None = None
    last_reset_time: datetime | None = None
    current_error_rate: float = 0.0
    consecutive_errors: int = 0

    def record_trip(self, reason: TripReason) -> None:
        """Record a circuit trip."""
        self.total_trips += 1
        self.last_trip_time = datetime.utcnow()

        reason_key = reason.value
        self.trips_by_reason[reason_key] = self.trips_by_reason.get(reason_key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trips": self.total_trips,
            "trips_by_reason": self.trips_by_reason,
            "total_blocked_orders": self.total_blocked_orders,
            "last_trip_time": self.last_trip_time.isoformat() if self.last_trip_time else None,
            "last_reset_time": self.last_reset_time.isoformat() if self.last_reset_time else None,
            "current_error_rate": round(self.current_error_rate, 4),
            "consecutive_errors": self.consecutive_errors,
        }


class CircuitBreaker:
    """
    Circuit breaker for order execution.

    Monitors error rates and equity changes to automatically halt
    trading when thresholds are exceeded.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        on_trip: Callable[[TripReason, str], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_trip: Callback when circuit trips (reason, message)
            on_reset: Callback when circuit resets
        """
        self._config = config or CircuitBreakerConfig()
        self._on_trip = on_trip
        self._on_reset = on_reset

        self._state = CircuitState.CLOSED
        self._trip_reason: TripReason | None = None
        self._trip_message: str | None = None
        self._trip_time: datetime | None = None

        # Order outcome tracking
        self._outcomes: deque[OrderOutcome] = deque(maxlen=1000)
        self._consecutive_errors = 0

        # Equity tracking
        self._session_start_equity: Decimal | None = None
        self._current_equity: Decimal = Decimal("0")

        # Half-open state tracking
        self._half_open_successes = 0

        # Recovery task
        self._recovery_task: asyncio.Task | None = None

        # Metrics
        self._metrics = CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (trading halted)."""
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    def can_execute(self) -> tuple[bool, str]:
        """
        Check if order execution is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if self._state == CircuitState.CLOSED:
            return True, ""

        if self._state == CircuitState.OPEN:
            self._metrics.total_blocked_orders += 1
            return False, f"Circuit breaker open: {self._trip_message}"

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited orders in half-open state
            return True, "half_open"

        return False, "Unknown circuit state"

    def record_success(self, symbol: str | None = None) -> None:
        """
        Record successful order execution.

        Args:
            symbol: Symbol of successful order
        """
        self._outcomes.append(
            OrderOutcome(
                timestamp=datetime.utcnow(),
                success=True,
                symbol=symbol,
            )
        )
        self._consecutive_errors = 0
        self._metrics.consecutive_errors = 0

        # Check half-open recovery
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._config.half_open_success_threshold:
                self._close_circuit()

    def record_failure(self, error_type: str, symbol: str | None = None) -> None:
        """
        Record failed order execution.

        Args:
            error_type: Type of error encountered
            symbol: Symbol of failed order
        """
        self._outcomes.append(
            OrderOutcome(
                timestamp=datetime.utcnow(),
                success=False,
                error_type=error_type,
                symbol=symbol,
            )
        )
        self._consecutive_errors += 1
        self._metrics.consecutive_errors = self._consecutive_errors

        # Reset half-open state on failure
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            self._trip_circuit(
                TripReason.ERROR_RATE,
                "Failed during half-open recovery",
            )
            return

        # Check consecutive errors
        if self._consecutive_errors >= self._config.consecutive_error_threshold:
            self._trip_circuit(
                TripReason.CONSECUTIVE_ERRORS,
                f"{self._consecutive_errors} consecutive order failures",
            )
            return

        # Check error rate
        error_rate = self._calculate_error_rate()
        self._metrics.current_error_rate = error_rate

        if error_rate >= self._config.error_rate_threshold:
            self._trip_circuit(
                TripReason.ERROR_RATE,
                f"Error rate {error_rate:.1%} exceeds {self._config.error_rate_threshold:.1%}",
            )

    def update_equity(self, equity: Decimal) -> None:
        """
        Update current equity and check for trip conditions.

        Args:
            equity: Current equity value
        """
        if self._session_start_equity is None:
            self._session_start_equity = equity

        self._current_equity = equity

        # Check equity drop
        if self._session_start_equity > 0:
            drop = (self._session_start_equity - equity) / self._session_start_equity
            if drop >= Decimal(str(self._config.equity_drop_threshold)):
                self._trip_circuit(
                    TripReason.EQUITY_DROP,
                    f"Equity dropped {float(drop):.1%} from session start",
                )

    def update_buying_power(self, buying_power: Decimal) -> None:
        """
        Update buying power and check for trip conditions.

        Args:
            buying_power: Current buying power
        """
        if buying_power < self._config.min_buying_power:
            self._trip_circuit(
                TripReason.BUYING_POWER_LOW,
                f"Buying power ${buying_power:,.2f} below minimum "
                f"${self._config.min_buying_power:,.2f}",
            )

    def trip_manual(self, reason: str) -> None:
        """
        Manually trip the circuit breaker.

        Args:
            reason: Reason for manual trip
        """
        self._trip_circuit(TripReason.MANUAL, f"Manual trip: {reason}")

    def reset(self, force: bool = False) -> bool:
        """
        Reset circuit breaker to closed state.

        Args:
            force: Force reset even if conditions not met

        Returns:
            True if reset successful
        """
        if self._state == CircuitState.CLOSED:
            return True

        if not force and self._state == CircuitState.OPEN:
            # Cannot reset directly from open, must go through half-open
            logger.warning("Cannot reset from OPEN state without force=True")
            return False

        self._close_circuit()
        return True

    def reset_session(self) -> None:
        """
        Reset session tracking (call at start of trading day).

        Clears error history and resets equity baseline.
        """
        self._outcomes.clear()
        self._consecutive_errors = 0
        self._session_start_equity = None
        self._metrics.consecutive_errors = 0
        self._metrics.current_error_rate = 0.0
        logger.info("Circuit breaker session reset")

    def _trip_circuit(self, reason: TripReason, message: str) -> None:
        """Trip the circuit breaker."""
        if self._state == CircuitState.OPEN:
            return  # Already open

        self._state = CircuitState.OPEN
        self._trip_reason = reason
        self._trip_message = message
        self._trip_time = datetime.utcnow()

        self._metrics.record_trip(reason)

        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason.value} - {message}")

        # Notify callback
        if self._on_trip:
            try:
                self._on_trip(reason, message)
            except Exception as e:
                logger.error(f"Trip callback error: {e}")

        # Start recovery timer if auto-recovery enabled
        if self._config.auto_recovery:
            self._schedule_recovery()

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        prev_state = self._state
        self._state = CircuitState.CLOSED
        self._trip_reason = None
        self._trip_message = None
        self._half_open_successes = 0

        self._metrics.last_reset_time = datetime.utcnow()

        logger.info(f"Circuit breaker reset (was {prev_state.value})")

        # Notify callback
        if self._on_reset:
            try:
                self._on_reset()
            except Exception as e:
                logger.error(f"Reset callback error: {e}")

    def _schedule_recovery(self) -> None:
        """Schedule automatic recovery attempt."""
        if self._recovery_task and not self._recovery_task.done():
            return  # Already scheduled

        async def recovery_wait():
            await asyncio.sleep(self._config.cooldown_seconds)

            if self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                logger.info(
                    f"Circuit breaker entering HALF_OPEN state after "
                    f"{self._config.cooldown_seconds}s cooldown"
                )

        try:
            self._recovery_task = asyncio.create_task(recovery_wait())
        except RuntimeError:
            # No event loop running
            pass

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate within window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self._config.error_rate_window_seconds)

        recent = [o for o in self._outcomes if o.timestamp >= cutoff]

        if not recent:
            return 0.0

        failures = sum(1 for o in recent if not o.success)
        return failures / len(recent)

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        # Update current error rate
        self._metrics.current_error_rate = self._calculate_error_rate()
        return self._metrics

    def to_dict(self) -> dict[str, Any]:
        """Get circuit breaker state as dictionary."""
        return {
            "state": self._state.value,
            "is_open": self.is_open,
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
            "trip_message": self._trip_message,
            "trip_time": self._trip_time.isoformat() if self._trip_time else None,
            "consecutive_errors": self._consecutive_errors,
            "error_rate": round(self._calculate_error_rate(), 4),
            "session_start_equity": str(self._session_start_equity)
            if self._session_start_equity
            else None,
            "current_equity": str(self._current_equity),
            "config": {
                "error_rate_threshold": self._config.error_rate_threshold,
                "consecutive_error_threshold": self._config.consecutive_error_threshold,
                "equity_drop_threshold": self._config.equity_drop_threshold,
                "cooldown_seconds": self._config.cooldown_seconds,
                "auto_recovery": self._config.auto_recovery,
            },
            "metrics": self._metrics.to_dict(),
        }
