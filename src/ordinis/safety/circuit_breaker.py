"""
Circuit Breaker for API connectivity monitoring.

Implements the circuit breaker pattern to:
- Detect API failures
- Prevent cascading failures
- Auto-recover when service is healthy
- Trigger kill switch on sustained failure

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure detected, requests blocked
- HALF_OPEN: Testing if service recovered
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure detected, blocking
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    state_changes: list[tuple[datetime, CircuitState]] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker for API connectivity.

    Monitors API health and transitions between states:
    - CLOSED -> OPEN: When failure threshold exceeded
    - OPEN -> HALF_OPEN: After recovery timeout
    - HALF_OPEN -> CLOSED: On successful call
    - HALF_OPEN -> OPEN: On failed call
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_calls: int = 3,
        on_state_change: Callable[[CircuitState], None] | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the service being monitored
            failure_threshold: Consecutive failures before opening
            success_threshold: Consecutive successes before closing
            recovery_timeout_seconds: Time before testing recovery
            half_open_max_calls: Max calls allowed in half-open state
            on_state_change: Callback for state changes
        """
        self.name = name
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self._half_open_max_calls = half_open_max_calls
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._opened_at: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking)."""
        return self._state == CircuitState.OPEN

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            await self._check_state_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker {self.name} is open. "
                    f"Will retry after {self._recovery_timeout.total_seconds()}s"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit breaker {self.name} is half-open with max calls reached"
                    )
                self._half_open_calls += 1

        # Execute call outside lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(e)
            raise

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self._success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._half_open_calls = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.utcnow()

            logger.warning(
                f"Circuit breaker {self.name} recorded failure "
                f"({self._stats.consecutive_failures}/{self._failure_threshold}): {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
                self._half_open_calls = 0

            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self._failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    async def _check_state_transition(self) -> None:
        """Check if state should transition (e.g., OPEN -> HALF_OPEN)."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = datetime.utcnow() - self._opened_at
            if elapsed >= self._recovery_timeout:
                await self._transition_to(CircuitState.HALF_OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes.append((datetime.utcnow(), new_state))

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.utcnow()
            logger.warning(
                f"Circuit breaker {self.name} OPENED - failures: {self._stats.consecutive_failures}"
            )
        elif new_state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker {self.name} testing recovery (half-open)")
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._stats.consecutive_failures = 0
            logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")

        if self._on_state_change:
            try:
                self._on_state_change(new_state)
            except Exception as e:
                logger.exception(f"Error in circuit breaker state change callback: {e}")

    async def force_open(self, reason: str = "Manual override") -> None:
        """Force circuit to open state."""
        async with self._lock:
            logger.warning(f"Circuit breaker {self.name} force opened: {reason}")
            await self._transition_to(CircuitState.OPEN)

    async def force_close(self) -> None:
        """Force circuit to closed state."""
        async with self._lock:
            logger.info(f"Circuit breaker {self.name} force closed")
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
            await self._transition_to(CircuitState.CLOSED)

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "failure_threshold": self._failure_threshold,
            "success_threshold": self._success_threshold,
            "last_failure": self._stats.last_failure_time.isoformat()
            if self._stats.last_failure_time
            else None,
            "last_success": self._stats.last_success_time.isoformat()
            if self._stats.last_success_time
            else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""


class BrokerCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for broker API.

    Includes additional monitoring for:
    - Order submission failures
    - Quote request failures
    - Account query failures
    """

    def __init__(
        self,
        name: str = "broker_api",
        failure_threshold: int = 3,  # Lower threshold for broker
        success_threshold: int = 2,
        recovery_timeout_seconds: float = 60.0,  # Longer recovery for broker
        on_state_change: Callable[[CircuitState], None] | None = None,
        on_open: Callable[[], None] | None = None,
    ):
        """
        Initialize broker circuit breaker.

        Args:
            name: Name of the broker
            failure_threshold: Failures before opening
            success_threshold: Successes before closing
            recovery_timeout_seconds: Recovery timeout
            on_state_change: State change callback
            on_open: Callback when circuit opens (for kill switch)
        """
        super().__init__(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            on_state_change=on_state_change,
        )
        self._on_open = on_open
        self._order_failures = 0
        self._quote_failures = 0
        self._account_failures = 0

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Override to call on_open callback."""
        await super()._transition_to(new_state)

        if new_state == CircuitState.OPEN and self._on_open:
            try:
                self._on_open()
            except Exception as e:
                logger.exception(f"Error in circuit breaker on_open callback: {e}")

    async def record_order_failure(self, error: Exception) -> None:
        """Record order submission failure."""
        self._order_failures += 1
        await self._record_failure(error)

    async def record_quote_failure(self, error: Exception) -> None:
        """Record quote request failure."""
        self._quote_failures += 1
        await self._record_failure(error)

    async def record_account_failure(self, error: Exception) -> None:
        """Record account query failure."""
        self._account_failures += 1
        await self._record_failure(error)

    def get_status(self) -> dict[str, Any]:
        """Get broker circuit breaker status."""
        status = super().get_status()
        status.update(
            {
                "order_failures": self._order_failures,
                "quote_failures": self._quote_failures,
                "account_failures": self._account_failures,
            }
        )
        return status
