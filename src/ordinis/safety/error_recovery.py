"""
Enhanced Circuit Breaker and Error Recovery System.

Implements production-grade error handling with:
- Multi-level circuit breakers (per-engine, per-operation)
- Adaptive thresholds based on error patterns
- Automatic recovery with gradual ramp-up
- Kill switch integration
- Error categorization and routing

Step 3 of Trade Enhancement Roadmap (P0 Critical).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from ordinis.safety.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = auto()  # Transient errors, retry immediately
    MEDIUM = auto()  # Significant errors, backoff before retry
    HIGH = auto()  # Critical errors, may trip circuit breaker
    CRITICAL = auto()  # Fatal errors, triggers kill switch


class ErrorCategory(Enum):
    """Categorized error types for routing."""
    
    # Network/connectivity
    CONNECTION_TIMEOUT = auto()
    CONNECTION_REFUSED = auto()
    DNS_FAILURE = auto()
    SSL_ERROR = auto()
    
    # API/broker
    RATE_LIMITED = auto()
    AUTHENTICATION_FAILED = auto()
    INSUFFICIENT_FUNDS = auto()
    ORDER_REJECTED = auto()
    POSITION_NOT_FOUND = auto()
    SYMBOL_NOT_FOUND = auto()
    MARKET_CLOSED = auto()
    
    # Data
    INVALID_RESPONSE = auto()
    DATA_CORRUPTION = auto()
    STALE_DATA = auto()
    
    # System
    OUT_OF_MEMORY = auto()
    DISK_FULL = auto()
    THREAD_POOL_EXHAUSTED = auto()
    
    # Unknown
    UNKNOWN = auto()


class RecoveryState(Enum):
    """Recovery state machine states."""
    
    NORMAL = auto()  # Normal operation
    DEGRADED = auto()  # Some features limited
    RECOVERING = auto()  # Testing recovery
    HALTED = auto()  # All operations stopped


@dataclass
class ErrorEvent:
    """Structured error event."""
    
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    operation: str
    engine: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recoverable: bool = True
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.category.name} in {self.engine}.{self.operation}: "
            f"{self.error}"
        )


@dataclass
class CircuitState:
    """State for a single circuit breaker."""
    
    name: str
    state: RecoveryState = RecoveryState.NORMAL
    failure_count: int = 0
    success_count: int = 0
    last_failure: datetime | None = None
    last_success: datetime | None = None
    opened_at: datetime | None = None
    half_open_successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def reset(self) -> None:
        """Reset all counters."""
        self.failure_count = 0
        self.success_count = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_successes = 0


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    
    # Thresholds
    failure_threshold: int = 5  # Consecutive failures to trip
    success_threshold: int = 3  # Consecutive successes to recover
    
    # Timeouts
    base_timeout_seconds: float = 30.0
    max_timeout_seconds: float = 300.0
    timeout_multiplier: float = 2.0
    
    # Rate limiting
    max_calls_per_minute: int = 60
    min_interval_seconds: float = 0.1
    
    # Recovery
    half_open_max_calls: int = 3
    recovery_test_interval_seconds: float = 30.0
    gradual_recovery_steps: int = 5
    gradual_recovery_interval_seconds: float = 10.0
    
    # Integration
    trigger_kill_switch_on_critical: bool = True
    
    def get_timeout(self, failure_count: int) -> float:
        """Calculate exponential backoff timeout."""
        timeout = self.base_timeout_seconds * (self.timeout_multiplier ** min(failure_count, 10))
        return min(timeout, self.max_timeout_seconds)


class ErrorClassifier:
    """Classifies errors into categories and severities."""
    
    # Error pattern mappings
    CATEGORY_PATTERNS: dict[str, ErrorCategory] = {
        "timeout": ErrorCategory.CONNECTION_TIMEOUT,
        "timed out": ErrorCategory.CONNECTION_TIMEOUT,
        "connection refused": ErrorCategory.CONNECTION_REFUSED,
        "dns": ErrorCategory.DNS_FAILURE,
        "ssl": ErrorCategory.SSL_ERROR,
        "certificate": ErrorCategory.SSL_ERROR,
        "rate limit": ErrorCategory.RATE_LIMITED,
        "too many requests": ErrorCategory.RATE_LIMITED,
        "429": ErrorCategory.RATE_LIMITED,
        "unauthorized": ErrorCategory.AUTHENTICATION_FAILED,
        "forbidden": ErrorCategory.AUTHENTICATION_FAILED,
        "401": ErrorCategory.AUTHENTICATION_FAILED,
        "403": ErrorCategory.AUTHENTICATION_FAILED,
        "insufficient": ErrorCategory.INSUFFICIENT_FUNDS,
        "buying power": ErrorCategory.INSUFFICIENT_FUNDS,
        "rejected": ErrorCategory.ORDER_REJECTED,
        "position not found": ErrorCategory.POSITION_NOT_FOUND,
        "symbol not found": ErrorCategory.SYMBOL_NOT_FOUND,
        "market closed": ErrorCategory.MARKET_CLOSED,
        "invalid": ErrorCategory.INVALID_RESPONSE,
        "json": ErrorCategory.INVALID_RESPONSE,
        "parse": ErrorCategory.INVALID_RESPONSE,
        "memory": ErrorCategory.OUT_OF_MEMORY,
        "disk": ErrorCategory.DISK_FULL,
    }
    
    SEVERITY_MAP: dict[ErrorCategory, ErrorSeverity] = {
        ErrorCategory.CONNECTION_TIMEOUT: ErrorSeverity.MEDIUM,
        ErrorCategory.CONNECTION_REFUSED: ErrorSeverity.HIGH,
        ErrorCategory.DNS_FAILURE: ErrorSeverity.HIGH,
        ErrorCategory.SSL_ERROR: ErrorSeverity.HIGH,
        ErrorCategory.RATE_LIMITED: ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHENTICATION_FAILED: ErrorSeverity.CRITICAL,
        ErrorCategory.INSUFFICIENT_FUNDS: ErrorSeverity.HIGH,
        ErrorCategory.ORDER_REJECTED: ErrorSeverity.MEDIUM,
        ErrorCategory.POSITION_NOT_FOUND: ErrorSeverity.LOW,
        ErrorCategory.SYMBOL_NOT_FOUND: ErrorSeverity.LOW,
        ErrorCategory.MARKET_CLOSED: ErrorSeverity.LOW,
        ErrorCategory.INVALID_RESPONSE: ErrorSeverity.MEDIUM,
        ErrorCategory.DATA_CORRUPTION: ErrorSeverity.HIGH,
        ErrorCategory.STALE_DATA: ErrorSeverity.MEDIUM,
        ErrorCategory.OUT_OF_MEMORY: ErrorSeverity.CRITICAL,
        ErrorCategory.DISK_FULL: ErrorSeverity.CRITICAL,
        ErrorCategory.THREAD_POOL_EXHAUSTED: ErrorSeverity.CRITICAL,
        ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
    }
    
    RECOVERABLE: set[ErrorCategory] = {
        ErrorCategory.CONNECTION_TIMEOUT,
        ErrorCategory.CONNECTION_REFUSED,
        ErrorCategory.DNS_FAILURE,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.INSUFFICIENT_FUNDS,
        ErrorCategory.ORDER_REJECTED,
        ErrorCategory.POSITION_NOT_FOUND,
        ErrorCategory.SYMBOL_NOT_FOUND,
        ErrorCategory.MARKET_CLOSED,
        ErrorCategory.INVALID_RESPONSE,
        ErrorCategory.STALE_DATA,
    }
    
    @classmethod
    def classify(cls, error: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
        """
        Classify an error.
        
        Returns:
            (category, severity, recoverable)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check patterns
        for pattern, category in cls.CATEGORY_PATTERNS.items():
            if pattern in error_str or pattern in error_type:
                severity = cls.SEVERITY_MAP.get(category, ErrorSeverity.MEDIUM)
                recoverable = category in cls.RECOVERABLE
                return category, severity, recoverable
                
        # Default
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, True


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with adaptive thresholds and gradual recovery.
    
    Features:
    - Error categorization and severity tracking
    - Adaptive thresholds based on error patterns
    - Gradual recovery with ramp-up
    - Kill switch integration
    - Comprehensive metrics
    
    Example:
        >>> breaker = EnhancedCircuitBreaker("broker_api")
        >>> 
        >>> async with breaker.protected():
        ...     result = await broker.get_account()
    """
    
    def __init__(
        self,
        name: str,
        config: RecoveryConfig | None = None,
        kill_switch: KillSwitch | None = None,
        on_state_change: Callable[[str, RecoveryState, RecoveryState], None] | None = None,
        on_error: Callable[[ErrorEvent], None] | None = None,
    ) -> None:
        """
        Initialize enhanced circuit breaker.
        
        Args:
            name: Breaker name for identification
            config: Recovery configuration
            kill_switch: Optional kill switch integration
            on_state_change: Callback for state transitions
            on_error: Callback for error events
        """
        self.name = name
        self.config = config or RecoveryConfig()
        self._kill_switch = kill_switch
        self._on_state_change = on_state_change
        self._on_error = on_error
        
        self._state = CircuitState(name=name)
        self._lock = asyncio.Lock()
        self._error_history: list[ErrorEvent] = []
        self._recovery_task: asyncio.Task | None = None
        
        # Rate limiting
        self._call_times: list[datetime] = []
        self._recovery_level: float = 1.0  # 0.0 = fully throttled, 1.0 = normal
        
        # Error pattern tracking
        self._error_counts: dict[ErrorCategory, int] = defaultdict(int)
        
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self._state.state in (RecoveryState.HALTED, RecoveryState.RECOVERING)
        
    @property
    def is_healthy(self) -> bool:
        """Check if circuit is in healthy state."""
        return self._state.state == RecoveryState.NORMAL
        
    @property
    def state(self) -> RecoveryState:
        """Get current state."""
        return self._state.state
        
    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        operation: str = "unknown",
        **kwargs: Any,
    ) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            operation: Operation name for error tracking
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            # Check state
            if self._state.state == RecoveryState.HALTED:
                raise CircuitOpenError(
                    f"Circuit breaker {self.name} is halted. Manual intervention required."
                )
                
            # Check if in recovery and limit calls
            if self._state.state == RecoveryState.RECOVERING:
                if self._state.half_open_successes >= self.config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit breaker {self.name} is in recovery. "
                        f"Max test calls ({self.config.half_open_max_calls}) reached."
                    )
                    
            # Rate limiting
            if not self._check_rate_limit():
                raise RateLimitError(f"Rate limit exceeded for {self.name}")
                
        # Execute call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure(e, operation)
            raise
            
    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._state.success_count += 1
            self._state.consecutive_successes += 1
            self._state.consecutive_failures = 0
            self._state.last_success = datetime.utcnow()
            
            if self._state.state == RecoveryState.RECOVERING:
                self._state.half_open_successes += 1
                
                if self._state.half_open_successes >= self.config.success_threshold:
                    await self._transition_to(RecoveryState.NORMAL)
                    self._start_gradual_recovery()
                    
            elif self._state.state == RecoveryState.DEGRADED:
                # Check if enough successes to return to normal
                if self._state.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(RecoveryState.NORMAL)
                    
    async def _record_failure(self, error: Exception, operation: str) -> None:
        """Record failed call."""
        # Classify error
        category, severity, recoverable = ErrorClassifier.classify(error)
        
        event = ErrorEvent(
            error=error,
            category=category,
            severity=severity,
            operation=operation,
            engine=self.name,
            recoverable=recoverable,
        )
        
        async with self._lock:
            self._state.failure_count += 1
            self._state.consecutive_failures += 1
            self._state.consecutive_successes = 0
            self._state.last_failure = datetime.utcnow()
            
            # Track error patterns
            self._error_counts[category] += 1
            self._error_history.append(event)
            
            # Keep history bounded
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-500:]
                
            # Notify callback
            if self._on_error:
                try:
                    self._on_error(event)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
                    
            # Check severity
            if severity == ErrorSeverity.CRITICAL:
                await self._transition_to(RecoveryState.HALTED)
                
                if self.config.trigger_kill_switch_on_critical and self._kill_switch:
                    logger.critical(f"Critical error in {self.name}: {error}")
                    self._kill_switch.activate(
                        reason=f"Critical error: {category.name}",
                        source=self.name,
                    )
                return
                
            # Check if threshold exceeded
            if self._state.consecutive_failures >= self.config.failure_threshold:
                if self._state.state == RecoveryState.NORMAL:
                    await self._transition_to(RecoveryState.DEGRADED)
                    
                    # Schedule recovery test
                    if self._recovery_task is None or self._recovery_task.done():
                        self._recovery_task = asyncio.create_task(
                            self._schedule_recovery_test()
                        )
                        
            # Special handling for specific error types
            if category == ErrorCategory.RATE_LIMITED:
                # Reduce recovery level
                self._recovery_level = max(0.1, self._recovery_level * 0.5)
                
    async def _transition_to(self, new_state: RecoveryState) -> None:
        """Transition to new state."""
        old_state = self._state.state
        
        if old_state == new_state:
            return
            
        self._state.state = new_state
        
        if new_state == RecoveryState.RECOVERING:
            self._state.half_open_successes = 0
            self._state.opened_at = datetime.utcnow()
            
        elif new_state == RecoveryState.NORMAL:
            self._state.reset()
            self._recovery_level = 1.0
            
        logger.info(f"Circuit breaker {self.name}: {old_state.name} -> {new_state.name}")
        
        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")
                
    async def _schedule_recovery_test(self) -> None:
        """Schedule transition to recovery state."""
        timeout = self.config.get_timeout(self._state.failure_count)
        logger.info(f"Circuit breaker {self.name}: testing recovery in {timeout:.1f}s")
        
        await asyncio.sleep(timeout)
        
        async with self._lock:
            if self._state.state == RecoveryState.DEGRADED:
                await self._transition_to(RecoveryState.RECOVERING)
                
    def _start_gradual_recovery(self) -> None:
        """Start gradual recovery process."""
        asyncio.create_task(self._gradual_recovery())
        
    async def _gradual_recovery(self) -> None:
        """Gradually restore full capacity."""
        step_increase = (1.0 - self._recovery_level) / self.config.gradual_recovery_steps
        
        for _ in range(self.config.gradual_recovery_steps):
            await asyncio.sleep(self.config.gradual_recovery_interval_seconds)
            
            async with self._lock:
                if self._state.state != RecoveryState.NORMAL:
                    return  # Recovery interrupted
                    
                self._recovery_level = min(1.0, self._recovery_level + step_increase)
                
        logger.info(f"Circuit breaker {self.name}: gradual recovery complete")
        
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows another call."""
        now = datetime.utcnow()
        
        # Remove old timestamps
        cutoff = now - timedelta(seconds=60)
        self._call_times = [t for t in self._call_times if t > cutoff]
        
        # Apply recovery level throttling
        effective_limit = int(self.config.max_calls_per_minute * self._recovery_level)
        
        if len(self._call_times) >= effective_limit:
            return False
            
        self._call_times.append(now)
        return True
        
    async def force_open(self, reason: str = "Manual override") -> None:
        """Force circuit to open state."""
        async with self._lock:
            logger.warning(f"Circuit breaker {self.name} force opened: {reason}")
            await self._transition_to(RecoveryState.HALTED)
            
    async def force_close(self) -> None:
        """Force circuit to closed state."""
        async with self._lock:
            logger.info(f"Circuit breaker {self.name} force closed")
            await self._transition_to(RecoveryState.NORMAL)
            
    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status."""
        return {
            "name": self.name,
            "state": self._state.state.name,
            "recovery_level": self._recovery_level,
            "total_calls": self._state.success_count + self._state.failure_count,
            "success_count": self._state.success_count,
            "failure_count": self._state.failure_count,
            "consecutive_failures": self._state.consecutive_failures,
            "consecutive_successes": self._state.consecutive_successes,
            "failure_threshold": self.config.failure_threshold,
            "success_threshold": self.config.success_threshold,
            "last_failure": self._state.last_failure.isoformat() if self._state.last_failure else None,
            "last_success": self._state.last_success.isoformat() if self._state.last_success else None,
            "error_categories": dict(self._error_counts),
            "recent_errors": len([
                e for e in self._error_history
                if e.timestamp > datetime.utcnow() - timedelta(minutes=5)
            ]),
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides:
    - Centralized circuit breaker management
    - Global state monitoring
    - Coordinated shutdown
    """
    
    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        on_global_state_change: Callable[[str, RecoveryState], None] | None = None,
    ) -> None:
        """Initialize registry."""
        self._breakers: dict[str, EnhancedCircuitBreaker] = {}
        self._kill_switch = kill_switch
        self._on_global_state_change = on_global_state_change
        
    def get_or_create(
        self,
        name: str,
        config: RecoveryConfig | None = None,
    ) -> EnhancedCircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = EnhancedCircuitBreaker(
                name=name,
                config=config,
                kill_switch=self._kill_switch,
                on_state_change=self._handle_state_change,
            )
        return self._breakers[name]
        
    def get(self, name: str) -> EnhancedCircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
        
    def _handle_state_change(
        self,
        name: str,
        old_state: RecoveryState,
        new_state: RecoveryState,
    ) -> None:
        """Handle state change from any breaker."""
        if self._on_global_state_change:
            self._on_global_state_change(name, new_state)
            
        # Check if any breakers are in critical state
        halted = [n for n, b in self._breakers.items() if b.state == RecoveryState.HALTED]
        if halted:
            logger.warning(f"Breakers in HALTED state: {halted}")
            
    def get_global_status(self) -> dict[str, Any]:
        """Get status of all breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
        
    def get_unhealthy(self) -> list[str]:
        """Get list of unhealthy breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if not breaker.is_healthy
        ]
        
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.force_close()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""


# =============================================================================
# Context Manager for Protected Calls
# =============================================================================


class ProtectedCall:
    """Context manager for circuit-breaker protected calls."""
    
    def __init__(
        self,
        breaker: EnhancedCircuitBreaker,
        operation: str = "unknown",
    ) -> None:
        """Initialize protected call context."""
        self._breaker = breaker
        self._operation = operation
        
    async def __aenter__(self) -> "ProtectedCall":
        """Enter context - check circuit state."""
        if self._breaker.is_open:
            raise CircuitOpenError(
                f"Circuit breaker {self._breaker.name} is open"
            )
        return self
        
    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context - record success or failure."""
        if exc_val is None:
            await self._breaker._record_success()
        else:
            await self._breaker._record_failure(exc_val, self._operation)
            
        return False  # Don't suppress exceptions


# Add method to EnhancedCircuitBreaker
def protected(self, operation: str = "unknown") -> ProtectedCall:
    """Get protected call context manager."""
    return ProtectedCall(self, operation)


EnhancedCircuitBreaker.protected = protected
