"""
Rate limiting infrastructure.

Provides rate limiting for API calls, order submissions, etc.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import logging
import time

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 1.0
    requests_per_minute: float = 60.0
    requests_per_hour: Optional[float] = None
    requests_per_day: Optional[float] = None
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET


@dataclass
class RateLimitState:
    """Current rate limiter state."""
    tokens: float
    last_update: datetime
    request_count: int = 0
    rejected_count: int = 0


class RateLimiter:
    """
    Token bucket rate limiter.

    Provides smooth rate limiting with burst capability.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats
        self.request_count = 0
        self.rejected_count = 0
        self.wait_time_total = 0.0

        # Calculate refill rate (tokens per second)
        self.refill_rate = config.requests_per_second

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens acquired, False if rate limited.
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.request_count += 1
                return True

            self.rejected_count += 1
            return False

    async def wait(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        start_time = time.monotonic()

        while True:
            async with self._lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.request_count += 1
                    wait_time = time.monotonic() - start_time
                    self.wait_time_total += wait_time
                    return wait_time

            # Calculate time to wait for enough tokens
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate

            # Wait with some buffer
            await asyncio.sleep(min(wait_time + 0.01, 1.0))

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)

    def get_state(self) -> RateLimitState:
        """Get current rate limiter state."""
        return RateLimitState(
            tokens=self.tokens,
            last_update=datetime.utcnow(),
            request_count=self.request_count,
            rejected_count=self.rejected_count
        )

    def reset(self) -> None:
        """Reset rate limiter state."""
        self.tokens = float(self.config.burst_size)
        self.last_update = time.monotonic()
        self.request_count = 0
        self.rejected_count = 0
        self.wait_time_total = 0.0


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    More precise than token bucket but uses more memory.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.window_size = 60.0  # 1 minute window
        self.max_requests = int(config.requests_per_minute)
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

        # Stats
        self.request_count = 0
        self.rejected_count = 0

    async def acquire(self) -> bool:
        """Try to acquire a slot."""
        async with self._lock:
            now = time.monotonic()
            self._cleanup(now)

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                self.request_count += 1
                return True

            self.rejected_count += 1
            return False

    async def wait(self) -> float:
        """Wait until a slot is available."""
        start_time = time.monotonic()

        while True:
            if await self.acquire():
                return time.monotonic() - start_time

            # Calculate wait time
            async with self._lock:
                now = time.monotonic()
                self._cleanup(now)

                if self.requests:
                    oldest = self.requests[0]
                    wait_time = (oldest + self.window_size) - now
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time + 0.01, 1.0))
                        continue

            await asyncio.sleep(0.1)

    def _cleanup(self, now: float) -> None:
        """Remove expired requests from window."""
        cutoff = now - self.window_size
        self.requests = [r for r in self.requests if r > cutoff]

    def get_state(self) -> RateLimitState:
        """Get current state."""
        return RateLimitState(
            tokens=self.max_requests - len(self.requests),
            last_update=datetime.utcnow(),
            request_count=self.request_count,
            rejected_count=self.rejected_count
        )


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter for complex rate limiting.

    Supports per-second, per-minute, per-hour, per-day limits.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.limiters: Dict[str, RateLimiter] = {}

        # Create limiter for each tier
        if config.requests_per_second:
            self.limiters["second"] = RateLimiter(RateLimitConfig(
                requests_per_second=config.requests_per_second,
                burst_size=max(1, int(config.requests_per_second * 2))
            ))

        if config.requests_per_minute:
            self.limiters["minute"] = SlidingWindowRateLimiter(RateLimitConfig(
                requests_per_minute=config.requests_per_minute
            ))

    async def acquire(self) -> bool:
        """Acquire from all tiers."""
        for limiter in self.limiters.values():
            if not await limiter.acquire():
                return False
        return True

    async def wait(self) -> float:
        """Wait for all tiers."""
        total_wait = 0.0
        for limiter in self.limiters.values():
            wait_time = await limiter.wait()
            total_wait += wait_time
        return total_wait


class RateLimitedFunction:
    """
    Decorator for rate-limiting async functions.
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter

    def __call__(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            await self.limiter.wait()
            return await func(*args, **kwargs)
        return wrapper


def rate_limited(
    requests_per_second: float = 1.0,
    burst_size: int = 10
) -> Callable:
    """
    Decorator to rate limit an async function.

    Args:
        requests_per_second: Maximum requests per second.
        burst_size: Maximum burst size.

    Returns:
        Decorated function.

    Example:
        @rate_limited(requests_per_second=5, burst_size=10)
        async def api_call():
            ...
    """
    config = RateLimitConfig(
        requests_per_second=requests_per_second,
        burst_size=burst_size
    )
    limiter = RateLimiter(config)

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            await limiter.wait()
            return await func(*args, **kwargs)
        return wrapper

    return decorator


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on response headers.

    Common headers:
    - X-RateLimit-Limit
    - X-RateLimit-Remaining
    - X-RateLimit-Reset
    - Retry-After
    """

    def __init__(self, initial_config: RateLimitConfig):
        self.config = initial_config
        self.limiter = RateLimiter(initial_config)
        self._lock = asyncio.Lock()

        # Track server-reported limits
        self.server_limit: Optional[int] = None
        self.server_remaining: Optional[int] = None
        self.server_reset: Optional[datetime] = None

    async def acquire(self) -> bool:
        """Acquire using adaptive limits."""
        # Check server-reported remaining
        if self.server_remaining is not None and self.server_remaining <= 1:
            if self.server_reset and datetime.utcnow() < self.server_reset:
                return False

        return await self.limiter.acquire()

    async def wait(self) -> float:
        """Wait using adaptive limits."""
        # Check if we should wait for server reset
        if self.server_remaining is not None and self.server_remaining <= 1:
            if self.server_reset:
                wait_time = (self.server_reset - datetime.utcnow()).total_seconds()
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.1f}s for rate limit reset")
                    await asyncio.sleep(wait_time)

        return await self.limiter.wait()

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update limits from response headers."""
        # Parse common rate limit headers
        if "X-RateLimit-Limit" in headers:
            self.server_limit = int(headers["X-RateLimit-Limit"])

        if "X-RateLimit-Remaining" in headers:
            self.server_remaining = int(headers["X-RateLimit-Remaining"])

        if "X-RateLimit-Reset" in headers:
            reset_value = headers["X-RateLimit-Reset"]
            try:
                # Could be Unix timestamp or seconds until reset
                if len(reset_value) > 6:
                    self.server_reset = datetime.fromtimestamp(int(reset_value))
                else:
                    self.server_reset = datetime.utcnow() + timedelta(seconds=int(reset_value))
            except ValueError:
                pass

        if "Retry-After" in headers:
            try:
                retry_after = int(headers["Retry-After"])
                self.server_reset = datetime.utcnow() + timedelta(seconds=retry_after)
                self.server_remaining = 0
            except ValueError:
                pass

        logger.debug(
            f"Rate limit updated: limit={self.server_limit}, "
            f"remaining={self.server_remaining}, reset={self.server_reset}"
        )
