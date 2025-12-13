"""
Tests for rate limiter implementations.

Tests cover:
- Token bucket rate limiter
- Basic acquire operations
- Token refill over time
- Burst capacity limits
- Wait operations
- Stats tracking
"""

import asyncio

import pytest

from ordinis.core.rate_limiter import RateLimitConfig, RateLimiter

# ==================== BASIC TESTS ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_basic_acquire():
    """Test basic token acquisition."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Should be able to acquire immediately
    result = await limiter.acquire()
    assert result is True
    assert limiter.request_count == 1
    assert limiter.rejected_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_multiple_acquire():
    """Test acquiring multiple tokens."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Should be able to acquire multiple tokens
    for i in range(5):
        result = await limiter.acquire()
        assert result is True

    assert limiter.request_count == 5
    assert limiter.rejected_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_burst_limit():
    """Test burst capacity limits."""
    config = RateLimitConfig(requests_per_second=1.0, burst_size=5)
    limiter = RateLimiter(config)

    # Should be able to burst up to capacity
    for i in range(5):
        result = await limiter.acquire()
        assert result is True

    # Next request should fail (capacity exceeded)
    result = await limiter.acquire()
    assert result is False
    assert limiter.rejected_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_token_refill():
    """Test token refill over time."""
    config = RateLimitConfig(requests_per_second=10.0, burst_size=10)
    limiter = RateLimiter(config)

    # Exhaust all tokens
    for _ in range(10):
        await limiter.acquire()

    # Wait for refill (0.2s = 2 tokens at 10/sec)
    await asyncio.sleep(0.2)

    # Should be able to acquire ~2 tokens
    result1 = await limiter.acquire()
    assert result1 is True

    result2 = await limiter.acquire()
    # Might succeed depending on timing
    # Just verify we got at least one token back
    assert limiter.request_count >= 11


# ==================== WAIT TESTS ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_wait():
    """Test waiting for tokens."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=5)
    limiter = RateLimiter(config)

    # Exhaust tokens
    for _ in range(5):
        await limiter.acquire()

    # Wait should eventually succeed
    wait_time = await limiter.wait()
    assert wait_time > 0  # Should have waited some time
    assert limiter.request_count == 6  # One more after wait


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_wait_multiple_tokens():
    """Test waiting for multiple tokens."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=5)
    limiter = RateLimiter(config)

    # Exhaust tokens
    for _ in range(5):
        await limiter.acquire()

    # Wait for 2 tokens
    wait_time = await limiter.wait(tokens=2)
    assert wait_time > 0
    assert limiter.request_count == 6  # Acquired 2 tokens


# ==================== CONCURRENT TESTS ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_concurrent_acquire():
    """Test concurrent token acquisition (thread-safety)."""
    config = RateLimitConfig(requests_per_second=10.0, burst_size=20)
    limiter = RateLimiter(config)

    # Launch 20 concurrent requests
    tasks = [limiter.acquire() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # All should succeed (within capacity)
    assert all(results)
    assert limiter.request_count == 20
    assert limiter.rejected_count == 0

    # Next request should fail
    result = await limiter.acquire()
    assert result is False
    assert limiter.rejected_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_concurrent_mixed():
    """Test concurrent mix of successful and failed acquisitions."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Launch 20 concurrent requests (only 10 should succeed)
    tasks = [limiter.acquire() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    successful = sum(results)
    failed = len(results) - successful

    assert successful == 10  # Burst capacity
    assert failed == 10
    assert limiter.request_count == 10
    assert limiter.rejected_count == 10


# ==================== STATS TESTS ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_stats_tracking():
    """Test stats are tracked correctly."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Some successful requests
    for _ in range(5):
        await limiter.acquire()

    assert limiter.request_count == 5
    assert limiter.rejected_count == 0

    # Exhaust tokens
    for _ in range(5):
        await limiter.acquire()

    # Try to acquire more (should fail)
    for _ in range(3):
        await limiter.acquire()

    assert limiter.request_count == 10
    assert limiter.rejected_count == 3


# ==================== EDGE CASES ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_zero_tokens_request():
    """Test acquiring zero tokens."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Acquiring 0 tokens should succeed
    result = await limiter.acquire(tokens=0)
    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_large_token_request():
    """Test requesting more tokens than capacity."""
    config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
    limiter = RateLimiter(config)

    # Request more than capacity
    result = await limiter.acquire(tokens=20)
    assert result is False
    assert limiter.rejected_count == 1


@pytest.mark.unit
def test_rate_limiter_config_validation():
    """Test rate limiter accepts valid configuration."""
    config = RateLimitConfig(
        requests_per_second=10.0,
        requests_per_minute=600.0,
        burst_size=20,
    )
    limiter = RateLimiter(config)

    assert limiter.config == config
    assert limiter.tokens == 20  # Initialized to burst size
    assert limiter.refill_rate == 10.0


# ==================== PERFORMANCE TESTS ====================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_high_concurrency():
    """Test rate limiter handles high concurrency."""
    config = RateLimitConfig(requests_per_second=100.0, burst_size=100)
    limiter = RateLimiter(config)

    # Launch 100 concurrent requests
    tasks = [limiter.acquire() for _ in range(100)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert sum(results) == 100
    assert limiter.request_count == 100


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_refill_accuracy():
    """Test token refill is accurate over time."""
    config = RateLimitConfig(requests_per_second=10.0, burst_size=10)
    limiter = RateLimiter(config)

    # Exhaust tokens
    for _ in range(10):
        await limiter.acquire()

    # Wait 1 second (should get ~10 tokens back)
    await asyncio.sleep(1.0)

    # Should be able to acquire ~10 more
    successful = 0
    for _ in range(12):
        if await limiter.acquire():
            successful += 1

    # Allow some variance due to timing
    assert 9 <= successful <= 11
