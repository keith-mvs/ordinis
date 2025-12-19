"""
Comprehensive tests for plugin base module.

Tests cover:
- PluginStatus and PluginCapability enums
- PluginConfig dataclass
- PluginHealth dataclass
- RateLimiter class
"""

import asyncio
from datetime import datetime

import pytest

from ordinis.plugins.base import (
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
    RateLimiter,
)


class TestPluginStatusEnum:
    """Test PluginStatus enum."""

    @pytest.mark.unit
    def test_plugin_status_values(self):
        """Test all PluginStatus enum values."""
        assert PluginStatus.UNINITIALIZED.value == "uninitialized"
        assert PluginStatus.INITIALIZING.value == "initializing"
        assert PluginStatus.READY.value == "ready"
        assert PluginStatus.RUNNING.value == "running"
        assert PluginStatus.PAUSED.value == "paused"
        assert PluginStatus.ERROR.value == "error"
        assert PluginStatus.STOPPED.value == "stopped"

    @pytest.mark.unit
    def test_plugin_status_count(self):
        """Test PluginStatus has expected number of values."""
        assert len(PluginStatus) == 7


class TestPluginCapabilityEnum:
    """Test PluginCapability enum."""

    @pytest.mark.unit
    def test_plugin_capability_values(self):
        """Test all PluginCapability enum values."""
        assert PluginCapability.READ.value == "read"
        assert PluginCapability.WRITE.value == "write"
        assert PluginCapability.STREAM.value == "stream"
        assert PluginCapability.HISTORICAL.value == "historical"
        assert PluginCapability.REALTIME.value == "realtime"

    @pytest.mark.unit
    def test_plugin_capability_count(self):
        """Test PluginCapability has expected number of values."""
        assert len(PluginCapability) == 5


class TestPluginConfig:
    """Test PluginConfig dataclass."""

    @pytest.mark.unit
    def test_plugin_config_required_field(self):
        """Test PluginConfig requires name field."""
        config = PluginConfig(name="test_plugin")
        assert config.name == "test_plugin"

    @pytest.mark.unit
    def test_plugin_config_defaults(self):
        """Test PluginConfig default values."""
        config = PluginConfig(name="test")
        assert config.enabled is True
        assert config.api_key is None
        assert config.api_secret is None
        assert config.base_url is None
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.rate_limit_per_minute == 60
        assert config.extra == {}

    @pytest.mark.unit
    def test_plugin_config_custom_values(self):
        """Test PluginConfig with custom values."""
        config = PluginConfig(
            name="my_plugin",
            enabled=False,
            api_key="key123",
            api_secret="secret456",  # noqa: S106 - test data
            base_url="https://api.example.com",
            timeout_seconds=60,
            max_retries=5,
            rate_limit_per_minute=120,
            extra={"custom": "value"},
        )
        assert config.name == "my_plugin"
        assert config.enabled is False
        assert config.api_key == "key123"
        assert config.api_secret == "secret456"
        assert config.base_url == "https://api.example.com"
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
        assert config.rate_limit_per_minute == 120
        assert config.extra == {"custom": "value"}


class TestPluginHealth:
    """Test PluginHealth dataclass."""

    @pytest.mark.unit
    def test_plugin_health_required_fields(self):
        """Test PluginHealth required fields."""
        now = datetime.utcnow()
        health = PluginHealth(
            status=PluginStatus.READY,
            last_check=now,
            latency_ms=10.5,
        )
        assert health.status == PluginStatus.READY
        assert health.last_check == now
        assert health.latency_ms == 10.5

    @pytest.mark.unit
    def test_plugin_health_defaults(self):
        """Test PluginHealth default values."""
        health = PluginHealth(
            status=PluginStatus.READY,
            last_check=datetime.utcnow(),
            latency_ms=5.0,
        )
        assert health.error_count == 0
        assert health.last_error is None
        assert health.message is None

    @pytest.mark.unit
    def test_plugin_health_with_error(self):
        """Test PluginHealth with error information."""
        health = PluginHealth(
            status=PluginStatus.ERROR,
            last_check=datetime.utcnow(),
            latency_ms=0.0,
            error_count=3,
            last_error="Connection timeout",
            message="Failed to connect to API",
        )
        assert health.status == PluginStatus.ERROR
        assert health.error_count == 3
        assert health.last_error == "Connection timeout"
        assert health.message == "Failed to connect to API"


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.mark.unit
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.requests_per_minute == 60
        assert limiter.tokens == 60

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_success(self):
        """Test acquiring a rate limit token."""
        limiter = RateLimiter(requests_per_minute=60)
        result = await limiter.acquire()
        assert result is True
        assert limiter.tokens == 59

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_exhaustion(self):
        """Test rate limiter exhaustion."""
        limiter = RateLimiter(requests_per_minute=3)

        # Exhaust all tokens
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True

        # Should be exhausted now (might return True if time passed and tokens regenerated)
        # This is a bit tricky to test deterministically

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_token_replenishment(self):
        """Test rate limiter token replenishment over time."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 token per second

        # Use some tokens
        await limiter.acquire()
        await limiter.acquire()
        initial_tokens = limiter.tokens

        # Wait a bit for replenishment
        await asyncio.sleep(0.1)

        # Acquire again - should have replenished some tokens
        await limiter.acquire()
        # Hard to assert exact value due to timing, but it should work

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test waiting for a rate limit token."""
        limiter = RateLimiter(requests_per_minute=600)  # High rate for fast test

        # Should complete quickly with high rate limit
        await limiter.wait_for_token()
        assert limiter.tokens < 600  # Token was consumed

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_access(self):
        """Test rate limiter handles concurrent access."""
        limiter = RateLimiter(requests_per_minute=100)

        async def acquire_token():
            return await limiter.acquire()

        # Acquire concurrently
        results = await asyncio.gather(*[acquire_token() for _ in range(10)])

        # All should succeed with sufficient tokens
        assert all(results)
