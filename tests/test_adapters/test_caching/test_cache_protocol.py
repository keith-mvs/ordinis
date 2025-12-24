"""Tests for cache protocol module.

Tests cover:
- CacheConfig dataclass
- CacheConfig.get_ttl_for_data_type method
"""

import pytest

from ordinis.adapters.caching.cache_protocol import CacheConfig


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test CacheConfig has sensible defaults."""
        config = CacheConfig()

        assert config.quote_ttl_seconds == 5
        assert config.historical_daily_ttl_seconds == 3600  # 1 hour
        assert config.historical_intraday_ttl_seconds == 300  # 5 minutes
        assert config.company_info_ttl_seconds == 86400  # 24 hours
        assert config.news_ttl_seconds == 900  # 15 minutes
        assert config.max_entries == 10000
        assert config.enabled is True
        assert config.extra == {}

    @pytest.mark.unit
    def test_custom_values(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            quote_ttl_seconds=10,
            historical_daily_ttl_seconds=7200,
            enabled=False,
            extra={"custom": "value"},
        )

        assert config.quote_ttl_seconds == 10
        assert config.historical_daily_ttl_seconds == 7200
        assert config.enabled is False
        assert config.extra == {"custom": "value"}


class TestCacheConfigGetTtl:
    """Tests for CacheConfig.get_ttl_for_data_type method."""

    @pytest.fixture
    def config(self):
        """Create config with known values."""
        return CacheConfig(
            quote_ttl_seconds=5,
            historical_daily_ttl_seconds=3600,
            historical_intraday_ttl_seconds=300,
            company_info_ttl_seconds=86400,
            news_ttl_seconds=900,
        )

    @pytest.mark.unit
    def test_get_ttl_quote(self, config):
        """Test get_ttl_for_data_type returns quote TTL."""
        ttl = config.get_ttl_for_data_type("quote")
        assert ttl == 5

    @pytest.mark.unit
    def test_get_ttl_historical_daily(self, config):
        """Test get_ttl_for_data_type returns historical daily TTL."""
        ttl = config.get_ttl_for_data_type("historical_daily")
        assert ttl == 3600

    @pytest.mark.unit
    def test_get_ttl_historical_intraday(self, config):
        """Test get_ttl_for_data_type returns historical intraday TTL."""
        ttl = config.get_ttl_for_data_type("historical_intraday")
        assert ttl == 300

    @pytest.mark.unit
    def test_get_ttl_company_info(self, config):
        """Test get_ttl_for_data_type returns company info TTL."""
        ttl = config.get_ttl_for_data_type("company_info")
        assert ttl == 86400

    @pytest.mark.unit
    def test_get_ttl_news(self, config):
        """Test get_ttl_for_data_type returns news TTL."""
        ttl = config.get_ttl_for_data_type("news")
        assert ttl == 900

    @pytest.mark.unit
    def test_get_ttl_unknown_type_returns_quote_ttl(self, config):
        """Test get_ttl_for_data_type returns quote TTL for unknown types."""
        ttl = config.get_ttl_for_data_type("unknown_type")
        assert ttl == 5  # Falls back to quote_ttl_seconds

    @pytest.mark.unit
    def test_get_ttl_empty_string_returns_quote_ttl(self, config):
        """Test get_ttl_for_data_type returns quote TTL for empty string."""
        ttl = config.get_ttl_for_data_type("")
        assert ttl == 5
