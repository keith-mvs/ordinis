"""
Engine Module Unit Tests

Tests for OptionsCoreEngine cache behavior and orchestration logic.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ordinis.engines.optionscore import OptionsCoreEngine, OptionsEngineConfig


@pytest.fixture
def engine_config():
    """Create test engine configuration."""
    return OptionsEngineConfig(
        engine_id="test_engine",
        cache_ttl_seconds=60,
        default_risk_free_rate=0.05,
    )


@pytest.fixture
def mock_polygon():
    """Create mock Polygon plugin."""
    plugin = MagicMock()
    plugin.status = MagicMock()
    plugin.status.value = "ready"
    return plugin


@pytest.fixture
def engine(engine_config, mock_polygon):
    """Create engine instance."""
    return OptionsCoreEngine(engine_config, mock_polygon)


def test_engine_initialization(engine_config, mock_polygon):
    """Test engine initialization."""
    engine = OptionsCoreEngine(engine_config, mock_polygon)

    assert engine.config == engine_config
    assert engine.polygon == mock_polygon
    assert engine.pricing_engine is not None
    assert engine.greeks_calc is not None
    assert engine.enrichment_engine is not None
    assert engine.cache == {}
    assert not engine.initialized


@pytest.mark.asyncio
async def test_initialize_success(engine):
    """Test successful initialization."""
    result = await engine.initialize()

    assert result is True
    assert engine.initialized


@pytest.mark.asyncio
async def test_initialize_already_initialized(engine):
    """Test initializing already initialized engine."""
    await engine.initialize()
    assert engine.initialized

    # Second initialization should still return True
    result = await engine.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_initialize_plugin_not_ready(engine_config):
    """Test initialization with plugin not ready."""
    mock_polygon = MagicMock()
    mock_polygon.status = MagicMock()
    mock_polygon.status.value = "error"

    engine = OptionsCoreEngine(engine_config, mock_polygon)

    with pytest.raises(RuntimeError, match="Polygon plugin not ready"):
        await engine.initialize()


def test_cache_get_set(engine):
    """Test cache get/set operations."""
    # Initially empty
    assert engine._get_cached("test_key") is None

    # Set value
    test_value = {"data": "value"}
    engine._set_cached("test_key", test_value)

    # Get value
    cached = engine._get_cached("test_key")
    assert cached == test_value


def test_cache_expiration(engine_config, mock_polygon):
    """Test cache TTL expiration."""
    # Create engine with very short TTL
    config = OptionsEngineConfig(
        engine_id="test",
        cache_ttl_seconds=1,  # 1 second TTL
    )
    engine = OptionsCoreEngine(config, mock_polygon)

    # Set value
    engine._set_cached("test_key", "value")

    # Should be cached immediately
    assert engine._get_cached("test_key") == "value"

    # Manually expire by modifying cache timestamp
    old_time = datetime.utcnow() - timedelta(seconds=2)
    engine.cache["test_key"] = (old_time, "value")

    # Should return None (expired)
    assert engine._get_cached("test_key") is None

    # Cache should be cleaned up
    assert "test_key" not in engine.cache


def test_clear_cache_all(engine):
    """Test clearing all cache."""
    # Add multiple entries
    engine._set_cached("chain:AAPL:all:all:all", "value1")
    engine._set_cached("chain:MSFT:all:all:all", "value2")
    engine._set_cached("other:key", "value3")

    assert len(engine.cache) == 3

    # Clear all
    engine.clear_cache()

    assert len(engine.cache) == 0


def test_clear_cache_by_symbol(engine):
    """Test clearing cache for specific symbol."""
    # Add multiple entries
    engine._set_cached("chain:AAPL:all:all:all", "value1")
    engine._set_cached("chain:AAPL:2025-01-17:all:all", "value2")
    engine._set_cached("chain:MSFT:all:all:all", "value3")

    assert len(engine.cache) == 3

    # Clear only AAPL
    engine.clear_cache("AAPL")

    assert len(engine.cache) == 1
    assert "chain:MSFT:all:all:all" in engine.cache


def test_cache_stats(engine):
    """Test cache statistics."""
    stats = engine.get_cache_stats()

    assert stats["total_items"] == 0
    assert stats["active_items"] == 0
    assert stats["expired_items"] == 0
    assert stats["ttl_seconds"] == 60

    # Add entries
    engine._set_cached("key1", "value1")
    engine._set_cached("key2", "value2")

    stats = engine.get_cache_stats()
    assert stats["total_items"] == 2
    assert stats["active_items"] == 2

    # Manually expire one entry
    old_time = datetime.utcnow() - timedelta(seconds=120)
    engine.cache["key1"] = (old_time, "value1")

    stats = engine.get_cache_stats()
    assert stats["total_items"] == 2
    assert stats["active_items"] == 1
    assert stats["expired_items"] == 1
