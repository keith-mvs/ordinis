"""Comprehensive tests for BaseEngine."""

from dataclasses import dataclass

import pytest

from ordinis.engines.base import (
    BaseEngine,
    BaseEngineConfig,
    EngineState,
    HealthLevel,
    HealthStatus,
)


@dataclass
class MockEngineConfig(BaseEngineConfig):
    """Mock configuration for testing."""

    test_setting: str = "default"


class MockEngine(BaseEngine[MockEngineConfig]):
    """Concrete implementation of BaseEngine for testing."""

    def __init__(self, config: MockEngineConfig | None = None):
        if config is None:
            config = MockEngineConfig()
        super().__init__(config=config)
        self._initialized = False

    async def _do_initialize(self) -> None:
        """Initialize the engine."""
        self._initialized = True

    async def _do_shutdown(self) -> None:
        """Shutdown the engine."""
        self._initialized = False

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        return HealthStatus(
            level=HealthLevel.HEALTHY,
            message="Mock engine is healthy",
            details={"initialized": self._initialized},
        )

    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "initialized": self._initialized,
            "state": self._state.value,
        }


@pytest.mark.asyncio
class TestBaseEngineStates:
    """Tests for BaseEngine state management."""

    async def test_initial_state(self):
        """Test engine starts in UNINITIALIZED state."""
        engine = MockEngine()
        assert engine.state == EngineState.UNINITIALIZED
        assert not engine._initialized

    async def test_initialize_state_transition(self):
        """Test state transitions during initialization."""
        engine = MockEngine()

        await engine.initialize()

        assert engine.state == EngineState.READY
        assert engine._initialized

    async def test_get_stats_before_init(self):
        """Test get_stats before initialization."""
        engine = MockEngine()

        stats = engine.get_stats()

        assert stats["initialized"] is False
        assert stats["state"] == "uninitialized"

    async def test_get_stats_after_init(self):
        """Test get_stats after initialization."""
        engine = MockEngine()
        await engine.initialize()

        stats = engine.get_stats()

        assert stats["initialized"] is True
        assert stats["state"] == "ready"


class TestBaseEngineConfig:
    """Tests for BaseEngineConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        config = MockEngineConfig()

        assert config.test_setting == "default"
        assert config.enabled is True

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = MockEngineConfig(test_setting="custom")

        assert config.test_setting == "custom"

    def test_config_inheritance(self):
        """Test that MockEngineConfig inherits from BaseEngineConfig."""
        config = MockEngineConfig()
        assert isinstance(config, BaseEngineConfig)


@pytest.mark.asyncio
class TestBaseEngineIntegration:
    """Integration tests for BaseEngine."""

    async def test_full_lifecycle(self):
        """Test full engine lifecycle."""
        engine = MockEngine()

        # Start
        await engine.initialize()
        assert engine.state == EngineState.READY

        # Health check
        health = await engine.health_check()
        assert health.level == HealthLevel.HEALTHY

        # Shutdown
        await engine.shutdown()
        assert engine.state == EngineState.STOPPING or engine.state == EngineState.STOPPED
