"""Comprehensive tests for BaseEngine."""

import pytest

from ordinis.engines.base import BaseEngine, BaseEngineConfig, EngineState


class MockEngineConfig(BaseEngineConfig):
    """Mock configuration for testing."""

    test_setting: str = "default"


class MockEngine(BaseEngine):
    """Concrete implementation of BaseEngine for testing."""

    def __init__(self, config: MockEngineConfig = None):
        if config is None:
            config = MockEngineConfig()
        super().__init__(config=config)
        self._initialized = False

    def initialize(self):
        """Initialize the engine."""
        if self._state == EngineState.READY:
            return

        self._state = EngineState.INITIALIZING
        self._initialized = True
        self._state = EngineState.READY

    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "initialized": self._initialized,
            "state": self._state.value,
            "type": str(self.config.engine_type.value),
        }


@pytest.mark.unit
class TestBaseEngineStates:
    """Tests for BaseEngine state management."""

    def test_initial_state(self):
        """Test engine starts in NOT_READY state."""
        engine = MockEngine()
        assert engine._state == EngineState.NOT_READY
        assert not engine._initialized

    def test_initialize_state_transition(self):
        """Test state transitions during initialization."""
        engine = MockEngine()

        engine.initialize()

        assert engine._state == EngineState.READY
        assert engine._initialized

    def test_get_stats_before_init(self):
        """Test get_stats before initialization."""
        engine = MockEngine()

        stats = engine.get_stats()

        assert stats["initialized"] is False
        assert stats["state"] == "not_ready"

    def test_get_stats_after_init(self):
        """Test get_stats after initialization."""
        engine = MockEngine()
        engine.initialize()

        stats = engine.get_stats()

        assert stats["initialized"] is True
        assert stats["state"] == "ready"


@pytest.mark.unit
class TestBaseEngineConfig:
    """Tests for BaseEngineConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        config = MockEngineConfig()

        assert config.engine_type == EngineType.CUSTOM
        assert config.test_setting == "default"

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = MockEngineConfig(test_setting="custom")

        assert config.test_setting == "custom"

    def test_config_inheritance(self):
        """Test that MockEngineConfig inherits from BaseEngineConfig."""
        config = MockEngineConfig()

        assert isinstance(config, BaseEngineConfig)
        assert hasattr(config, "engine_type")


@pytest.mark.unit
class TestEngineType:
    """Tests for EngineType enum."""

    def test_engine_types_exist(self):
        """Test that standard engine types exist."""
        assert hasattr(EngineType, "SIGNALCORE")
        assert hasattr(EngineType, "RISKGUARD")
        assert hasattr(EngineType, "HELIX")
        assert hasattr(EngineType, "CODEGEN")
        assert hasattr(EngineType, "SYNAPSE")
        assert hasattr(EngineType, "CUSTOM")

    def test_engine_type_values(self):
        """Test engine type string values."""
        assert EngineType.SIGNALCORE.value == "signalcore"
        assert EngineType.RISKGUARD.value == "riskguard"
        assert EngineType.HELIX.value == "helix"
        assert EngineType.CODEGEN.value == "codegen"
        assert EngineType.SYNAPSE.value == "synapse"
        assert EngineType.CUSTOM.value == "custom"


@pytest.mark.unit
class TestEngineState:
    """Tests for EngineState enum."""

    def test_engine_states_exist(self):
        """Test that all engine states exist."""
        assert hasattr(EngineState, "NOT_READY")
        assert hasattr(EngineState, "INITIALIZING")
        assert hasattr(EngineState, "READY")
        assert hasattr(EngineState, "ERROR")

    def test_engine_state_values(self):
        """Test engine state string values."""
        assert EngineState.NOT_READY.value == "not_ready"
        assert EngineState.INITIALIZING.value == "initializing"
        assert EngineState.READY.value == "ready"
        assert EngineState.ERROR.value == "error"

    def test_state_transitions(self):
        """Test valid state transitions."""
        engine = MockEngine()

        # NOT_READY → INITIALIZING
        engine._state = EngineState.INITIALIZING
        assert engine._state == EngineState.INITIALIZING

        # INITIALIZING → READY
        engine._state = EngineState.READY
        assert engine._state == EngineState.READY

        # Any → ERROR
        engine._state = EngineState.ERROR
        assert engine._state == EngineState.ERROR


@pytest.mark.unit
class TestBaseEngineIntegration:
    """Integration tests for BaseEngine."""

    def test_full_lifecycle(self):
        """Test complete engine lifecycle."""
        config = MockEngineConfig(test_setting="lifecycle_test")
        engine = MockEngine(config=config)

        # Check initial state
        assert engine._state == EngineState.NOT_READY
        stats = engine.get_stats()
        assert stats["initialized"] is False

        # Initialize
        engine.initialize()
        assert engine._state == EngineState.READY
        stats = engine.get_stats()
        assert stats["initialized"] is True

        # Re-initialize should be idempotent
        engine.initialize()
        assert engine._state == EngineState.READY

    def test_config_passed_to_engine(self):
        """Test that configuration is properly stored."""
        config = MockEngineConfig(test_setting="test_value")
        engine = MockEngine(config=config)

        assert engine.config.test_setting == "test_value"
        assert engine.config.engine_type == EngineType.CUSTOM
