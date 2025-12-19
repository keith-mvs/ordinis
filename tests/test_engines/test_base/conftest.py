"""Shared fixtures for base engine tests.

This module provides mock implementations and fixtures for testing
the base engine framework components.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from ordinis.engines.base import (
    BaseEngine,
    BaseEngineConfig,
    BaseGovernanceHook,
    Decision,
    EngineError,
    GovernanceHook,
    HealthLevel,
    HealthStatus,
    PreflightContext,
    PreflightResult,
)


@dataclass
class MockConfig(BaseEngineConfig):
    """Mock configuration for testing.

    Extends BaseEngineConfig with test-specific settings.
    """

    test_setting: str = "default"
    fail_initialize: bool = False
    fail_shutdown: bool = False
    fail_health_check: bool = False
    health_level: HealthLevel = HealthLevel.HEALTHY


class MockEngine(BaseEngine[MockConfig]):
    """Mock engine implementation for testing.

    Provides a concrete implementation of BaseEngine that can be
    configured to simulate various behaviors and failure modes.
    """

    def __init__(
        self,
        config: MockConfig,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the mock engine."""
        super().__init__(config, governance_hook)
        self.initialized = False
        self.shutdown_called = False
        self.health_check_count = 0

    async def _do_initialize(self) -> None:
        """Mock initialization logic."""
        if self.config.fail_initialize:
            raise RuntimeError("Simulated initialization failure")
        self.initialized = True

    async def _do_shutdown(self) -> None:
        """Mock shutdown logic."""
        if self.config.fail_shutdown:
            raise RuntimeError("Simulated shutdown failure")
        self.shutdown_called = True
        self.initialized = False

    async def _do_health_check(self) -> HealthStatus:
        """Mock health check logic."""
        self.health_check_count += 1

        if self.config.fail_health_check:
            raise RuntimeError("Simulated health check failure")

        return HealthStatus(
            level=self.config.health_level,
            message=f"Mock engine health check #{self.health_check_count}",
            details={"initialized": self.initialized},
        )


class MockGovernanceHook(BaseGovernanceHook):
    """Mock governance hook for testing.

    Tracks all calls and can be configured to return specific decisions.
    """

    def __init__(
        self,
        engine_name: str,
        preflight_decision: Decision = Decision.ALLOW,
        preflight_reason: str = "Test policy",
    ) -> None:
        """Initialize the mock governance hook.

        Args:
            engine_name: Name of the engine.
            preflight_decision: Decision to return from preflight.
            preflight_reason: Reason for the decision.
        """
        super().__init__(engine_name)
        self.preflight_decision = preflight_decision
        self.preflight_reason = preflight_reason

        # Call tracking
        self.preflight_calls: list[PreflightContext] = []
        self.audit_calls: list[Any] = []
        self.error_calls: list[EngineError] = []

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Mock preflight that tracks calls."""
        self.preflight_calls.append(context)
        return PreflightResult(
            decision=self.preflight_decision,
            reason=self.preflight_reason,
            policy_id="MOCK-001",
            policy_version=self.policy_version,
        )

    async def audit(self, record: Any) -> None:
        """Mock audit that tracks calls."""
        self.audit_calls.append(record)

    async def on_error(self, error: EngineError) -> None:
        """Mock error handler that tracks calls."""
        self.error_calls.append(error)


@pytest.fixture
def mock_config() -> MockConfig:
    """Provide a default mock configuration.

    Returns:
        MockConfig instance with default settings.
    """
    return MockConfig(name="TestEngine")


@pytest.fixture
def mock_config_with_governance_disabled() -> MockConfig:
    """Provide a mock configuration with governance disabled.

    Returns:
        MockConfig instance with governance disabled.
    """
    return MockConfig(
        name="TestEngine",
        governance_enabled=False,
        audit_enabled=False,
    )


@pytest.fixture
def mock_governance_hook() -> MockGovernanceHook:
    """Provide a mock governance hook that allows all operations.

    Returns:
        MockGovernanceHook instance.
    """
    return MockGovernanceHook("TestEngine")


@pytest.fixture
def mock_governance_hook_deny() -> MockGovernanceHook:
    """Provide a mock governance hook that denies operations.

    Returns:
        MockGovernanceHook instance configured to deny.
    """
    return MockGovernanceHook(
        "TestEngine",
        preflight_decision=Decision.DENY,
        preflight_reason="Test denial",
    )


@pytest.fixture
def mock_engine(mock_config: MockConfig) -> MockEngine:
    """Provide a mock engine with default configuration.

    Args:
        mock_config: Mock configuration fixture.

    Returns:
        MockEngine instance.
    """
    return MockEngine(mock_config)


@pytest.fixture
def mock_engine_with_hook(
    mock_config: MockConfig,
    mock_governance_hook: MockGovernanceHook,
) -> MockEngine:
    """Provide a mock engine with custom governance hook.

    Args:
        mock_config: Mock configuration fixture.
        mock_governance_hook: Mock governance hook fixture.

    Returns:
        MockEngine instance with custom hook.
    """
    return MockEngine(mock_config, mock_governance_hook)


@pytest.fixture
async def initialized_engine(mock_engine: MockEngine) -> MockEngine:
    """Provide an initialized mock engine.

    Args:
        mock_engine: Mock engine fixture.

    Returns:
        Initialized MockEngine instance.
    """
    await mock_engine.initialize()
    yield mock_engine
    # Cleanup
    if mock_engine.is_running:
        await mock_engine.shutdown()


@pytest.fixture
def sample_preflight_context() -> PreflightContext:
    """Provide a sample preflight context for testing.

    Returns:
        PreflightContext instance.
    """
    return PreflightContext(
        engine="TestEngine",
        action="test_action",
        inputs={"symbol": "AAPL", "quantity": 100},
        trace_id="test-trace-123",
        user_id="test-user",
        metadata={"source": "test"},
    )
