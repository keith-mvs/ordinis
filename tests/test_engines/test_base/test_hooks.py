"""Tests for governance hooks.

This module tests the governance hook protocol, base implementations,
and composite hook functionality.
"""

from datetime import datetime

import pytest

from ordinis.engines.base import (
    BaseGovernanceHook,
    CompositeGovernanceHook,
    Decision,
    EngineError,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.base.models import AuditRecord
from tests.test_engines.test_base.conftest import MockGovernanceHook


class TestDecisionEnum:
    """Test Decision enum values."""

    def test_decision_values(self) -> None:
        """Test all decision enum values exist."""
        assert Decision.ALLOW.value == "allow"
        assert Decision.DENY.value == "deny"
        assert Decision.WARN.value == "warn"
        assert Decision.DEFER.value == "defer"

    def test_decision_comparison(self) -> None:
        """Test decision enum comparisons."""
        assert Decision.ALLOW == Decision.ALLOW
        assert Decision.DENY != Decision.ALLOW
        assert Decision.WARN != Decision.DENY


class TestPreflightContext:
    """Test PreflightContext dataclass."""

    def test_minimal_context(self) -> None:
        """Test creating context with minimal fields."""
        context = PreflightContext(
            engine="TestEngine",
            action="test_action",
        )

        assert context.engine == "TestEngine"
        assert context.action == "test_action"
        assert context.inputs == {}
        assert context.trace_id is None
        assert context.user_id is None
        assert context.metadata == {}

    def test_full_context(self) -> None:
        """Test creating context with all fields."""
        context = PreflightContext(
            engine="TestEngine",
            action="test_action",
            inputs={"key": "value"},
            trace_id="trace-123",
            user_id="user-456",
            metadata={"source": "test"},
        )

        assert context.engine == "TestEngine"
        assert context.action == "test_action"
        assert context.inputs == {"key": "value"}
        assert context.trace_id == "trace-123"
        assert context.user_id == "user-456"
        assert context.metadata == {"source": "test"}

    def test_context_default_factories(self) -> None:
        """Test default factories create independent instances."""
        context1 = PreflightContext(engine="E1", action="A1")
        context2 = PreflightContext(engine="E2", action="A2")

        context1.inputs["key"] = "value1"
        context2.inputs["key"] = "value2"

        assert context1.inputs["key"] == "value1"
        assert context2.inputs["key"] == "value2"


class TestPreflightResult:
    """Test PreflightResult dataclass."""

    def test_minimal_result(self) -> None:
        """Test creating result with minimal fields."""
        result = PreflightResult(decision=Decision.ALLOW)

        assert result.decision == Decision.ALLOW
        assert result.reason == ""
        assert result.policy_id is None
        assert result.policy_version is None
        assert result.adjustments == {}
        assert result.warnings == []
        assert result.expires_at is None

    def test_full_result(self) -> None:
        """Test creating result with all fields."""
        expires = datetime.utcnow()
        result = PreflightResult(
            decision=Decision.DENY,
            reason="Policy violation",
            policy_id="POL-001",
            policy_version="1.0.0",
            adjustments={"key": "value"},
            warnings=["Warning 1", "Warning 2"],
            expires_at=expires,
        )

        assert result.decision == Decision.DENY
        assert result.reason == "Policy violation"
        assert result.policy_id == "POL-001"
        assert result.policy_version == "1.0.0"
        assert result.adjustments == {"key": "value"}
        assert result.warnings == ["Warning 1", "Warning 2"]
        assert result.expires_at == expires

    def test_allowed_property(self) -> None:
        """Test allowed property for different decisions."""
        allow_result = PreflightResult(decision=Decision.ALLOW)
        warn_result = PreflightResult(decision=Decision.WARN)
        deny_result = PreflightResult(decision=Decision.DENY)
        defer_result = PreflightResult(decision=Decision.DEFER)

        assert allow_result.allowed is True
        assert warn_result.allowed is True
        assert deny_result.allowed is False
        assert defer_result.allowed is False

    def test_blocked_property(self) -> None:
        """Test blocked property for different decisions."""
        allow_result = PreflightResult(decision=Decision.ALLOW)
        warn_result = PreflightResult(decision=Decision.WARN)
        deny_result = PreflightResult(decision=Decision.DENY)
        defer_result = PreflightResult(decision=Decision.DEFER)

        assert allow_result.blocked is False
        assert warn_result.blocked is False
        assert deny_result.blocked is True
        assert defer_result.blocked is False


class TestBaseGovernanceHook:
    """Test BaseGovernanceHook implementation."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test hook initialization."""
        hook = BaseGovernanceHook("TestEngine")

        assert hook.engine_name == "TestEngine"
        assert hook.policy_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_default_preflight_allows(self) -> None:
        """Test default preflight implementation allows all operations."""
        hook = BaseGovernanceHook("TestEngine")
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await hook.preflight(context)

        assert result.decision == Decision.ALLOW
        assert "allow all" in result.reason.lower()
        assert result.policy_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_default_audit_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test default audit implementation logs to standard logger."""
        hook = BaseGovernanceHook("TestEngine")
        record = AuditRecord(
            engine="TestEngine",
            action="test_action",
            trace_id="trace-123",
            decision="allow",
        )

        with caplog.at_level("INFO"):
            await hook.audit(record)

        assert "AUDIT" in caplog.text
        assert "TestEngine" in caplog.text
        assert "test_action" in caplog.text

    @pytest.mark.asyncio
    async def test_default_error_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test default error handler logs errors."""
        hook = BaseGovernanceHook("TestEngine")
        error = EngineError(
            code="TEST_ERROR",
            message="Test error message",
            engine="TestEngine",
            recoverable=False,
        )

        with caplog.at_level("ERROR"):
            await hook.on_error(error)

        assert "ENGINE_ERROR" in caplog.text
        assert "TEST_ERROR" in caplog.text
        assert "Test error message" in caplog.text

    @pytest.mark.asyncio
    async def test_error_recoverable_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test recoverable errors log as warnings."""
        hook = BaseGovernanceHook("TestEngine")
        error = EngineError(
            code="TEST_ERROR",
            message="Recoverable error",
            engine="TestEngine",
            recoverable=True,
        )

        with caplog.at_level("WARNING"):
            await hook.on_error(error)

        assert "ENGINE_ERROR" in caplog.text

    @pytest.mark.asyncio
    async def test_policy_version_property(self) -> None:
        """Test policy_version property accessor."""
        hook = BaseGovernanceHook("TestEngine")

        assert hook.policy_version == "1.0.0"


class TestGovernanceHookProtocol:
    """Test GovernanceHook protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_hook_implements_protocol(self) -> None:
        """Test MockGovernanceHook implements GovernanceHook protocol."""
        hook = MockGovernanceHook("TestEngine")

        assert hasattr(hook, "preflight")
        assert hasattr(hook, "audit")
        assert hasattr(hook, "on_error")

    @pytest.mark.asyncio
    async def test_base_hook_implements_protocol(self) -> None:
        """Test BaseGovernanceHook implements GovernanceHook protocol."""
        hook = BaseGovernanceHook("TestEngine")

        assert hasattr(hook, "preflight")
        assert hasattr(hook, "audit")
        assert hasattr(hook, "on_error")


class TestMockGovernanceHook:
    """Test MockGovernanceHook test fixture."""

    @pytest.mark.asyncio
    async def test_tracks_preflight_calls(self) -> None:
        """Test mock hook tracks preflight calls."""
        hook = MockGovernanceHook("TestEngine")
        context = PreflightContext(engine="TestEngine", action="test_action")

        await hook.preflight(context)
        await hook.preflight(context)

        assert len(hook.preflight_calls) == 2
        assert hook.preflight_calls[0].action == "test_action"

    @pytest.mark.asyncio
    async def test_tracks_audit_calls(self) -> None:
        """Test mock hook tracks audit calls."""
        hook = MockGovernanceHook("TestEngine")
        record = AuditRecord(engine="TestEngine", action="test_action")

        await hook.audit(record)
        await hook.audit(record)

        assert len(hook.audit_calls) == 2

    @pytest.mark.asyncio
    async def test_tracks_error_calls(self) -> None:
        """Test mock hook tracks error calls."""
        hook = MockGovernanceHook("TestEngine")
        error = EngineError(code="TEST", message="Test", engine="TestEngine")

        await hook.on_error(error)
        await hook.on_error(error)

        assert len(hook.error_calls) == 2

    @pytest.mark.asyncio
    async def test_custom_preflight_decision(self) -> None:
        """Test configuring custom preflight decision."""
        hook = MockGovernanceHook(
            "TestEngine",
            preflight_decision=Decision.DENY,
            preflight_reason="Custom denial",
        )
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await hook.preflight(context)

        assert result.decision == Decision.DENY
        assert result.reason == "Custom denial"
        assert result.policy_id == "MOCK-001"


class TestCompositeGovernanceHook:
    """Test CompositeGovernanceHook combining multiple hooks."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test composite hook initialization."""
        hook1 = BaseGovernanceHook("TestEngine")
        hook2 = BaseGovernanceHook("TestEngine")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])

        assert composite.engine_name == "TestEngine"
        assert len(composite._hooks) == 2

    @pytest.mark.asyncio
    async def test_preflight_all_allow(self) -> None:
        """Test preflight with all hooks allowing."""
        hook1 = MockGovernanceHook("TestEngine", Decision.ALLOW)
        hook2 = MockGovernanceHook("TestEngine", Decision.ALLOW)
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await composite.preflight(context)

        assert result.decision == Decision.ALLOW
        assert len(hook1.preflight_calls) == 1
        assert len(hook2.preflight_calls) == 1

    @pytest.mark.asyncio
    async def test_preflight_one_deny(self) -> None:
        """Test preflight with one hook denying."""
        hook1 = MockGovernanceHook("TestEngine", Decision.ALLOW)
        hook2 = MockGovernanceHook("TestEngine", Decision.DENY, "Denied by hook2")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await composite.preflight(context)

        assert result.decision == Decision.DENY
        assert result.reason == "Denied by hook2"

    @pytest.mark.asyncio
    async def test_preflight_short_circuit_on_deny(self) -> None:
        """Test preflight short-circuits on first DENY."""
        hook1 = MockGovernanceHook("TestEngine", Decision.DENY, "First denial")
        hook2 = MockGovernanceHook("TestEngine", Decision.ALLOW)
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await composite.preflight(context)

        assert result.decision == Decision.DENY
        assert len(hook1.preflight_calls) == 1
        assert len(hook2.preflight_calls) == 0

    @pytest.mark.asyncio
    async def test_preflight_priority_defer_over_warn(self) -> None:
        """Test DEFER takes priority over WARN."""
        hook1 = MockGovernanceHook("TestEngine", Decision.WARN)
        hook2 = MockGovernanceHook("TestEngine", Decision.DEFER)
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await composite.preflight(context)

        assert result.decision == Decision.DEFER

    @pytest.mark.asyncio
    async def test_preflight_priority_warn_over_allow(self) -> None:
        """Test WARN takes priority over ALLOW."""
        hook1 = MockGovernanceHook("TestEngine", Decision.ALLOW)
        hook2 = MockGovernanceHook("TestEngine", Decision.WARN)
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        context = PreflightContext(engine="TestEngine", action="test_action")

        result = await composite.preflight(context)

        assert result.decision == Decision.WARN

    @pytest.mark.asyncio
    async def test_audit_calls_all_hooks(self) -> None:
        """Test audit calls all hooks."""
        hook1 = MockGovernanceHook("TestEngine")
        hook2 = MockGovernanceHook("TestEngine")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        record = AuditRecord(engine="TestEngine", action="test_action")

        await composite.audit(record)

        assert len(hook1.audit_calls) == 1
        assert len(hook2.audit_calls) == 1

    @pytest.mark.asyncio
    async def test_audit_continues_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test audit continues even if a hook fails."""

        class FailingHook(BaseGovernanceHook):
            async def audit(self, record: AuditRecord) -> None:
                raise RuntimeError("Audit failed")

        hook1 = FailingHook("TestEngine")
        hook2 = MockGovernanceHook("TestEngine")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        record = AuditRecord(engine="TestEngine", action="test_action")

        with caplog.at_level("DEBUG"):
            await composite.audit(record)

        assert len(hook2.audit_calls) == 1

    @pytest.mark.asyncio
    async def test_error_calls_all_hooks(self) -> None:
        """Test on_error calls all hooks."""
        hook1 = MockGovernanceHook("TestEngine")
        hook2 = MockGovernanceHook("TestEngine")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        error = EngineError(code="TEST", message="Test", engine="TestEngine")

        await composite.on_error(error)

        assert len(hook1.error_calls) == 1
        assert len(hook2.error_calls) == 1

    @pytest.mark.asyncio
    async def test_error_continues_on_hook_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test on_error continues even if a hook fails."""

        class FailingHook(BaseGovernanceHook):
            async def on_error(self, error: EngineError) -> None:
                raise RuntimeError("Error handler failed")

        hook1 = FailingHook("TestEngine")
        hook2 = MockGovernanceHook("TestEngine")
        composite = CompositeGovernanceHook("TestEngine", [hook1, hook2])
        error = EngineError(code="TEST", message="Test", engine="TestEngine")

        with caplog.at_level("DEBUG"):
            await composite.on_error(error)

        assert len(hook2.error_calls) == 1

    @pytest.mark.asyncio
    async def test_empty_hooks_list(self) -> None:
        """Test composite hook with empty hooks list.

        Note: This currently raises ValueError because min() is called on
        an empty results list. This is a known limitation - composite hooks
        should have at least one hook.
        """
        composite = CompositeGovernanceHook("TestEngine", [])
        context = PreflightContext(engine="TestEngine", action="test_action")

        with pytest.raises(ValueError, match="empty sequence"):
            await composite.preflight(context)
