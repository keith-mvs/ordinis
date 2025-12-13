"""
Tests for kill switch module.

Tests cover:
- KillSwitchReason enum
- KillSwitchState dataclass
- KillSwitch initialization
- State properties
- Trigger conditions
- Callbacks
"""

import asyncio
from datetime import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest

from ordinis.safety.kill_switch import (
    KillSwitch,
    KillSwitchReason,
    KillSwitchState,
)


class TestKillSwitchReason:
    """Test KillSwitchReason enum."""

    @pytest.mark.unit
    def test_kill_switch_reason_values(self):
        """Test KillSwitchReason enum values."""
        assert KillSwitchReason.MANUAL.value == "manual"
        assert KillSwitchReason.DAILY_LOSS_LIMIT.value == "daily_loss_limit"
        assert KillSwitchReason.MAX_DRAWDOWN.value == "max_drawdown"
        assert KillSwitchReason.CONSECUTIVE_LOSSES.value == "consecutive_losses"
        assert KillSwitchReason.API_CONNECTIVITY.value == "api_connectivity"
        assert KillSwitchReason.SYSTEM_HEALTH.value == "system_health"
        assert KillSwitchReason.FILE_TRIGGER.value == "file_trigger"
        assert KillSwitchReason.POSITION_RECONCILIATION.value == "position_reconciliation"
        assert KillSwitchReason.BROKER_ERROR.value == "broker_error"
        assert KillSwitchReason.UNKNOWN.value == "unknown"


class TestKillSwitchState:
    """Test KillSwitchState dataclass."""

    @pytest.mark.unit
    def test_kill_switch_state_defaults(self):
        """Test KillSwitchState default values."""
        state = KillSwitchState()
        assert state.active is False
        assert state.reason == KillSwitchReason.UNKNOWN
        assert state.message == ""
        assert state.timestamp is None
        assert state.triggered_by == ""
        assert state.metadata == {}

    @pytest.mark.unit
    def test_kill_switch_state_custom_values(self):
        """Test KillSwitchState with custom values."""
        now = datetime.utcnow()
        state = KillSwitchState(
            active=True,
            reason=KillSwitchReason.MANUAL,
            message="Test trigger",
            timestamp=now,
            triggered_by="test_user",
            metadata={"key": "value"},
        )
        assert state.active is True
        assert state.reason == KillSwitchReason.MANUAL
        assert state.message == "Test trigger"
        assert state.timestamp == now
        assert state.triggered_by == "test_user"
        assert state.metadata == {"key": "value"}


class TestKillSwitch:
    """Test KillSwitch class."""

    @pytest.mark.unit
    def test_kill_switch_initialization_defaults(self):
        """Test KillSwitch initialization with defaults."""
        ks = KillSwitch()
        assert ks.is_active is False
        assert ks.state.active is False
        assert ks._daily_loss_limit == 1000.0
        assert ks._max_drawdown_pct == 5.0
        assert ks._consecutive_loss_limit == 5

    @pytest.mark.unit
    def test_kill_switch_initialization_custom(self):
        """Test KillSwitch initialization with custom values."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(
                kill_file=kill_file,
                daily_loss_limit=500.0,
                max_drawdown_pct=3.0,
                consecutive_loss_limit=3,
                check_interval_seconds=2.0,
            )
            assert ks._kill_file == kill_file
            assert ks._daily_loss_limit == 500.0
            assert ks._max_drawdown_pct == 3.0
            assert ks._consecutive_loss_limit == 3
            assert ks._check_interval == 2.0

    @pytest.mark.unit
    def test_is_active_property(self):
        """Test is_active property."""
        ks = KillSwitch()
        assert ks.is_active is False
        ks._state.active = True
        assert ks.is_active is True

    @pytest.mark.unit
    def test_state_property(self):
        """Test state property returns KillSwitchState."""
        ks = KillSwitch()
        state = ks.state
        assert isinstance(state, KillSwitchState)


class TestKillSwitchTrigger:
    """Test KillSwitch trigger functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_activates_switch(self):
        """Test triggering activates the kill switch."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(
                reason=KillSwitchReason.MANUAL,
                message="Test trigger",
                triggered_by="test",
            )

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.MANUAL
            assert ks.state.message == "Test trigger"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_creates_kill_file(self):
        """Test triggering creates kill file."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"  # No nested dir
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(
                reason=KillSwitchReason.MANUAL,
                message="Test",
                triggered_by="test",
            )

            assert kill_file.exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_deactivates_switch(self):
        """Test reset deactivates kill switch."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            # First activate
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
            assert ks.is_active is True

            # Then reset
            await ks.reset(force=True)
            assert ks.is_active is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_removes_kill_file(self):
        """Test reset removes kill file."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
            assert kill_file.exists()

            await ks.reset(force=True)
            assert not kill_file.exists()


class TestKillSwitchCallbacks:
    """Test KillSwitch callback functionality."""

    @pytest.mark.unit
    def test_register_callback(self):
        """Test registering a callback."""
        ks = KillSwitch()
        callback_called = []

        def my_callback(state: KillSwitchState):
            callback_called.append(state)

        ks.register_callback(my_callback)
        assert len(ks._callbacks) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_callback_invoked_on_trigger(self):
        """Test callback is invoked when triggered."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)
            callback_states = []

            def my_callback(state: KillSwitchState):
                callback_states.append(state)

            ks.register_callback(my_callback)
            await ks.trigger(KillSwitchReason.MANUAL, "Test")

            assert len(callback_states) == 1
            assert callback_states[0].active is True

    @pytest.mark.unit
    def test_register_async_callback(self):
        """Test registering an async callback."""
        ks = KillSwitch()

        async def async_callback(state: KillSwitchState):
            pass

        ks.register_async_callback(async_callback)
        assert len(ks._async_callbacks) == 1


class TestKillSwitchRiskChecks:
    """Test KillSwitch risk monitoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_daily_loss_triggers(self):
        """Test daily loss limit triggers kill switch."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(
                kill_file=kill_file,
                daily_loss_limit=100.0,
            )

            # Check daily loss exceeding limit
            await ks.check_daily_loss(-150.0)

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.DAILY_LOSS_LIMIT

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_consecutive_losses_triggers(self):
        """Test consecutive losses triggers kill switch."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(
                kill_file=kill_file,
                consecutive_loss_limit=3,
            )

            # Record consecutive losses (negative pnl)
            await ks.record_trade_result(-10.0)
            await ks.record_trade_result(-20.0)
            await ks.record_trade_result(-30.0)

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.CONSECUTIVE_LOSSES

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_winning_trade_resets_consecutive_losses(self):
        """Test winning trade resets consecutive loss counter."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(
                kill_file=kill_file,
                consecutive_loss_limit=5,
            )

            await ks.record_trade_result(-10.0)  # Loss
            await ks.record_trade_result(-20.0)  # Loss
            assert ks._consecutive_losses == 2

            await ks.record_trade_result(50.0)  # Win (positive pnl)
            assert ks._consecutive_losses == 0


class TestKillSwitchStatus:
    """Test KillSwitch status reporting."""

    @pytest.mark.unit
    def test_get_status(self):
        """Test get_status returns correct information."""
        ks = KillSwitch(
            daily_loss_limit=500.0,
            max_drawdown_pct=3.0,
        )
        status = ks.get_status()

        assert status["active"] is False
        assert status["limits"]["daily_loss"] == 500.0
        assert status["limits"]["max_drawdown_pct"] == 3.0
        assert "consecutive_losses" in status
        assert "daily_pnl" in status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_status_after_trigger(self):
        """Test status after kill switch is triggered."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
            status = ks.get_status()

            assert status["active"] is True
            assert status["reason"] == "manual"


class TestKillSwitchInitialization:
    """Test KillSwitch initialization and startup scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_creates_kill_file_directory(self):
        """Test initialize creates kill file parent directory."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "nested" / "dir" / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.initialize()
            assert kill_file.parent.exists()
            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_detects_existing_kill_file(self):
        """Test initialize triggers if kill file exists."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            kill_file.write_text("Emergency stop")

            ks = KillSwitch(kill_file=kill_file)
            result = await ks.initialize()

            assert result is True
            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.FILE_TRIGGER
            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_starts_file_monitoring(self):
        """Test initialize starts file check task."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.initialize()
            assert ks._running is True
            assert ks._file_check_task is not None
            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_stops_monitoring(self):
        """Test shutdown stops file check task."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.initialize()
            assert ks._running is True

            await ks.shutdown()
            assert ks._running is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_cancels_file_check_task(self):
        """Test shutdown properly cancels file check task."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, check_interval_seconds=0.1)

            await ks.initialize()
            task = ks._file_check_task
            assert task is not None
            assert not task.done()

            await ks.shutdown()
            assert task.cancelled() or task.done()


class TestKillSwitchFileMonitoring:
    """Test file-based kill switch monitoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_file_check_loop_detects_file(self):
        """Test file monitoring loop detects kill file creation."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, check_interval_seconds=0.05)

            await ks.initialize()

            # Create kill file after initialization
            await asyncio.sleep(0.02)
            kill_file.write_text("Emergency")
            await asyncio.sleep(0.15)

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.FILE_TRIGGER
            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_file_check_loop_ignores_when_active(self):
        """Test file loop doesn't re-trigger when already active."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, check_interval_seconds=0.05)

            # Activate first
            await ks.trigger(KillSwitchReason.MANUAL, "First trigger", "test")
            initial_timestamp = ks.state.timestamp

            await ks.initialize()
            await asyncio.sleep(0.15)

            # Timestamp shouldn't change
            assert ks.state.timestamp == initial_timestamp
            assert ks.state.reason == KillSwitchReason.MANUAL
            await ks.shutdown()


class TestKillSwitchTriggerEdgeCases:
    """Test edge cases in trigger functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_when_already_active_returns_true(self):
        """Test triggering when already active returns True without changing state."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            # First trigger
            await ks.trigger(KillSwitchReason.MANUAL, "First", "test1")
            first_timestamp = ks.state.timestamp

            # Second trigger
            result = await ks.trigger(KillSwitchReason.DAILY_LOSS_LIMIT, "Second", "test2")

            assert result is True
            assert ks.state.reason == KillSwitchReason.MANUAL
            assert ks.state.timestamp == first_timestamp

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_with_metadata(self):
        """Test triggering with metadata."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            metadata = {"loss": 500.0, "limit": 1000.0}
            await ks.trigger(
                KillSwitchReason.DAILY_LOSS_LIMIT,
                "Loss exceeded",
                "test",
                metadata=metadata,
            )

            assert ks.state.metadata == metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_creates_kill_file_with_details(self):
        """Test trigger creates kill file with full details."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(
                KillSwitchReason.MAX_DRAWDOWN,
                "Drawdown exceeded",
                "risk_monitor",
            )

            assert kill_file.exists()
            content = kill_file.read_text()
            assert "Drawdown exceeded" in content
            assert "max_drawdown" in content
            assert "risk_monitor" in content


class TestKillSwitchReset:
    """Test reset functionality edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_when_inactive_returns_true(self):
        """Test reset when already inactive returns True."""
        ks = KillSwitch()
        result = await ks.reset(force=True)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_without_force_blocks_on_loss_limit(self):
        """Test reset without force blocks when daily loss still exceeded."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, daily_loss_limit=100.0)

            # Trigger and track loss
            await ks.check_daily_loss(-150.0)
            assert ks.is_active is True

            # Try to reset without force
            result = await ks.reset(force=False)
            assert result is False
            assert ks.is_active is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_with_force_succeeds_despite_loss(self):
        """Test reset with force succeeds even with loss exceeded."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, daily_loss_limit=100.0)

            await ks.check_daily_loss(-150.0)
            assert ks.is_active is True

            result = await ks.reset(force=True)
            assert result is True
            assert ks.is_active is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_without_force_succeeds_when_cleared(self):
        """Test reset without force succeeds when conditions cleared."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, daily_loss_limit=100.0)

            # Trigger
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            # Update daily PnL to acceptable level
            ks._daily_pnl = -50.0

            # Reset should succeed
            result = await ks.reset(force=False)
            assert result is True
            assert ks.is_active is False


class TestKillSwitchCallbackEdgeCases:
    """Test callback error handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_prevent_trigger(self):
        """Test exception in callback doesn't prevent kill switch activation."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            def failing_callback(state):
                raise ValueError("Callback error")

            ks.register_callback(failing_callback)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            # Should still be active despite callback error
            assert ks.is_active is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_callback_exception_doesnt_prevent_trigger(self):
        """Test exception in async callback doesn't prevent activation."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            async def failing_async_callback(state):
                raise ValueError("Async callback error")

            ks.register_async_callback(failing_async_callback)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            assert ks.is_active is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_invoked(self):
        """Test all callbacks are invoked on trigger."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            callback1_called = []
            callback2_called = []

            def callback1(state):
                callback1_called.append(state)

            def callback2(state):
                callback2_called.append(state)

            ks.register_callback(callback1)
            ks.register_callback(callback2)

            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            assert len(callback1_called) == 1
            assert len(callback2_called) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_callbacks_invoked(self):
        """Test async callbacks are properly invoked."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            async_callback_called = []

            async def async_callback(state):
                async_callback_called.append(state)

            ks.register_async_callback(async_callback)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            assert len(async_callback_called) == 1


class TestKillSwitchDrawdownCheck:
    """Test drawdown monitoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_drawdown_triggers_on_limit(self):
        """Test drawdown check triggers when limit exceeded."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, max_drawdown_pct=10.0)

            # Peak at 10000, current at 8500 = 15% drawdown
            result = await ks.check_drawdown(current_equity=8500.0, peak_equity=10000.0)

            assert result is True
            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.MAX_DRAWDOWN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_drawdown_no_trigger_within_limit(self):
        """Test drawdown check doesn't trigger within limit."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, max_drawdown_pct=10.0)

            # Peak at 10000, current at 9500 = 5% drawdown
            result = await ks.check_drawdown(current_equity=9500.0, peak_equity=10000.0)

            assert result is False
            assert ks.is_active is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_drawdown_updates_peak(self):
        """Test drawdown check updates peak equity."""
        ks = KillSwitch(max_drawdown_pct=10.0)

        await ks.check_drawdown(current_equity=10000.0, peak_equity=10000.0)
        assert ks._peak_equity == 10000.0

        await ks.check_drawdown(current_equity=11000.0, peak_equity=11000.0)
        assert ks._peak_equity == 11000.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_drawdown_zero_peak_no_trigger(self):
        """Test drawdown check with zero peak doesn't trigger."""
        ks = KillSwitch(max_drawdown_pct=10.0)

        result = await ks.check_drawdown(current_equity=5000.0, peak_equity=0.0)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_drawdown_includes_metadata(self):
        """Test drawdown trigger includes detailed metadata."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, max_drawdown_pct=10.0)

            await ks.check_drawdown(current_equity=8500.0, peak_equity=10000.0)

            assert "drawdown_pct" in ks.state.metadata
            assert "current_equity" in ks.state.metadata
            assert "peak_equity" in ks.state.metadata
            assert ks.state.metadata["current_equity"] == 8500.0
            assert ks.state.metadata["peak_equity"] == 10000.0


class TestKillSwitchAPIConnectivity:
    """Test API connectivity monitoring."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_connectivity_triggers_on_disconnect(self):
        """Test API connectivity check triggers when disconnected."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            result = await ks.check_api_connectivity(is_connected=False)

            assert result is True
            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.API_CONNECTIVITY

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_api_connectivity_no_trigger_when_connected(self):
        """Test API connectivity check doesn't trigger when connected."""
        ks = KillSwitch()

        result = await ks.check_api_connectivity(is_connected=True)

        assert result is False
        assert ks.is_active is False


class TestKillSwitchDailyLoss:
    """Test daily loss monitoring edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_daily_loss_no_trigger_within_limit(self):
        """Test daily loss doesn't trigger within limit."""
        ks = KillSwitch(daily_loss_limit=100.0)

        result = await ks.check_daily_loss(-50.0)

        assert result is False
        assert ks.is_active is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_daily_loss_updates_pnl(self):
        """Test daily loss check updates daily PnL tracking."""
        ks = KillSwitch(daily_loss_limit=100.0)

        await ks.check_daily_loss(-75.0)
        assert ks._daily_pnl == -75.0

        await ks.check_daily_loss(-90.0)
        assert ks._daily_pnl == -90.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_daily_loss_includes_metadata(self):
        """Test daily loss trigger includes metadata."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, daily_loss_limit=100.0)

            await ks.check_daily_loss(-150.0)

            assert "daily_pnl" in ks.state.metadata
            assert "limit" in ks.state.metadata
            assert ks.state.metadata["daily_pnl"] == -150.0
            assert ks.state.metadata["limit"] == 100.0


class TestKillSwitchConsecutiveLosses:
    """Test consecutive losses edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_trade_result_no_trigger_below_limit(self):
        """Test consecutive losses below limit doesn't trigger."""
        ks = KillSwitch(consecutive_loss_limit=5)

        await ks.record_trade_result(-10.0)
        await ks.record_trade_result(-20.0)

        assert ks.is_active is False
        assert ks._consecutive_losses == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_trade_result_zero_pnl_resets(self):
        """Test zero PnL trade resets consecutive losses."""
        ks = KillSwitch(consecutive_loss_limit=5)

        await ks.record_trade_result(-10.0)
        await ks.record_trade_result(-20.0)
        assert ks._consecutive_losses == 2

        await ks.record_trade_result(0.0)
        assert ks._consecutive_losses == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_trade_result_includes_metadata(self):
        """Test consecutive loss trigger includes metadata."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file, consecutive_loss_limit=3)

            await ks.record_trade_result(-10.0)
            await ks.record_trade_result(-20.0)
            await ks.record_trade_result(-30.0)

            assert "consecutive_losses" in ks.state.metadata
            assert ks.state.metadata["consecutive_losses"] == 3


class TestKillSwitchOrderSubmission:
    """Test order submission checks."""

    @pytest.mark.unit
    def test_can_submit_order_when_inactive(self):
        """Test can submit orders when kill switch inactive."""
        ks = KillSwitch()

        allowed, reason = ks.can_submit_order()

        assert allowed is True
        assert reason == "OK"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_can_submit_order_blocked_when_active(self):
        """Test order submission blocked when kill switch active."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(KillSwitchReason.MANUAL, "Testing", "test")

            allowed, reason = ks.can_submit_order()

            assert allowed is False
            assert "Testing" in reason

    @pytest.mark.unit
    def test_can_submit_order_blocked_by_file_existence(self):
        """Test order submission blocked if kill file exists."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            kill_file.write_text("Emergency stop")

            ks = KillSwitch(kill_file=kill_file)

            allowed, reason = ks.can_submit_order()

            assert allowed is False
            assert str(kill_file) in reason


class TestKillSwitchStatusReporting:
    """Test comprehensive status reporting."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_status_includes_all_fields(self):
        """Test get_status includes all expected fields."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(
                kill_file=kill_file,
                daily_loss_limit=500.0,
                max_drawdown_pct=5.0,
                consecutive_loss_limit=3,
            )

            status = ks.get_status()

            assert "active" in status
            assert "reason" in status
            assert "message" in status
            assert "timestamp" in status
            assert "triggered_by" in status
            assert "kill_file_exists" in status
            assert "daily_pnl" in status
            assert "consecutive_losses" in status
            assert "limits" in status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_status_when_active(self):
        """Test get_status shows details when active."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(KillSwitchReason.BROKER_ERROR, "Broker down", "monitor")
            status = ks.get_status()

            assert status["active"] is True
            assert status["reason"] == "broker_error"
            assert status["message"] == "Broker down"
            assert status["triggered_by"] == "monitor"
            assert status["timestamp"] is not None


class TestKillSwitchDatabaseIntegration:
    """Test database persistence integration."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_loads_from_database(self):
        """Test initialize loads active state from database."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()
            mock_repo.get_kill_switch_info.return_value = {
                "active": True,
                "reason": "daily_loss_limit",
                "timestamp": "2025-12-13T10:00:00",
            }
            mock_repo.get_daily_loss_limit.return_value = 500.0
            mock_repo.get_max_drawdown_pct.return_value = 3.0

            ks = KillSwitch(
                kill_file=kill_file,
                system_state_repo=mock_repo,
                daily_loss_limit=500.0,
                max_drawdown_pct=3.0,
            )
            await ks.initialize()

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.DAILY_LOSS_LIMIT

            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_loads_unknown_reason_from_database(self):
        """Test initialize handles unknown reason from database."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()
            mock_repo.get_kill_switch_info.return_value = {
                "active": True,
                "reason": "invalid_reason_not_in_enum",
                "timestamp": "2025-12-13T10:00:00",
            }
            mock_repo.get_daily_loss_limit.return_value = 1000.0
            mock_repo.get_max_drawdown_pct.return_value = 5.0

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            await ks.initialize()

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.UNKNOWN

            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_handles_database_exception(self):
        """Test initialize activates kill switch on database error."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()
            mock_repo.get_kill_switch_info.side_effect = Exception("Database error")

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            result = await ks.initialize()

            assert result is True
            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.SYSTEM_HEALTH

            await ks.shutdown()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_persists_to_database(self):
        """Test trigger saves state to database."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            await ks.trigger(KillSwitchReason.MANUAL, "Test message", "test")

            mock_repo.activate_kill_switch.assert_called_once_with("Test message")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_handles_database_save_failure(self):
        """Test trigger still activates if database save fails."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()
            mock_repo.activate_kill_switch.side_effect = Exception("DB write error")

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            assert ks.is_active is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_persists_to_database(self):
        """Test reset saves deactivated state to database."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
            await ks.reset(force=True)

            mock_repo.deactivate_kill_switch.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_handles_database_save_failure(self):
        """Test reset still deactivates if database save fails."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            mock_repo = AsyncMock()
            mock_repo.deactivate_kill_switch.side_effect = Exception("DB write error")

            ks = KillSwitch(kill_file=kill_file, system_state_repo=mock_repo)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
            await ks.reset(force=True)

            assert ks.is_active is False


class TestKillSwitchFileOperationFailures:
    """Test file operation error handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_handles_file_write_failure(self):
        """Test trigger still activates if file write fails."""
        with TemporaryDirectory() as tmpdir:
            # Create a read-only directory to cause write failure
            kill_file = Path(tmpdir) / "readonly" / "KILL_SWITCH"
            kill_file.parent.mkdir()

            if os.name != "nt":
                kill_file.parent.chmod(0o444)

            ks = KillSwitch(kill_file=kill_file)

            try:
                await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")
                assert ks.is_active is True
            finally:
                if os.name != "nt":
                    kill_file.parent.chmod(0o755)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_handles_file_delete_failure(self):
        """Test reset still deactivates if file delete fails."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"

            ks = KillSwitch(kill_file=kill_file)
            await ks.trigger(KillSwitchReason.MANUAL, "Test", "test")

            kill_file.unlink()

            result = await ks.reset(force=True)
            assert result is True
            assert ks.is_active is False


class TestKillSwitchRiskTriggerVariations:
    """Test various trigger reason scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_position_reconciliation(self):
        """Test position reconciliation trigger."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(
                KillSwitchReason.POSITION_RECONCILIATION,
                "Position mismatch detected",
                "reconciliation_monitor",
            )

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.POSITION_RECONCILIATION

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_system_health(self):
        """Test system health trigger."""
        with TemporaryDirectory() as tmpdir:
            kill_file = Path(tmpdir) / "KILL_SWITCH"
            ks = KillSwitch(kill_file=kill_file)

            await ks.trigger(
                KillSwitchReason.SYSTEM_HEALTH,
                "System degraded",
                "health_monitor",
            )

            assert ks.is_active is True
            assert ks.state.reason == KillSwitchReason.SYSTEM_HEALTH

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_reason_enums_work(self):
        """Test all KillSwitchReason enum values can trigger."""
        reasons = [
            KillSwitchReason.MANUAL,
            KillSwitchReason.DAILY_LOSS_LIMIT,
            KillSwitchReason.MAX_DRAWDOWN,
            KillSwitchReason.CONSECUTIVE_LOSSES,
            KillSwitchReason.API_CONNECTIVITY,
            KillSwitchReason.SYSTEM_HEALTH,
            KillSwitchReason.FILE_TRIGGER,
            KillSwitchReason.POSITION_RECONCILIATION,
            KillSwitchReason.BROKER_ERROR,
            KillSwitchReason.UNKNOWN,
        ]

        for reason in reasons:
            with TemporaryDirectory() as tmpdir:
                kill_file = Path(tmpdir) / "KILL_SWITCH"
                ks = KillSwitch(kill_file=kill_file)

                await ks.trigger(reason, f"Test {reason.value}", "test")

                assert ks.is_active is True
                assert ks.state.reason == reason
