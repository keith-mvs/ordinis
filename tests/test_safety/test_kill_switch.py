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

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

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
