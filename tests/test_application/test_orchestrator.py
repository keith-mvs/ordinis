import asyncio
from datetime import datetime
from pathlib import Path
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.adapters.storage.database import DatabaseManager
from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.system_state import SystemStateRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository
from ordinis.application.services.orchestrator import (
    OrchestratorConfig,
    OrdinisOrchestrator,
    SystemState,
)
from ordinis.safety.kill_switch import KillSwitch, KillSwitchReason, KillSwitchState


@pytest.fixture
def mock_db():
    db = MagicMock(spec=DatabaseManager)
    db.initialize = AsyncMock(return_value=True)
    db.shutdown = AsyncMock()
    db.is_connected = True
    db.get_table_count = AsyncMock(return_value=10)
    return db


@pytest.fixture
def mock_kill_switch():
    ks = MagicMock(spec=KillSwitch)
    ks.initialize = AsyncMock()
    ks.shutdown = AsyncMock()
    ks.is_active = False
    ks.can_submit_order = MagicMock(return_value=(True, "OK"))
    ks.register_async_callback = MagicMock()
    ks.state = KillSwitchState(
        active=False, reason=KillSwitchReason.UNKNOWN, message="", timestamp=None
    )
    return ks


@pytest.fixture
def mock_repos():
    return {
        "system_state": MagicMock(spec=SystemStateRepository),
        "order": MagicMock(spec=OrderRepository),
        "position": MagicMock(spec=PositionRepository),
        "trade": MagicMock(spec=TradeRepository),
    }


@pytest.fixture
def orchestrator(mock_db, mock_kill_switch, mock_repos):
    config = OrchestratorConfig(
        db_path=Path(":memory:"),
        reconciliation_on_startup=False,
        cancel_stale_orders=True,
        health_check_interval_seconds=0.1,
    )

    orch = OrdinisOrchestrator(config)

    # Inject mocks directly into components
    orch._components.db = mock_db
    orch._components.kill_switch = mock_kill_switch
    orch._components.system_state_repo = mock_repos["system_state"]
    orch._components.order_repo = mock_repos["order"]
    orch._components.position_repo = mock_repos["position"]
    orch._components.trade_repo = mock_repos["trade"]

    # Setup mock repos return values
    mock_repos["system_state"].was_clean_shutdown = AsyncMock(return_value=True)
    mock_repos["system_state"].record_startup = AsyncMock()
    mock_repos["system_state"].record_shutdown = AsyncMock()
    mock_repos["system_state"].record_checkpoint = AsyncMock()

    mock_repos["order"].cancel_all_active = AsyncMock(return_value=5)

    mock_repos["position"].get_total_realized_pnl = AsyncMock(return_value=1000.0)
    mock_repos["position"].create_snapshot = AsyncMock()

    return orch


@pytest.mark.asyncio
async def test_initialization(orchestrator):
    assert orchestrator.state == SystemState.UNINITIALIZED
    assert not orchestrator.is_running
    assert orchestrator.config.db_path == Path(":memory:")


@pytest.mark.asyncio
async def test_start_success(orchestrator, mock_db, mock_kill_switch, mock_repos):
    started = await orchestrator.start()

    assert started is True
    assert orchestrator.state == SystemState.RUNNING
    assert orchestrator.is_running

    # Verify initialization steps
    mock_db.initialize.assert_called_once()
    mock_kill_switch.initialize.assert_called_once()
    mock_repos["system_state"].was_clean_shutdown.assert_called_once()
    mock_repos["order"].cancel_all_active.assert_called_once()  # Stale orders
    mock_repos["system_state"].record_startup.assert_called_once()

    # Cleanup
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_start_failure_db(orchestrator, mock_db):
    mock_db.initialize.return_value = False

    started = await orchestrator.start()

    assert started is False
    assert orchestrator.state == SystemState.INITIALIZING


@pytest.mark.asyncio
async def test_start_failure_kill_switch_active(orchestrator, mock_kill_switch):
    mock_kill_switch.is_active = True

    started = await orchestrator.start()

    assert started is False
    assert orchestrator.state == SystemState.ERROR


@pytest.mark.asyncio
async def test_stop_success(orchestrator, mock_db, mock_kill_switch, mock_repos):
    await orchestrator.start()
    stopped = await orchestrator.stop("Test Shutdown")

    assert stopped is True
    assert orchestrator.state == SystemState.STOPPED

    # Verify shutdown steps
    mock_repos["order"].cancel_all_active.assert_called_with("Test Shutdown")
    mock_repos["position"].create_snapshot.assert_called_once()
    mock_repos["system_state"].record_shutdown.assert_called_once()
    mock_kill_switch.shutdown.assert_called_once()
    mock_db.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_can_trade(orchestrator, mock_kill_switch):
    # Not running
    allowed, reason = orchestrator.can_trade()
    assert not allowed
    assert "not running" in reason

    # Running
    await orchestrator.start()
    allowed, reason = orchestrator.can_trade()
    assert allowed
    assert reason == "OK"

    # Kill switch active
    mock_kill_switch.can_submit_order.return_value = (False, "Kill Switch Active")
    allowed, reason = orchestrator.can_trade()
    assert not allowed
    assert reason == "Kill Switch Active"

    await orchestrator.stop()


@pytest.mark.asyncio
async def test_health_check_loop(orchestrator, mock_repos):
    # Reduce interval for test
    orchestrator.config.health_check_interval_seconds = 0.01

    await orchestrator.start()

    # Let loop run for a bit
    await asyncio.sleep(0.05)

    await orchestrator.stop()

    # Verify health check actions
    assert mock_repos["system_state"].record_checkpoint.call_count >= 1


@pytest.mark.asyncio
async def test_kill_switch_trigger_handling(orchestrator, mock_kill_switch, mock_repos):
    await orchestrator.start()

    # Simulate trigger
    # Get the callback function passed to register_async_callback
    args, _ = mock_kill_switch.register_async_callback.call_args
    callback = args[0]

    trigger_state = KillSwitchState(
        active=True,
        timestamp=datetime.utcnow(),
        reason=KillSwitchReason.DAILY_LOSS_LIMIT,
        message="Loss Limit Exceeded",
    )

    await callback(trigger_state)

    # Verify actions
    # cancel_all_active called twice: once on startup, once on trigger
    assert mock_repos["order"].cancel_all_active.call_count == 2
    args, _ = mock_repos["order"].cancel_all_active.call_args
    assert args[0] == "Kill switch triggered: Loss Limit Exceeded"

    await orchestrator.stop()


@pytest.mark.asyncio
async def test_signal_handling_windows(orchestrator):
    # Mock sys.platform
    with patch("sys.platform", "win32"), patch("signal.signal") as mock_signal:
        await orchestrator.start()

        # Verify signal registration
        assert mock_signal.call_count == 2  # SIGINT, SIGTERM

        # Simulate signal
        # We need to access the handler passed to signal.signal
        # args[0] is signum, args[1] is handler
        handler = mock_signal.call_args[0][1]

        # Call handler
        with patch("asyncio.create_task") as mock_create_task:
            handler(signal.SIGINT, None)
            mock_create_task.assert_called_once()

        await orchestrator.stop()


@pytest.mark.asyncio
async def test_run_loop(orchestrator):
    await orchestrator.start()

    # Run loop in background task
    task = asyncio.create_task(orchestrator.run())

    await asyncio.sleep(0.1)

    # Stop should exit loop
    await orchestrator.stop()
    await task

    assert task.done()


@pytest.mark.asyncio
async def test_get_status(orchestrator):
    status = orchestrator.get_status()
    assert status["state"] == "uninitialized"

    await orchestrator.start()
    status = orchestrator.get_status()
    assert status["state"] == "running"
    assert status["startup_time"] is not None
    assert status["uptime_seconds"] >= 0

    await orchestrator.stop()
