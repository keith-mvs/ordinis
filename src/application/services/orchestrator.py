"""
Central orchestrator for Ordinis trading system.

Coordinates:
- System startup and shutdown
- Engine lifecycle management
- Component health monitoring
- Kill switch integration
- Position reconciliation
"""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import signal
from typing import TYPE_CHECKING, Any

from persistence.database import DatabaseManager, get_database
from persistence.repositories.order import OrderRepository
from persistence.repositories.position import PositionRepository
from persistence.repositories.system_state import SystemStateRepository
from persistence.repositories.trade import TradeRepository
from safety.kill_switch import KillSwitch

if TYPE_CHECKING:
    from alerting.manager import AlertManager

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System lifecycle states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""

    db_path: Path = Path("data/ordinis.db")
    backup_dir: Path = Path("data/backups")
    kill_file: Path = Path("data/KILL_SWITCH")
    daily_loss_limit: float = 1000.0
    max_drawdown_pct: float = 5.0
    consecutive_loss_limit: int = 5
    reconciliation_on_startup: bool = True
    cancel_stale_orders: bool = True
    shutdown_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0


@dataclass
class SystemComponents:
    """Container for system components."""

    db: DatabaseManager | None = None
    position_repo: PositionRepository | None = None
    order_repo: OrderRepository | None = None
    trade_repo: TradeRepository | None = None
    system_state_repo: SystemStateRepository | None = None
    kill_switch: KillSwitch | None = None
    alert_manager: "AlertManager | None" = None
    broker_adapter: Any = None
    signal_engine: Any = None
    risk_engine: Any = None


class OrdinisOrchestrator:
    """
    Central orchestrator for Ordinis trading system.

    Manages system lifecycle including:
    - Database initialization
    - Kill switch setup
    - Position reconciliation
    - Graceful shutdown

    Usage:
        orchestrator = OrdinisOrchestrator(config)
        if await orchestrator.start():
            # System is running
            await orchestrator.run()
        await orchestrator.stop()
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self._state = SystemState.UNINITIALIZED
        self._components = SystemComponents()
        self._startup_time: datetime | None = None
        self._shutdown_requested = False
        self._running_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None  # Windows signal handler
        self._lock = asyncio.Lock()

        # Statistics
        self._stats: dict[str, Any] = {
            "startup_count": 0,
            "shutdown_count": 0,
            "errors": [],
        }

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._state == SystemState.RUNNING

    @property
    def components(self) -> SystemComponents:
        """Get system components."""
        return self._components

    async def start(self) -> bool:
        """
        Start the trading system.

        Performs:
        1. Database initialization
        2. Kill switch check
        3. Position reconciliation
        4. Component startup

        Returns:
            True if started successfully
        """
        async with self._lock:
            if self._state not in [SystemState.UNINITIALIZED, SystemState.STOPPED]:
                logger.warning(f"Cannot start from state: {self._state.value}")
                return False

            self._state = SystemState.INITIALIZING
            logger.info("Starting Ordinis trading system...")

            try:
                # Step 1: Initialize database
                if not await self._init_database():
                    return False

                # Step 2: Check and initialize kill switch
                if not await self._init_kill_switch():
                    return False

                # Step 3: Check if kill switch is active
                if self._components.kill_switch and self._components.kill_switch.is_active:
                    logger.warning(
                        "System startup aborted: Kill switch is active. "
                        "Run 'ordinis reset-kill-switch' to clear."
                    )
                    self._state = SystemState.ERROR
                    return False

                # Step 4: Check for clean shutdown
                await self._check_previous_session()

                # Step 5: Initialize repositories
                self._init_repositories()

                # Step 6: Reconcile positions (if enabled)
                if self.config.reconciliation_on_startup:
                    await self._reconcile_positions()

                # Step 7: Cancel stale orders (if enabled)
                if self.config.cancel_stale_orders:
                    await self._cancel_stale_orders()

                # Step 8: Record startup
                if self._components.system_state_repo:
                    await self._components.system_state_repo.record_startup()

                # Step 9: Start health check loop
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                # Step 10: Setup signal handlers
                self._setup_signal_handlers()

                self._state = SystemState.RUNNING
                self._startup_time = datetime.utcnow()
                self._stats["startup_count"] += 1

                logger.info("Ordinis trading system started successfully")
                return True

            except Exception as e:
                logger.exception("Failed to start system")
                self._state = SystemState.ERROR
                self._stats["errors"].append(f"{datetime.utcnow()}: {e}")
                return False

    async def stop(self, reason: str = "Normal shutdown") -> bool:
        """
        Stop the trading system gracefully.

        Performs:
        1. Stop accepting new signals
        2. Cancel pending orders
        3. Save system state
        4. Close connections

        Args:
            reason: Shutdown reason

        Returns:
            True if stopped successfully
        """
        async with self._lock:
            if self._state == SystemState.STOPPED:
                return True

            if self._state not in [SystemState.RUNNING, SystemState.ERROR]:
                logger.warning(f"Cannot stop from state: {self._state.value}")
                return False

            self._state = SystemState.STOPPING
            self._shutdown_requested = True
            logger.info(f"Stopping Ordinis trading system: {reason}")

            try:
                # Step 1: Stop health check
                if self._health_check_task:
                    self._health_check_task.cancel()
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(
                            self._health_check_task,
                            timeout=5.0,
                        )

                # Step 2: Cancel pending orders
                await self._cancel_all_pending_orders(reason)

                # Step 3: Save final state
                await self._save_final_state()

                # Step 4: Record shutdown
                if self._components.system_state_repo:
                    await self._components.system_state_repo.record_shutdown()

                # Step 5: Stop kill switch
                if self._components.kill_switch:
                    await self._components.kill_switch.shutdown()

                # Step 6: Close database
                if self._components.db:
                    await self._components.db.shutdown()

                self._state = SystemState.STOPPED
                self._stats["shutdown_count"] += 1

                logger.info("Ordinis trading system stopped")
                return True

            except Exception as e:
                logger.exception("Error during shutdown")
                self._state = SystemState.ERROR
                return False

    async def run(self) -> None:
        """
        Run the main trading loop.

        This is a blocking call that runs until stop() is called.
        """
        logger.info("Entering main trading loop")
        while self._state == SystemState.RUNNING and not self._shutdown_requested:
            try:
                # Main trading loop - placeholder for signal processing
                await asyncio.sleep(1.0)

                # Check kill switch
                if self._components.kill_switch and self._components.kill_switch.is_active:
                    logger.warning("Kill switch activated during operation")
                    await self._handle_kill_switch_triggered()

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.exception("Error in main loop")
                await asyncio.sleep(1.0)

        logger.info("Exiting main trading loop")

    # ==================== INITIALIZATION HELPERS ====================

    async def _init_database(self) -> bool:
        """Initialize database connection."""
        try:
            self._components.db = get_database(
                db_path=self.config.db_path,
                backup_dir=self.config.backup_dir,
            )
            result = await self._components.db.initialize()
            if result:
                logger.info(f"Database initialized: {self.config.db_path}")
            return result
        except Exception as e:
            logger.exception("Database initialization failed")
            return False

    async def _init_kill_switch(self) -> bool:
        """Initialize kill switch."""
        try:
            if not self._components.db:
                logger.error("Database must be initialized before kill switch")
                return False

            self._components.system_state_repo = SystemStateRepository(self._components.db)

            self._components.kill_switch = KillSwitch(
                db=self._components.db,
                system_state_repo=self._components.system_state_repo,
                kill_file=self.config.kill_file,
                daily_loss_limit=self.config.daily_loss_limit,
                max_drawdown_pct=self.config.max_drawdown_pct,
                consecutive_loss_limit=self.config.consecutive_loss_limit,
            )

            # Register callback for kill switch activation
            self._components.kill_switch.register_async_callback(self._on_kill_switch_triggered)

            await self._components.kill_switch.initialize()
            logger.info("Kill switch initialized")
            return True
        except Exception as e:
            logger.exception("Kill switch initialization failed")
            return False

    def _init_repositories(self) -> None:
        """Initialize data repositories."""
        if self._components.db:
            self._components.position_repo = PositionRepository(self._components.db)
            self._components.order_repo = OrderRepository(self._components.db)
            self._components.trade_repo = TradeRepository(self._components.db)
            logger.info("Repositories initialized")

    async def _check_previous_session(self) -> None:
        """Check if previous session had clean shutdown."""
        if self._components.system_state_repo:
            clean = await self._components.system_state_repo.was_clean_shutdown()
            if not clean:
                logger.warning(
                    "Previous session did not shutdown cleanly. "
                    "Position reconciliation recommended."
                )

    async def _reconcile_positions(self) -> None:
        """Reconcile local positions with broker."""
        # Placeholder - full implementation in reconciliation.py
        logger.info("Position reconciliation skipped (broker not connected)")

    async def _cancel_stale_orders(self) -> None:
        """Cancel any stale orders from previous session."""
        if self._components.order_repo:
            cancelled = await self._components.order_repo.cancel_all_active(
                "Stale order cleanup on startup"
            )
            if cancelled > 0:
                logger.info(f"Cancelled {cancelled} stale orders")

    # ==================== SHUTDOWN HELPERS ====================

    async def _cancel_all_pending_orders(self, reason: str) -> None:
        """Cancel all pending orders."""
        if self._components.order_repo:
            cancelled = await self._components.order_repo.cancel_all_active(reason)
            logger.info(f"Cancelled {cancelled} pending orders")

    async def _save_final_state(self) -> None:
        """Save final system state."""
        if self._components.position_repo:
            # Create portfolio snapshot
            total_pnl = await self._components.position_repo.get_total_realized_pnl()
            # Placeholder for equity calculation
            await self._components.position_repo.create_snapshot(
                cash=0.0,  # Would come from broker
                total_equity=total_pnl,
            )
            logger.info("Final state saved")

    # ==================== KILL SWITCH HANDLING ====================

    async def _on_kill_switch_triggered(self, state: Any) -> None:
        """Handle kill switch activation."""
        logger.critical(f"Kill switch triggered: {state.message}")
        await self._handle_kill_switch_triggered()

    async def _handle_kill_switch_triggered(self) -> None:
        """Handle kill switch being triggered."""
        # Cancel all pending orders
        if self._components.order_repo:
            await self._components.order_repo.cancel_all_active("Kill switch triggered")

        # Send alert if alert manager configured
        if self._components.alert_manager:
            try:
                await self._components.alert_manager.send_emergency(
                    "Kill Switch Activated",
                    self._components.kill_switch.state.message
                    if self._components.kill_switch
                    else "Unknown reason",
                )
            except Exception as e:
                logger.exception("Failed to send kill switch alert")

    # ==================== HEALTH MONITORING ====================

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._state == SystemState.RUNNING:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                # Check database health
                if self._components.db:
                    count = await self._components.db.get_table_count("system_state")
                    if count == 0:
                        logger.warning("Database health check: system_state empty")

                # Record checkpoint
                if self._components.system_state_repo:
                    await self._components.system_state_repo.record_checkpoint()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Health check error")

    # ==================== SIGNAL HANDLERS ====================

    def _setup_signal_handlers(self) -> None:
        """Setup OS signal handlers for graceful shutdown."""
        import sys

        if sys.platform == "win32":
            # Windows-specific: use signal.signal()
            signal.signal(signal.SIGINT, self._sync_signal_handler)
            signal.signal(signal.SIGTERM, self._sync_signal_handler)
            logger.debug("Windows signal handlers configured")
        else:
            # Unix-like: use async handlers
            try:
                loop = asyncio.get_event_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(
                        sig,
                        lambda s=sig: asyncio.create_task(self._signal_handler(s)),
                    )
                logger.debug("Unix signal handlers configured")
            except NotImplementedError:
                logger.warning("Signal handlers not supported on this platform")

    def _sync_signal_handler(self, signum: int, frame: Any) -> None:
        """Synchronous signal handler for Windows."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}, initiating shutdown")
        # Schedule async shutdown from sync context
        if self._state == SystemState.RUNNING:
            # Store task reference to prevent garbage collection
            self._shutdown_task = asyncio.create_task(self.stop(f"Signal {sig_name}"))

    async def _signal_handler(self, sig: signal.Signals) -> None:
        """Async signal handler for Unix."""
        logger.info(f"Received signal {sig.name}, initiating shutdown")
        await self.stop(f"Signal {sig.name}")

    # ==================== STATUS METHODS ====================

    def get_status(self) -> dict[str, Any]:
        """Get system status summary."""
        return {
            "state": self._state.value,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "uptime_seconds": (datetime.utcnow() - self._startup_time).total_seconds()
            if self._startup_time
            else 0,
            "kill_switch_active": self._components.kill_switch.is_active
            if self._components.kill_switch
            else False,
            "database_connected": self._components.db.is_connected
            if self._components.db
            else False,
            "stats": self._stats,
        }

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if self._state != SystemState.RUNNING:
            return False, f"System not running (state: {self._state.value})"

        if self._shutdown_requested:
            return False, "Shutdown in progress"

        if self._components.kill_switch:
            can_submit, reason = self._components.kill_switch.can_submit_order()
            if not can_submit:
                return False, reason

        return True, "OK"
