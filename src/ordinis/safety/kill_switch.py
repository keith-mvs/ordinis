"""
Kill Switch implementation for emergency trading halt.

Features:
- Persistent state (survives restart)
- File-based fallback (touch file to halt)
- Multiple trigger conditions
- Atomic state transitions
- Integration with FlowRoute engine

The kill switch has three trigger sources:
1. Programmatic: API calls, risk breaches
2. File-based: Touch KILL_SWITCH file (emergency fallback)
3. Database: Persisted state from previous session
"""

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager
    from ordinis.adapters.storage.repositories.system_state import SystemStateRepository

logger = logging.getLogger(__name__)

# Default kill switch file location
DEFAULT_KILL_FILE = Path("data/KILL_SWITCH")


class KillSwitchReason(Enum):
    """Reasons for kill switch activation."""

    MANUAL = "manual"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    API_CONNECTIVITY = "api_connectivity"
    SYSTEM_HEALTH = "system_health"
    FILE_TRIGGER = "file_trigger"
    POSITION_RECONCILIATION = "position_reconciliation"
    BROKER_ERROR = "broker_error"
    UNKNOWN = "unknown"


@dataclass
class KillSwitchState:
    """Current state of the kill switch."""

    active: bool = False
    reason: KillSwitchReason = KillSwitchReason.UNKNOWN
    message: str = ""
    timestamp: datetime | None = None
    triggered_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class KillSwitch:
    """
    Emergency kill switch for live trading.

    Provides multiple trigger sources and persistent state.
    When activated:
    - Blocks all new order submissions
    - Signals to cancel all open orders
    - Persists state to database
    - Creates file marker for visibility
    - Notifies registered callbacks
    """

    def __init__(
        self,
        db: "DatabaseManager | None" = None,
        system_state_repo: "SystemStateRepository | None" = None,
        kill_file: Path | None = None,
        daily_loss_limit: float = 1000.0,
        max_drawdown_pct: float = 5.0,
        consecutive_loss_limit: int = 5,
        check_interval_seconds: float = 1.0,
    ):
        """
        Initialize kill switch.

        Args:
            db: Database manager for persistence
            system_state_repo: System state repository
            kill_file: Path to kill switch file (fallback trigger)
            daily_loss_limit: Maximum daily loss before auto-trigger
            max_drawdown_pct: Maximum drawdown percentage
            consecutive_loss_limit: Maximum consecutive losing trades
            check_interval_seconds: Interval for file check loop
        """
        self._db = db
        self._repo = system_state_repo
        self._kill_file = kill_file or DEFAULT_KILL_FILE
        self._daily_loss_limit = daily_loss_limit
        self._max_drawdown_pct = max_drawdown_pct
        self._consecutive_loss_limit = consecutive_loss_limit
        self._check_interval = check_interval_seconds

        self._state = KillSwitchState()
        self._callbacks: list[Callable[[KillSwitchState], None]] = []
        self._async_callbacks: list[Callable[[KillSwitchState], Any]] = []
        self._lock = asyncio.Lock()
        self._file_check_task: asyncio.Task | None = None
        self._running = False

        # Statistics
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0

    @property
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self._state.active

    @property
    def state(self) -> KillSwitchState:
        """Get current state."""
        return self._state

    async def initialize(self) -> bool:
        """
        Initialize kill switch from persisted state.

        Returns:
            True if initialized (may be active from previous session)
        """
        try:
            # Ensure kill file directory exists
            self._kill_file.parent.mkdir(parents=True, exist_ok=True)

            # Check file-based trigger first (highest priority)
            if self._kill_file.exists():
                await self._trigger(
                    KillSwitchReason.FILE_TRIGGER,
                    "Kill switch file detected on startup",
                    "file_check",
                )
                logger.warning(f"Kill switch file detected: {self._kill_file}")
                return True

            # Load state from database
            if self._repo:
                info = await self._repo.get_kill_switch_info()
                if info["active"]:
                    self._state = KillSwitchState(
                        active=True,
                        reason=KillSwitchReason(info.get("reason", "unknown"))
                        if info.get("reason") in [r.value for r in KillSwitchReason]
                        else KillSwitchReason.UNKNOWN,
                        message=info.get("reason", ""),
                        timestamp=datetime.fromisoformat(info["timestamp"])
                        if info.get("timestamp")
                        else None,
                        triggered_by="database_restore",
                    )
                    logger.warning(f"Kill switch restored from database: {self._state.message}")
                    return True

                # Load risk limits from database
                self._daily_loss_limit = await self._repo.get_daily_loss_limit()
                self._max_drawdown_pct = await self._repo.get_max_drawdown_pct()

            # Start file monitoring
            self._running = True
            self._file_check_task = asyncio.create_task(self._file_check_loop())

            logger.info("Kill switch initialized (inactive)")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize kill switch: {e}")
            # Fail safe: activate if initialization fails
            await self._trigger(
                KillSwitchReason.SYSTEM_HEALTH,
                f"Kill switch initialization failed: {e}",
                "initialization",
            )
            return True

    async def shutdown(self) -> None:
        """Stop kill switch monitoring."""
        self._running = False
        if self._file_check_task:
            self._file_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._file_check_task
        logger.info("Kill switch monitoring stopped")

    async def trigger(
        self,
        reason: KillSwitchReason,
        message: str,
        triggered_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Trigger the kill switch.

        Args:
            reason: Reason for activation
            message: Human-readable message
            triggered_by: Component that triggered
            metadata: Additional context

        Returns:
            True if triggered (or was already active)
        """
        return await self._trigger(reason, message, triggered_by, metadata)

    async def _trigger(
        self,
        reason: KillSwitchReason,
        message: str,
        triggered_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Internal trigger implementation."""
        async with self._lock:
            if self._state.active:
                logger.debug("Kill switch already active, ignoring trigger")
                return True

            self._state = KillSwitchState(
                active=True,
                reason=reason,
                message=message,
                timestamp=datetime.utcnow(),
                triggered_by=triggered_by,
                metadata=metadata or {},
            )

            # Persist to database
            if self._repo:
                try:
                    await self._repo.activate_kill_switch(message)
                except Exception as e:
                    logger.exception(f"Failed to persist kill switch state: {e}")

            # Create file marker
            try:
                self._kill_file.write_text(
                    f"Kill switch activated: {message}\n"
                    f"Reason: {reason.value}\n"
                    f"Time: {self._state.timestamp.isoformat()}\n"
                    f"Triggered by: {triggered_by}\n"
                )
            except Exception as e:
                logger.exception(f"Failed to create kill switch file: {e}")

            logger.critical(f"KILL SWITCH ACTIVATED - Reason: {reason.value} - {message}")

            # Notify callbacks
            await self._notify_callbacks()

            return True

    async def reset(self, force: bool = False) -> bool:
        """
        Reset kill switch (deactivate).

        Args:
            force: Force reset even if conditions haven't cleared

        Returns:
            True if reset successfully
        """
        async with self._lock:
            if not self._state.active:
                return True

            if not force:
                # Check if conditions have cleared
                if self._daily_pnl < -self._daily_loss_limit:
                    logger.warning("Cannot reset: daily loss limit still exceeded")
                    return False

            # Reset state
            old_state = self._state
            self._state = KillSwitchState()

            # Persist to database
            if self._repo:
                try:
                    await self._repo.deactivate_kill_switch()
                except Exception as e:
                    logger.exception(f"Failed to persist kill switch reset: {e}")

            # Remove file marker
            try:
                if self._kill_file.exists():
                    self._kill_file.unlink()
            except Exception as e:
                logger.exception(f"Failed to remove kill switch file: {e}")

            logger.warning(f"Kill switch RESET - Was active for reason: {old_state.reason.value}")

            return True

    def register_callback(
        self,
        callback: Callable[[KillSwitchState], None],
    ) -> None:
        """
        Register callback for kill switch activation.

        Args:
            callback: Function to call when kill switch triggers
        """
        self._callbacks.append(callback)

    def register_async_callback(
        self,
        callback: Callable[[KillSwitchState], Any],
    ) -> None:
        """
        Register async callback for kill switch activation.

        Args:
            callback: Async function to call when kill switch triggers
        """
        self._async_callbacks.append(callback)

    async def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception as e:
                logger.exception(f"Kill switch callback error: {e}")

        for async_callback in self._async_callbacks:
            try:
                await async_callback(self._state)
            except Exception as e:
                logger.exception(f"Kill switch async callback error: {e}")

    async def _file_check_loop(self) -> None:
        """Background loop to check for file-based trigger."""
        while self._running:
            try:
                if not self._state.active and self._kill_file.exists():
                    await self._trigger(
                        KillSwitchReason.FILE_TRIGGER,
                        "Kill switch file detected",
                        "file_monitor",
                    )
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in file check loop: {e}")
                await asyncio.sleep(self._check_interval)

    # ==================== AUTOMATED TRIGGERS ====================

    async def check_daily_loss(self, daily_pnl: float) -> bool:
        """
        Check if daily loss limit exceeded.

        Args:
            daily_pnl: Current daily P&L (negative for loss)

        Returns:
            True if kill switch triggered
        """
        self._daily_pnl = daily_pnl

        if daily_pnl < -self._daily_loss_limit:
            await self._trigger(
                KillSwitchReason.DAILY_LOSS_LIMIT,
                f"Daily loss limit exceeded: ${abs(daily_pnl):.2f} > ${self._daily_loss_limit:.2f}",
                "risk_monitor",
                {"daily_pnl": daily_pnl, "limit": self._daily_loss_limit},
            )
            return True
        return False

    async def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """
        Check if maximum drawdown exceeded.

        Args:
            current_equity: Current portfolio equity
            peak_equity: Peak equity (high water mark)

        Returns:
            True if kill switch triggered
        """
        self._current_equity = current_equity
        self._peak_equity = max(self._peak_equity, peak_equity)

        if self._peak_equity > 0:
            drawdown_pct = ((self._peak_equity - current_equity) / self._peak_equity) * 100

            if drawdown_pct >= self._max_drawdown_pct:
                await self._trigger(
                    KillSwitchReason.MAX_DRAWDOWN,
                    f"Maximum drawdown exceeded: {drawdown_pct:.2f}% > {self._max_drawdown_pct:.2f}%",
                    "risk_monitor",
                    {
                        "drawdown_pct": drawdown_pct,
                        "limit": self._max_drawdown_pct,
                        "current_equity": current_equity,
                        "peak_equity": peak_equity,
                    },
                )
                return True
        return False

    async def record_trade_result(self, pnl: float) -> bool:
        """
        Record trade result for consecutive loss tracking.

        Args:
            pnl: Trade P&L

        Returns:
            True if kill switch triggered
        """
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._consecutive_loss_limit:
                await self._trigger(
                    KillSwitchReason.CONSECUTIVE_LOSSES,
                    f"Consecutive loss limit reached: {self._consecutive_losses} losses",
                    "risk_monitor",
                    {"consecutive_losses": self._consecutive_losses},
                )
                return True
        else:
            self._consecutive_losses = 0
        return False

    async def check_api_connectivity(self, is_connected: bool) -> bool:
        """
        Check API connectivity status.

        Args:
            is_connected: Whether API is connected

        Returns:
            True if kill switch triggered
        """
        if not is_connected:
            await self._trigger(
                KillSwitchReason.API_CONNECTIVITY,
                "Lost connectivity to broker API",
                "connectivity_monitor",
            )
            return True
        return False

    # ==================== STATUS METHODS ====================

    def get_status(self) -> dict[str, Any]:
        """Get kill switch status summary."""
        return {
            "active": self._state.active,
            "reason": self._state.reason.value if self._state.active else None,
            "message": self._state.message if self._state.active else None,
            "timestamp": self._state.timestamp.isoformat() if self._state.timestamp else None,
            "triggered_by": self._state.triggered_by if self._state.active else None,
            "kill_file_exists": self._kill_file.exists(),
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "limits": {
                "daily_loss": self._daily_loss_limit,
                "max_drawdown_pct": self._max_drawdown_pct,
                "consecutive_loss": self._consecutive_loss_limit,
            },
        }

    def can_submit_order(self) -> tuple[bool, str]:
        """
        Check if order submission is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if self._state.active:
            return False, f"Kill switch active: {self._state.message}"

        # Also check file trigger in case loop hasn't caught it
        if self._kill_file.exists() and not self._state.active:
            return False, f"Kill switch file exists: {self._kill_file}"

        return True, "OK"
