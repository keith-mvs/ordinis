"""
System state repository for kill switch and runtime state persistence.

Provides operations for:
- Kill switch state management
- System checkpoints
- Configuration persistence
- Runtime state tracking
"""

from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

from ordinis.adapters.storage.models import SystemStateRow

if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class SystemStateRepository:
    """Repository for system state operations."""

    # Known state keys
    KEY_KILL_SWITCH_ACTIVE = "kill_switch_active"
    KEY_KILL_SWITCH_REASON = "kill_switch_reason"
    KEY_KILL_SWITCH_TIMESTAMP = "kill_switch_timestamp"
    KEY_LAST_STARTUP = "last_startup"
    KEY_LAST_SHUTDOWN = "last_shutdown"
    KEY_LAST_CHECKPOINT = "last_checkpoint"
    KEY_TRADING_ENABLED = "trading_enabled"
    KEY_DAILY_LOSS_LIMIT = "daily_loss_limit"
    KEY_MAX_DRAWDOWN_PCT = "max_drawdown_pct"

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize system state repository.

        Args:
            db: Database manager instance
        """
        self.db = db

    async def get(self, key: str) -> SystemStateRow | None:
        """
        Get state by key.

        Args:
            key: State key

        Returns:
            SystemStateRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM system_state WHERE key = ?",
            (key,),
        )
        return SystemStateRow.from_row(row) if row else None

    async def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get typed value for key.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            Typed value or default
        """
        state = await self.get(key)
        if state:
            return state.get_typed_value()
        return default

    async def get_all(self) -> list[SystemStateRow]:
        """
        Get all state entries.

        Returns:
            List of SystemStateRow
        """
        rows = await self.db.fetch_all("SELECT * FROM system_state ORDER BY key")
        return [SystemStateRow.from_row(row) for row in rows]

    async def set(
        self,
        key: str,
        value: Any,
        value_type: str = "string",
        description: str | None = None,
    ) -> bool:
        """
        Set state value.

        Args:
            key: State key
            value: Value to set
            value_type: Type hint (string, json, int, float, bool)
            description: Optional description

        Returns:
            True if successful
        """
        try:
            # Convert value to string
            if value_type == "json":
                str_value = json.dumps(value)
            elif value_type == "bool":
                str_value = "true" if value else "false"
            else:
                str_value = str(value)

            await self.db.execute(
                """
                INSERT INTO system_state (key, value, value_type, description)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    value_type = excluded.value_type,
                    description = COALESCE(excluded.description, system_state.description),
                    updated_at = datetime('now')
                """,
                (key, str_value, value_type, description),
            )
            await self.db.commit()

            await self._log_audit("set", key, str_value)
            return True
        except Exception as e:
            logger.exception(f"Failed to set state {key}: {e}")
            await self.db.rollback()
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete state entry.

        Args:
            key: State key

        Returns:
            True if deleted
        """
        try:
            await self.db.execute(
                "DELETE FROM system_state WHERE key = ?",
                (key,),
            )
            await self.db.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to delete state {key}: {e}")
            await self.db.rollback()
            return False

    # ==================== KILL SWITCH OPERATIONS ====================

    async def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is active.

        Returns:
            True if kill switch is active
        """
        return await self.get_value(self.KEY_KILL_SWITCH_ACTIVE, False)

    async def activate_kill_switch(self, reason: str) -> bool:
        """
        Activate kill switch.

        Args:
            reason: Reason for activation

        Returns:
            True if activated
        """
        try:
            now = datetime.utcnow().isoformat()

            await self.db.begin_transaction()

            await self.set(self.KEY_KILL_SWITCH_ACTIVE, True, "bool")
            await self.set(self.KEY_KILL_SWITCH_REASON, reason, "string")
            await self.set(self.KEY_KILL_SWITCH_TIMESTAMP, now, "string")
            await self.set(self.KEY_TRADING_ENABLED, False, "bool")

            await self.db.commit()

            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
            await self._log_audit(
                "kill_switch_activate",
                "kill_switch",
                json.dumps({"reason": reason, "timestamp": now}),
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to activate kill switch: {e}")
            await self.db.rollback()
            return False

    async def deactivate_kill_switch(self) -> bool:
        """
        Deactivate kill switch.

        Returns:
            True if deactivated
        """
        try:
            await self.db.begin_transaction()

            await self.set(self.KEY_KILL_SWITCH_ACTIVE, False, "bool")
            await self.set(self.KEY_KILL_SWITCH_REASON, "", "string")
            await self.set(self.KEY_TRADING_ENABLED, True, "bool")

            await self.db.commit()

            logger.warning("Kill switch deactivated")
            await self._log_audit("kill_switch_deactivate", "kill_switch", "")
            return True
        except Exception as e:
            logger.exception(f"Failed to deactivate kill switch: {e}")
            await self.db.rollback()
            return False

    async def get_kill_switch_info(self) -> dict[str, Any]:
        """
        Get kill switch status information.

        Returns:
            Dictionary with kill switch info
        """
        return {
            "active": await self.get_value(self.KEY_KILL_SWITCH_ACTIVE, False),
            "reason": await self.get_value(self.KEY_KILL_SWITCH_REASON, ""),
            "timestamp": await self.get_value(self.KEY_KILL_SWITCH_TIMESTAMP, ""),
            "trading_enabled": await self.get_value(self.KEY_TRADING_ENABLED, True),
        }

    # ==================== CHECKPOINT OPERATIONS ====================

    async def record_startup(self) -> bool:
        """
        Record system startup timestamp.

        Returns:
            True if recorded
        """
        now = datetime.utcnow().isoformat()
        return await self.set(self.KEY_LAST_STARTUP, now, "string")

    async def record_shutdown(self) -> bool:
        """
        Record system shutdown timestamp.

        Returns:
            True if recorded
        """
        now = datetime.utcnow().isoformat()
        return await self.set(self.KEY_LAST_SHUTDOWN, now, "string")

    async def record_checkpoint(self) -> bool:
        """
        Record checkpoint timestamp.

        Returns:
            True if recorded
        """
        now = datetime.utcnow().isoformat()
        return await self.set(self.KEY_LAST_CHECKPOINT, now, "string")

    async def get_last_startup(self) -> str | None:
        """Get last startup timestamp."""
        return await self.get_value(self.KEY_LAST_STARTUP)

    async def get_last_shutdown(self) -> str | None:
        """Get last shutdown timestamp."""
        return await self.get_value(self.KEY_LAST_SHUTDOWN)

    async def was_clean_shutdown(self) -> bool:
        """
        Check if last session had clean shutdown.

        Returns:
            True if last shutdown was after last startup
        """
        startup = await self.get_last_startup()
        shutdown = await self.get_last_shutdown()

        if not startup:
            return True  # No previous startup
        if not shutdown:
            return False  # Started but never shut down

        # Parse ISO timestamps for proper comparison
        try:
            startup_dt = datetime.fromisoformat(startup)
            shutdown_dt = datetime.fromisoformat(shutdown)
            return shutdown_dt >= startup_dt
        except ValueError:
            # Fallback to string comparison if parsing fails
            logger.warning("Failed to parse timestamps, using string comparison")
            return shutdown >= startup

    # ==================== RISK LIMIT OPERATIONS ====================

    async def get_daily_loss_limit(self) -> float:
        """Get daily loss limit."""
        return await self.get_value(self.KEY_DAILY_LOSS_LIMIT, 1000.0)

    async def set_daily_loss_limit(self, limit: float) -> bool:
        """Set daily loss limit."""
        return await self.set(self.KEY_DAILY_LOSS_LIMIT, limit, "float")

    async def get_max_drawdown_pct(self) -> float:
        """Get maximum drawdown percentage."""
        return await self.get_value(self.KEY_MAX_DRAWDOWN_PCT, 5.0)

    async def set_max_drawdown_pct(self, pct: float) -> bool:
        """Set maximum drawdown percentage."""
        return await self.set(self.KEY_MAX_DRAWDOWN_PCT, pct, "float")

    async def is_trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        return await self.get_value(self.KEY_TRADING_ENABLED, True)

    async def set_trading_enabled(self, enabled: bool) -> bool:
        """Set trading enabled status."""
        return await self.set(self.KEY_TRADING_ENABLED, enabled, "bool")

    # ==================== CUSTOM STATE OPERATIONS ====================

    async def set_json(self, key: str, value: dict | list, description: str | None = None) -> bool:
        """
        Set JSON state value.

        Args:
            key: State key
            value: JSON-serializable value
            description: Optional description

        Returns:
            True if successful
        """
        return await self.set(key, value, "json", description)

    async def get_json(self, key: str, default: dict | list | None = None) -> dict | list | None:
        """
        Get JSON state value.

        Args:
            key: State key
            default: Default value

        Returns:
            Parsed JSON or default
        """
        state = await self.get(key)
        if state and state.value_type == "json":
            return state.get_typed_value()
        return default

    async def _log_audit(
        self,
        action: str,
        entity_id: str,
        new_value: str,
        old_value: str | None = None,
    ) -> None:
        """Log audit event for state change."""
        try:
            await self.db.execute(
                """
                INSERT INTO persistence_audit (
                    event_type, entity_type, entity_id, action, old_value, new_value
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("state_change", "system_state", entity_id, action, old_value, new_value),
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
