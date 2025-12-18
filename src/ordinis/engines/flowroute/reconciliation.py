"""
Position reconciliation for FlowRoute engine.

Provides periodic synchronization between internal position tracking
and broker state. Detects and logs discrepancies, auto-corrects
internal state, and emits metrics for monitoring.

Phase 2 implementation (2025-12-17).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ordinis.engines.flowroute.core.engine import FlowRouteEngine

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of a single reconciliation cycle."""

    timestamp: datetime
    success: bool
    positions_checked: int = 0
    discrepancies_found: int = 0
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    auto_corrected: int = 0
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class ReconciliationMetrics:
    """Cumulative metrics for reconciliation operations."""

    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    total_discrepancies: int = 0
    total_auto_corrections: int = 0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    avg_duration_ms: float = 0.0

    def record_cycle(self, result: ReconciliationResult) -> None:
        """Record metrics from a reconciliation cycle."""
        self.total_cycles += 1

        if result.success:
            self.successful_cycles += 1
            self.last_success = result.timestamp
        else:
            self.failed_cycles += 1
            self.last_failure = result.timestamp

        self.total_discrepancies += result.discrepancies_found
        self.total_auto_corrections += result.auto_corrected

        # Rolling average duration
        if self.total_cycles == 1:
            self.avg_duration_ms = result.duration_ms
        else:
            self.avg_duration_ms = (
                self.avg_duration_ms * (self.total_cycles - 1) + result.duration_ms
            ) / self.total_cycles

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "success_rate": self.successful_cycles / self.total_cycles
            if self.total_cycles > 0
            else 0.0,
            "total_discrepancies": self.total_discrepancies,
            "total_auto_corrections": self.total_auto_corrections,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
        }


class PositionReconciler:
    """
    Periodic position reconciliation with broker.

    Runs in background, syncing internal state with broker every
    configured interval. Detects discrepancies and auto-corrects
    internal tracking.
    """

    def __init__(
        self,
        engine: FlowRouteEngine,
        interval_seconds: float = 30.0,
        auto_correct: bool = True,
        on_discrepancy: Callable[[dict[str, Any]], None] | None = None,
    ):
        """
        Initialize reconciler.

        Args:
            engine: FlowRoute engine to reconcile
            interval_seconds: Sync interval (default: 30 seconds)
            auto_correct: Whether to auto-correct discrepancies
            on_discrepancy: Callback when discrepancy detected
        """
        self._engine = engine
        self._interval = interval_seconds
        self._auto_correct = auto_correct
        self._on_discrepancy = on_discrepancy

        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics = ReconciliationMetrics()
        self._history: list[ReconciliationResult] = []
        self._max_history = 100

    async def start(self) -> None:
        """Start periodic reconciliation."""
        if self._running:
            logger.warning("Reconciler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._reconciliation_loop())
        logger.info(f"Position reconciler started (interval: {self._interval}s)")

    async def stop(self) -> None:
        """Stop periodic reconciliation."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Position reconciler stopped")

    async def reconcile_now(self) -> ReconciliationResult:
        """
        Run immediate reconciliation.

        Can be called manually outside of the periodic loop.

        Returns:
            ReconciliationResult with details
        """
        start_time = datetime.utcnow()
        start_ms = asyncio.get_event_loop().time() * 1000

        try:
            # Get current internal state
            internal_positions = {p.symbol: p for p in self._engine.get_all_positions()}

            # Sync with broker (this updates internal state)
            sync_result = await self._engine.sync_broker_state()

            if not sync_result.success:
                result = ReconciliationResult(
                    timestamp=start_time,
                    success=False,
                    error=sync_result.error,
                    duration_ms=asyncio.get_event_loop().time() * 1000 - start_ms,
                )
                self._record_result(result)
                return result

            # Get updated state after sync
            broker_positions = {p.symbol: p for p in self._engine.get_all_positions()}

            # Detect discrepancies
            discrepancies = self._detect_discrepancies(internal_positions, broker_positions)

            # Auto-correct count (sync already corrected internal state)
            auto_corrected = len(discrepancies) if self._auto_correct else 0

            # Notify callback for each discrepancy
            if self._on_discrepancy:
                for disc in discrepancies:
                    try:
                        self._on_discrepancy(disc)
                    except Exception as e:
                        logger.error(f"Discrepancy callback error: {e}")

            result = ReconciliationResult(
                timestamp=start_time,
                success=True,
                positions_checked=len(broker_positions),
                discrepancies_found=len(discrepancies),
                discrepancies=discrepancies,
                auto_corrected=auto_corrected,
                duration_ms=asyncio.get_event_loop().time() * 1000 - start_ms,
            )

            self._record_result(result)

            if discrepancies:
                logger.warning(
                    f"Reconciliation found {len(discrepancies)} discrepancies "
                    f"(auto-corrected: {auto_corrected})"
                )
            else:
                logger.debug(f"Reconciliation complete: {len(broker_positions)} positions OK")

            return result

        except Exception as e:
            result = ReconciliationResult(
                timestamp=start_time,
                success=False,
                error=str(e),
                duration_ms=asyncio.get_event_loop().time() * 1000 - start_ms,
            )
            self._record_result(result)
            logger.exception("Reconciliation failed")
            return result

    def _detect_discrepancies(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Detect discrepancies between before and after states.

        Args:
            before: Internal positions before sync
            after: Broker positions after sync

        Returns:
            List of discrepancy details
        """
        discrepancies = []

        # Check for missing positions (in before but not after)
        for symbol in before:
            if symbol not in after:
                discrepancies.append(
                    {
                        "type": "position_closed",
                        "symbol": symbol,
                        "internal_qty": str(before[symbol].quantity),
                        "broker_qty": "0",
                        "severity": "high",
                    }
                )

        # Check for new positions (in after but not before)
        for symbol in after:
            if symbol not in before:
                discrepancies.append(
                    {
                        "type": "position_opened",
                        "symbol": symbol,
                        "internal_qty": "0",
                        "broker_qty": str(after[symbol].quantity),
                        "severity": "high",
                    }
                )

        # Check for quantity mismatches
        for symbol in before:
            if symbol in after:
                internal_qty = before[symbol].quantity
                broker_qty = after[symbol].quantity

                if internal_qty != broker_qty:
                    discrepancies.append(
                        {
                            "type": "quantity_mismatch",
                            "symbol": symbol,
                            "internal_qty": str(internal_qty),
                            "broker_qty": str(broker_qty),
                            "diff": str(broker_qty - internal_qty),
                            "severity": "medium",
                        }
                    )

        return discrepancies

    def _record_result(self, result: ReconciliationResult) -> None:
        """Record result in history and metrics."""
        self._metrics.record_cycle(result)

        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    async def _reconciliation_loop(self) -> None:
        """Background reconciliation loop."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)

                if not self._running:
                    break

                await self.reconcile_now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Reconciliation loop error: {e}")
                # Continue running despite errors
                await asyncio.sleep(5)

    def get_metrics(self) -> ReconciliationMetrics:
        """Get reconciliation metrics."""
        return self._metrics

    def get_history(self, limit: int = 10) -> list[ReconciliationResult]:
        """Get recent reconciliation history."""
        return self._history[-limit:]

    def is_running(self) -> bool:
        """Check if reconciler is running."""
        return self._running

    def to_dict(self) -> dict[str, Any]:
        """Get reconciler state as dictionary."""
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "auto_correct": self._auto_correct,
            "metrics": self._metrics.to_dict(),
            "recent_results": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "success": r.success,
                    "positions_checked": r.positions_checked,
                    "discrepancies_found": r.discrepancies_found,
                    "duration_ms": round(r.duration_ms, 2),
                }
                for r in self._history[-5:]
            ],
        }
