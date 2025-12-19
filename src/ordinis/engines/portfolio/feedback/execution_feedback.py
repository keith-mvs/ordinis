"""
Execution Feedback Collector - Tracks execution quality for learning.

Provides closed-loop feedback between trade execution and sizing decisions.
Integrates with LearningEngine to continuously improve cost estimates and
position sizing accuracy.

Gap Addressed: No execution quality feedback loop existed between
execution outcomes and sizing decisions.

Reference: Alpaca API provides filled_avg_price, filled_qty for actual
execution quality measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable
import logging
import uuid

import numpy as np

if TYPE_CHECKING:
    from ordinis.engines.learning.collectors.feedback import FeedbackCollector
    from ordinis.engines.portfolio.costs.transaction_cost_model import (
        AdaptiveCostModel,
        TransactionCostEstimate,
    )

logger = logging.getLogger(__name__)


class ExecutionQualityLevel(Enum):
    """Quality levels for execution assessment."""

    EXCELLENT = auto()  # < 5 bps slippage
    GOOD = auto()  # 5-15 bps slippage
    ACCEPTABLE = auto()  # 15-30 bps slippage
    POOR = auto()  # 30-50 bps slippage
    VERY_POOR = auto()  # > 50 bps slippage


@dataclass
class ExecutionRecord:
    """Record of a single trade execution.

    Captures expected vs actual execution for quality analysis.
    Aligned with Alpaca order response fields.

    Attributes:
        order_id: Unique order identifier
        symbol: Traded symbol
        side: 'buy' or 'sell'
        expected_price: Price at order submission
        filled_avg_price: Actual average fill price
        expected_qty: Requested quantity
        filled_qty: Actual filled quantity
        expected_cost_bps: Estimated transaction cost
        actual_cost_bps: Realized transaction cost
        slippage_bps: Price slippage in basis points
        execution_time_ms: Time to fill in milliseconds
        timestamp: Fill timestamp
    """

    order_id: str
    symbol: str
    side: str
    expected_price: float
    filled_avg_price: float
    expected_qty: float
    filled_qty: float
    expected_cost_bps: float
    actual_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.expected_price > 0:
            price_diff = abs(self.filled_avg_price - self.expected_price)
            self.slippage_bps = (price_diff / self.expected_price) * 10000.0

        if self.expected_cost_bps > 0:
            self.actual_cost_bps = self.slippage_bps + self.expected_cost_bps

    @property
    def quality_level(self) -> ExecutionQualityLevel:
        """Assess execution quality based on slippage."""
        if self.slippage_bps < 5:
            return ExecutionQualityLevel.EXCELLENT
        elif self.slippage_bps < 15:
            return ExecutionQualityLevel.GOOD
        elif self.slippage_bps < 30:
            return ExecutionQualityLevel.ACCEPTABLE
        elif self.slippage_bps < 50:
            return ExecutionQualityLevel.POOR
        else:
            return ExecutionQualityLevel.VERY_POOR

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate (actual / expected quantity)."""
        if self.expected_qty <= 0:
            return 0.0
        return self.filled_qty / self.expected_qty

    @property
    def is_partial_fill(self) -> bool:
        """Check if order was only partially filled."""
        return 0 < self.fill_rate < 1.0

    @property
    def notional_value(self) -> Decimal:
        """Calculate filled notional value."""
        return Decimal(str(self.filled_qty * self.filled_avg_price))


@dataclass
class ExecutionQualityMetrics:
    """Aggregated execution quality metrics over a time period.

    Attributes:
        period_start: Start of measurement period
        period_end: End of measurement period
        n_executions: Number of executions
        avg_slippage_bps: Average slippage in basis points
        median_slippage_bps: Median slippage
        p95_slippage_bps: 95th percentile slippage
        avg_fill_rate: Average fill rate
        total_notional: Total notional value traded
        cost_estimate_rmse: RMSE of cost estimates
        cost_estimate_bias: Bias in cost estimates
    """

    period_start: datetime
    period_end: datetime
    n_executions: int
    avg_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    avg_fill_rate: float
    total_notional: Decimal
    cost_estimate_rmse: float
    cost_estimate_bias: float
    symbol_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def overall_quality(self) -> ExecutionQualityLevel:
        """Assess overall execution quality."""
        if self.avg_slippage_bps < 5:
            return ExecutionQualityLevel.EXCELLENT
        elif self.avg_slippage_bps < 15:
            return ExecutionQualityLevel.GOOD
        elif self.avg_slippage_bps < 30:
            return ExecutionQualityLevel.ACCEPTABLE
        elif self.avg_slippage_bps < 50:
            return ExecutionQualityLevel.POOR
        else:
            return ExecutionQualityLevel.VERY_POOR


class ExecutionFeedbackCollector:
    """
    Collects and analyzes trade execution feedback.

    Provides closed-loop feedback for:
    1. Transaction cost model calibration
    2. Position sizing adjustments
    3. Execution venue analysis
    4. LearningEngine training data

    Example:
        >>> collector = ExecutionFeedbackCollector()

        >>> # Record expected execution
        >>> collector.record_order_submission(
        ...     order_id="order_123",
        ...     symbol="AAPL",
        ...     expected_price=150.0,
        ...     expected_qty=100,
        ...     estimated_cost_bps=10.0,
        ... )

        >>> # Record actual execution (from Alpaca fill)
        >>> collector.record_fill(
        ...     order_id="order_123",
        ...     filled_avg_price=150.05,
        ...     filled_qty=100,
        ...     execution_time_ms=50,
        ... )

        >>> # Get quality metrics
        >>> metrics = collector.get_quality_metrics(lookback_hours=24)
    """

    def __init__(
        self,
        max_history_size: int = 10000,
        adaptive_cost_model: "AdaptiveCostModel | None" = None,
        learning_engine_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize execution feedback collector.

        Args:
            max_history_size: Maximum execution records to retain
            adaptive_cost_model: Cost model to update with feedback
            learning_engine_callback: Callback to send data to LearningEngine
        """
        self.max_history_size = max_history_size
        self._adaptive_cost_model = adaptive_cost_model
        self._learning_callback = learning_engine_callback

        # Pending orders (submission recorded, not yet filled)
        self._pending_orders: dict[str, dict[str, Any]] = {}

        # Completed execution records
        self._execution_history: list[ExecutionRecord] = []

        # Per-symbol statistics
        self._symbol_stats: dict[str, dict[str, Any]] = {}

    def record_order_submission(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        expected_qty: float,
        estimated_cost_bps: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record order submission for later fill matching.

        Call this when submitting an order, before knowing the fill price.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            expected_price: Price at submission time
            expected_qty: Requested quantity
            estimated_cost_bps: Estimated transaction cost
            metadata: Additional order metadata
        """
        self._pending_orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "expected_price": expected_price,
            "expected_qty": expected_qty,
            "estimated_cost_bps": estimated_cost_bps,
            "submitted_at": datetime.now(UTC),
            "metadata": metadata or {},
        }

        logger.debug(f"Recorded order submission: {order_id} for {symbol}")

    def record_fill(
        self,
        order_id: str,
        filled_avg_price: float,
        filled_qty: float,
        execution_time_ms: float | None = None,
        filled_at: datetime | None = None,
    ) -> ExecutionRecord | None:
        """Record order fill and calculate execution quality.

        Call this when receiving fill confirmation from broker.
        Matches with pending order submission to calculate slippage.

        Args:
            order_id: Order identifier (must match previous submission)
            filled_avg_price: Average fill price
            filled_qty: Quantity filled
            execution_time_ms: Time to fill (optional)
            filled_at: Fill timestamp (optional)

        Returns:
            ExecutionRecord if matched, None if order not found
        """
        pending = self._pending_orders.pop(order_id, None)
        if not pending:
            logger.warning(f"No pending order found for fill: {order_id}")
            return None

        # Calculate execution time if not provided
        if execution_time_ms is None and "submitted_at" in pending:
            submitted = pending["submitted_at"]
            filled = filled_at or datetime.now(UTC)
            execution_time_ms = (filled - submitted).total_seconds() * 1000

        # Create execution record
        record = ExecutionRecord(
            order_id=order_id,
            symbol=pending["symbol"],
            side=pending["side"],
            expected_price=pending["expected_price"],
            filled_avg_price=filled_avg_price,
            expected_qty=pending["expected_qty"],
            filled_qty=filled_qty,
            expected_cost_bps=pending["estimated_cost_bps"],
            execution_time_ms=execution_time_ms or 0.0,
            timestamp=filled_at or datetime.now(UTC),
            metadata=pending.get("metadata", {}),
        )

        self._add_execution_record(record)

        return record

    def record_execution(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        filled_avg_price: float,
        expected_qty: float,
        filled_qty: float,
        estimated_cost_bps: float = 0.0,
        execution_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        """Record a complete execution directly (without pending order).

        Use when you have both submission and fill data at once.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            expected_price: Expected/submission price
            filled_avg_price: Actual fill price
            expected_qty: Requested quantity
            filled_qty: Filled quantity
            estimated_cost_bps: Estimated transaction cost
            execution_time_ms: Time to fill
            metadata: Additional metadata

        Returns:
            ExecutionRecord
        """
        record = ExecutionRecord(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            filled_avg_price=filled_avg_price,
            expected_qty=expected_qty,
            filled_qty=filled_qty,
            expected_cost_bps=estimated_cost_bps,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )

        self._add_execution_record(record)

        return record

    def _add_execution_record(self, record: ExecutionRecord) -> None:
        """Add execution record and update models.

        Args:
            record: Execution record to add
        """
        # Add to history
        self._execution_history.append(record)

        # Trim history if needed
        if len(self._execution_history) > self.max_history_size:
            self._execution_history = self._execution_history[-self.max_history_size :]

        # Update symbol statistics
        self._update_symbol_stats(record)

        # Update adaptive cost model if available
        if self._adaptive_cost_model:
            self._adaptive_cost_model.record_execution(
                symbol=record.symbol,
                estimated_cost_bps=record.expected_cost_bps,
                actual_cost_bps=record.actual_cost_bps,
                notional=float(record.notional_value),
            )

        # Send to learning engine if callback provided
        if self._learning_callback:
            self._learning_callback(
                {
                    "type": "execution_feedback",
                    "symbol": record.symbol,
                    "slippage_bps": record.slippage_bps,
                    "quality": record.quality_level.name,
                    "fill_rate": record.fill_rate,
                    "notional": float(record.notional_value),
                    "timestamp": record.timestamp.isoformat(),
                }
            )

        logger.debug(
            f"Recorded execution: {record.symbol} slippage={record.slippage_bps:.2f}bps "
            f"quality={record.quality_level.name}"
        )

    def _update_symbol_stats(self, record: ExecutionRecord) -> None:
        """Update per-symbol statistics with new execution.

        Args:
            record: New execution record
        """
        symbol = record.symbol
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = {
                "n_executions": 0,
                "total_slippage_bps": 0.0,
                "total_notional": Decimal("0"),
                "slippages": [],
            }

        stats = self._symbol_stats[symbol]
        stats["n_executions"] += 1
        stats["total_slippage_bps"] += record.slippage_bps
        stats["total_notional"] += record.notional_value
        stats["slippages"].append(record.slippage_bps)

        # Keep only recent slippages for percentile calculation
        if len(stats["slippages"]) > 1000:
            stats["slippages"] = stats["slippages"][-1000:]

    def get_quality_metrics(
        self,
        lookback_hours: float = 24.0,
        symbol: str | None = None,
    ) -> ExecutionQualityMetrics:
        """Calculate execution quality metrics over a time period.

        Args:
            lookback_hours: Hours to look back
            symbol: Filter to specific symbol (optional)

        Returns:
            ExecutionQualityMetrics
        """
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

        # Filter executions
        filtered = [
            r
            for r in self._execution_history
            if r.timestamp >= cutoff and (symbol is None or r.symbol == symbol)
        ]

        if not filtered:
            return ExecutionQualityMetrics(
                period_start=cutoff,
                period_end=datetime.now(UTC),
                n_executions=0,
                avg_slippage_bps=0.0,
                median_slippage_bps=0.0,
                p95_slippage_bps=0.0,
                avg_fill_rate=0.0,
                total_notional=Decimal("0"),
                cost_estimate_rmse=0.0,
                cost_estimate_bias=0.0,
            )

        slippages = [r.slippage_bps for r in filtered]
        fill_rates = [r.fill_rate for r in filtered]
        notionals = [r.notional_value for r in filtered]

        # Cost estimate errors
        estimate_errors = [
            r.actual_cost_bps - r.expected_cost_bps
            for r in filtered
            if r.expected_cost_bps > 0
        ]

        # Per-symbol breakdown
        symbol_breakdown: dict[str, dict[str, float]] = {}
        for r in filtered:
            if r.symbol not in symbol_breakdown:
                symbol_breakdown[r.symbol] = {"n": 0, "avg_slippage": 0.0, "notional": 0.0}
            sb = symbol_breakdown[r.symbol]
            sb["n"] += 1
            sb["avg_slippage"] = (
                sb["avg_slippage"] * (sb["n"] - 1) + r.slippage_bps
            ) / sb["n"]
            sb["notional"] += float(r.notional_value)

        return ExecutionQualityMetrics(
            period_start=cutoff,
            period_end=datetime.now(UTC),
            n_executions=len(filtered),
            avg_slippage_bps=float(np.mean(slippages)),
            median_slippage_bps=float(np.median(slippages)),
            p95_slippage_bps=float(np.percentile(slippages, 95)),
            avg_fill_rate=float(np.mean(fill_rates)),
            total_notional=sum(notionals, Decimal("0")),
            cost_estimate_rmse=(
                float(np.sqrt(np.mean(np.square(estimate_errors))))
                if estimate_errors
                else 0.0
            ),
            cost_estimate_bias=float(np.mean(estimate_errors)) if estimate_errors else 0.0,
            symbol_breakdown=symbol_breakdown,
        )

    def get_symbol_stats(self, symbol: str) -> dict[str, Any]:
        """Get statistics for a specific symbol.

        Args:
            symbol: Symbol to get stats for

        Returns:
            Dictionary with symbol statistics
        """
        stats = self._symbol_stats.get(symbol, {})
        if not stats:
            return {"symbol": symbol, "n_executions": 0}

        slippages = stats.get("slippages", [])
        return {
            "symbol": symbol,
            "n_executions": stats.get("n_executions", 0),
            "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
            "median_slippage_bps": float(np.median(slippages)) if slippages else 0.0,
            "total_notional": float(stats.get("total_notional", 0)),
        }

    def get_worst_executions(
        self,
        n: int = 10,
        lookback_hours: float = 24.0,
    ) -> list[ExecutionRecord]:
        """Get worst executions by slippage.

        Args:
            n: Number of worst executions to return
            lookback_hours: Hours to look back

        Returns:
            List of ExecutionRecord sorted by slippage (worst first)
        """
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        filtered = [r for r in self._execution_history if r.timestamp >= cutoff]
        sorted_records = sorted(filtered, key=lambda r: r.slippage_bps, reverse=True)
        return sorted_records[:n]

    def should_adjust_sizing(
        self,
        symbol: str,
        threshold_bps: float = 30.0,
        min_samples: int = 10,
    ) -> tuple[bool, float]:
        """Determine if position sizing should be adjusted based on execution quality.

        Args:
            symbol: Symbol to check
            threshold_bps: Slippage threshold for adjustment
            min_samples: Minimum samples required

        Returns:
            Tuple of (should_adjust, recommended_multiplier)
        """
        stats = self._symbol_stats.get(symbol, {})
        slippages = stats.get("slippages", [])

        if len(slippages) < min_samples:
            return False, 1.0

        avg_slippage = float(np.mean(slippages))

        if avg_slippage > threshold_bps:
            # High slippage: recommend reducing size
            # Scale down proportionally to slippage
            multiplier = max(0.5, threshold_bps / avg_slippage)
            return True, multiplier
        elif avg_slippage < threshold_bps * 0.5:
            # Very low slippage: could potentially increase size
            multiplier = min(1.5, threshold_bps / (avg_slippage + 1))
            return True, multiplier

        return False, 1.0

    def export_for_learning(
        self,
        lookback_hours: float = 168.0,  # 1 week
    ) -> list[dict[str, Any]]:
        """Export execution data for LearningEngine training.

        Args:
            lookback_hours: Hours of data to export

        Returns:
            List of execution records as dictionaries
        """
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        filtered = [r for r in self._execution_history if r.timestamp >= cutoff]

        return [
            {
                "order_id": r.order_id,
                "symbol": r.symbol,
                "side": r.side,
                "expected_price": r.expected_price,
                "filled_avg_price": r.filled_avg_price,
                "expected_qty": r.expected_qty,
                "filled_qty": r.filled_qty,
                "expected_cost_bps": r.expected_cost_bps,
                "actual_cost_bps": r.actual_cost_bps,
                "slippage_bps": r.slippage_bps,
                "execution_time_ms": r.execution_time_ms,
                "quality": r.quality_level.name,
                "fill_rate": r.fill_rate,
                "notional": float(r.notional_value),
                "timestamp": r.timestamp.isoformat(),
            }
            for r in filtered
        ]
