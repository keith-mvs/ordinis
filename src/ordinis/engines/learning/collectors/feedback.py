"""
Feedback Collector - Continual Learning Data Pipeline.

Collects feedback from development (backtests, parameter sweeps, sensitivity analysis)
and trading (live execution, signal accuracy, regime changes) to feed back into:
1. LearningEngine for model training
2. ChromaDB for semantic search/RAG
3. SQLite for structured queries

This creates a closed-loop system where insights from analysis and trading
continuously improve the models and strategy selection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
from typing import TYPE_CHECKING, Any, Callable
import uuid

from loguru import logger

if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager
    from ordinis.engines.learning.core.engine import LearningEngine
    from ordinis.rag.vectordb.chroma_client import ChromaClient


# =============================================================================
# Circuit Breaker Monitor
# =============================================================================
# Per SDR GovernanceEngine: "Policy Checks (Pre-flight): Before any critical
# action is taken by another engine, the GovernanceEngine can perform checks."
#
# This would have prevented 12/17 disaster by:
# 1. Detecting error rate spike after first ~100 failures
# 2. Triggering signal throttle when buying power exhausted
# 3. Forcing position reconciliation when mismatches detected
# =============================================================================


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    HALF_OPEN = "half_open"  # Testing recovery
    OPEN = "open"  # Trading halted


@dataclass
class ErrorWindow:
    """Sliding window for error rate monitoring."""

    window_seconds: int = 60
    max_errors: int = 10  # Threshold before circuit breaker trips

    _timestamps: deque = field(default_factory=deque)

    def record_error(self) -> int:
        """Record an error and return current count in window."""
        now = datetime.now(UTC)
        self._timestamps.append(now)
        self._prune_old()
        return len(self._timestamps)

    def _prune_old(self) -> None:
        """Remove timestamps outside the window."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.window_seconds)
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def get_rate_per_minute(self) -> float:
        """Get errors per minute."""
        self._prune_old()
        count = len(self._timestamps)
        return count * (60.0 / self.window_seconds)

    def is_threshold_exceeded(self) -> bool:
        """Check if error threshold is exceeded."""
        self._prune_old()
        return len(self._timestamps) >= self.max_errors


@dataclass
class CircuitBreakerMonitor:
    """
    Monitors error patterns and triggers circuit breakers.

    Per SDR GovernanceEngine Section 9:
    "The GovernanceEngine serves as a cross-cutting layer that enforces policy,
    risk limits, and compliance... Policy Checks (Pre-flight): Before any
    critical action is taken by another engine, the GovernanceEngine can
    perform checks to ensure:
    - Trading is allowed under current market conditions
    - Risk limits haven't been exceeded
    - Required approvals are in place"

    On 12/17, none of these checks were in place:
    - 6,123 "insufficient buying power" errors occurred without adaptation
    - Error rate was ~100+/minute with no throttling
    - Position tracking diverged from broker with no reconciliation

    This class implements the missing feedback loop.
    """

    # Error tracking per error type
    error_windows: dict[str, ErrorWindow] = field(default_factory=dict)

    # Circuit breaker state per engine
    engine_states: dict[str, CircuitBreakerState] = field(default_factory=dict)

    # Timestamps when breakers were opened
    breaker_opened_at: dict[str, datetime] = field(default_factory=dict)

    # Recovery timeout (how long to wait before testing half-open)
    recovery_timeout_seconds: int = 300  # 5 minutes

    # Callbacks for when state changes
    _on_state_change: Callable[[str, CircuitBreakerState], None] | None = None
    _on_throttle: Callable[[str, int], None] | None = None

    def __post_init__(self):
        """Initialize default error windows."""
        self.error_windows = {
            "execution_failure": ErrorWindow(window_seconds=60, max_errors=10),
            "insufficient_capital": ErrorWindow(window_seconds=60, max_errors=3),  # Very sensitive
            "order_rejected": ErrorWindow(window_seconds=60, max_errors=5),
            "position_mismatch": ErrorWindow(
                window_seconds=300, max_errors=1
            ),  # Any mismatch is critical
            "broker_sync_failure": ErrorWindow(window_seconds=60, max_errors=2),
        }
        # All engines start in CLOSED state (normal operation)
        self.engine_states = {
            "signal_engine": CircuitBreakerState.CLOSED,
            "risk_engine": CircuitBreakerState.CLOSED,
            "execution_engine": CircuitBreakerState.CLOSED,
            "portfolio_engine": CircuitBreakerState.CLOSED,
        }

    def record_error(self, error_type: str, engine: str = "execution_engine") -> tuple[bool, str]:
        """
        Record an error and check if circuit breaker should trip.

        Args:
            error_type: Type of error (execution_failure, insufficient_capital, etc.)
            engine: Engine that experienced the error

        Returns:
            (should_trip, reason) - True if circuit breaker should open
        """
        if error_type not in self.error_windows:
            # Create new window for unknown error types
            self.error_windows[error_type] = ErrorWindow(window_seconds=60, max_errors=10)

        window = self.error_windows[error_type]
        count = window.record_error()
        rate = window.get_rate_per_minute()

        # Check if threshold exceeded
        if window.is_threshold_exceeded():
            reason = (
                f"Error rate exceeded for {error_type}: "
                f"{count} errors in {window.window_seconds}s "
                f"({rate:.1f}/min, threshold: {window.max_errors})"
            )
            self._trip_breaker(engine, reason)
            return True, reason

        return False, ""

    def _trip_breaker(self, engine: str, reason: str) -> None:
        """Open the circuit breaker for an engine."""
        if engine not in self.engine_states:
            self.engine_states[engine] = CircuitBreakerState.CLOSED

        old_state = self.engine_states[engine]
        if old_state != CircuitBreakerState.OPEN:
            self.engine_states[engine] = CircuitBreakerState.OPEN
            self.breaker_opened_at[engine] = datetime.now(UTC)
            logger.warning(f"Circuit breaker OPENED for {engine}: {reason}")

            if self._on_state_change:
                self._on_state_change(engine, CircuitBreakerState.OPEN)

    def is_open(self, engine: str) -> bool:
        """Check if circuit breaker is open for an engine."""
        return (
            self.engine_states.get(engine, CircuitBreakerState.CLOSED) == CircuitBreakerState.OPEN
        )

    def should_allow_signal(self, engine: str = "signal_engine") -> tuple[bool, str]:
        """
        Check if signals should be allowed.

        Returns:
            (allow, reason) - False if trading should be blocked
        """
        state = self.engine_states.get(engine, CircuitBreakerState.CLOSED)

        if state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            opened_at = self.breaker_opened_at.get(engine)
            if opened_at:
                elapsed = (datetime.now(UTC) - opened_at).total_seconds()
                if elapsed >= self.recovery_timeout_seconds:
                    # Move to half-open for testing
                    self.engine_states[engine] = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker HALF-OPEN for {engine} after {elapsed:.0f}s")
                    return True, "half_open_test"

            return False, "circuit_breaker_open"

        if state == CircuitBreakerState.HALF_OPEN:
            return True, "half_open_test"

        return True, ""

    def record_success(self, engine: str) -> None:
        """Record a successful operation (for half-open recovery)."""
        state = self.engine_states.get(engine, CircuitBreakerState.CLOSED)
        if state == CircuitBreakerState.HALF_OPEN:
            self.engine_states[engine] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker CLOSED for {engine} after successful operation")
            if self._on_state_change:
                self._on_state_change(engine, CircuitBreakerState.CLOSED)

    def get_status(self) -> dict[str, Any]:
        """Get current status of all circuit breakers."""
        status = {}
        for engine, state in self.engine_states.items():
            window_stats = {}
            for error_type, window in self.error_windows.items():
                window._prune_old()
                window_stats[error_type] = {
                    "count": len(window._timestamps),
                    "rate_per_minute": window.get_rate_per_minute(),
                    "threshold": window.max_errors,
                    "is_exceeded": window.is_threshold_exceeded(),
                }
            status[engine] = {
                "state": state.value,
                "opened_at": self.breaker_opened_at.get(engine, None),
                "error_windows": window_stats,
            }
        return status


# =============================================================================
# Feedback Types and Records
# =============================================================================


class FeedbackType(Enum):
    """Types of feedback collected for learning."""

    # Development feedback
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    PARAMETER_SWEEP = "parameter_sweep"
    WALK_FORWARD_VALIDATION = "walk_forward_validation"
    BACKTEST_RESULT = "backtest_result"
    STRATEGY_EVALUATION = "strategy_evaluation"
    REGIME_ANALYSIS = "regime_analysis"

    # Trading feedback
    TRADE_OUTCOME = "trade_outcome"
    SIGNAL_ACCURACY = "signal_accuracy"
    EXECUTION_QUALITY = "execution_quality"
    RISK_EVENT = "risk_event"
    REGIME_CHANGE = "regime_change"
    DRAWDOWN_EVENT = "drawdown_event"

    # Model feedback
    PREDICTION_ACCURACY = "prediction_accuracy"
    MODEL_DRIFT = "model_drift"
    FEATURE_IMPORTANCE = "feature_importance"

    # Execution feedback (NEW - would have caught 12/17 issues)
    EXECUTION_FAILURE = "execution_failure"
    ORDER_REJECTED = "order_rejected"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    POSITION_LIMIT_HIT = "position_limit_hit"

    # Account state feedback (NEW)
    ACCOUNT_STATE_CHANGE = "account_state_change"
    BUYING_POWER_EXHAUSTED = "buying_power_exhausted"
    MARGIN_WARNING = "margin_warning"

    # Circuit breaker feedback (NEW)
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    ERROR_RATE_SPIKE = "error_rate_spike"
    SIGNAL_THROTTLE = "signal_throttle"

    # Position reconciliation (NEW)
    POSITION_MISMATCH = "position_mismatch"
    BROKER_SYNC_FAILURE = "broker_sync_failure"


class FeedbackPriority(Enum):
    """Priority level for feedback processing."""

    LOW = "low"  # Background processing
    NORMAL = "normal"  # Standard batch processing
    HIGH = "high"  # Process within hour
    CRITICAL = "critical"  # Immediate processing


@dataclass
class FeedbackRecord:
    """A single feedback record for the learning pipeline."""

    feedback_id: str = field(default_factory=lambda: f"FB-{uuid.uuid4().hex[:12].upper()}")
    feedback_type: FeedbackType = FeedbackType.BACKTEST_RESULT
    priority: FeedbackPriority = FeedbackPriority.NORMAL
    source: str = ""  # sensitivity_analysis, live_trading, backtest, etc.
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Core payload
    strategy: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    # Outcomes
    train_performance: dict[str, float] = field(default_factory=dict)
    test_performance: dict[str, float] = field(default_factory=dict)
    is_overfit: bool = False
    overfitting_score: float = 0.0

    # Context
    market_regime: str | None = None
    volatility_regime: str | None = None
    data_quality_score: float = 1.0

    # Labels for retrieval
    labels: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Text summary for ChromaDB embedding
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "priority": self.priority.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "train_performance": self.train_performance,
            "test_performance": self.test_performance,
            "is_overfit": self.is_overfit,
            "overfitting_score": self.overfitting_score,
            "market_regime": self.market_regime,
            "volatility_regime": self.volatility_regime,
            "data_quality_score": self.data_quality_score,
            "labels": self.labels,
            "tags": self.tags,
            "summary": self.summary,
        }

    def to_embedding_text(self) -> str:
        """Generate text for embedding in ChromaDB."""
        parts = [
            f"Strategy: {self.strategy}" if self.strategy else "",
            f"Type: {self.feedback_type.value}",
            f"Source: {self.source}",
        ]

        if self.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            parts.append(f"Parameters: {param_str}")

        if self.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
            parts.append(f"Metrics: {metric_str}")

        if self.train_performance:
            parts.append(f"Train: {self.train_performance}")

        if self.test_performance:
            parts.append(f"Test: {self.test_performance}")

        if self.is_overfit:
            parts.append(f"OVERFIT (score: {self.overfitting_score:.2f})")

        if self.summary:
            parts.append(f"Summary: {self.summary}")

        return "\n".join(p for p in parts if p)


# SQL schema for feedback storage
FEEDBACK_SCHEMA = """
-- Learning feedback table
CREATE TABLE IF NOT EXISTS learning_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id TEXT NOT NULL UNIQUE,
    feedback_type TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    source TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    strategy TEXT,
    symbol TEXT,
    timeframe TEXT,
    parameters TEXT,  -- JSON
    metrics TEXT,  -- JSON
    train_performance TEXT,  -- JSON
    test_performance TEXT,  -- JSON
    is_overfit INTEGER NOT NULL DEFAULT 0,
    overfitting_score REAL NOT NULL DEFAULT 0.0,
    market_regime TEXT,
    volatility_regime TEXT,
    data_quality_score REAL NOT NULL DEFAULT 1.0,
    labels TEXT,  -- JSON
    tags TEXT,  -- JSON array
    summary TEXT,
    processed INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_id ON learning_feedback(feedback_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON learning_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_strategy ON learning_feedback(strategy);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON learning_feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_processed ON learning_feedback(processed);

-- Optimal parameters table (derived from feedback)
CREATE TABLE IF NOT EXISTS optimal_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT,  -- NULL for universal params
    timeframe TEXT,
    parameters TEXT NOT NULL,  -- JSON
    train_sharpe REAL,
    test_sharpe REAL,
    overfitting_score REAL,
    confidence_score REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    source_feedback_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(strategy, symbol, timeframe, is_active)
);

CREATE INDEX IF NOT EXISTS idx_optimal_strategy ON optimal_parameters(strategy);
CREATE INDEX IF NOT EXISTS idx_optimal_active ON optimal_parameters(is_active);

-- Regime history table
CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_type TEXT NOT NULL,  -- market, volatility, momentum
    regime_value TEXT NOT NULL,  -- trending, ranging, high_vol, low_vol
    symbol TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    indicators TEXT,  -- JSON with ADX, ATR, etc.
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_regime_type ON regime_history(regime_type);
CREATE INDEX IF NOT EXISTS idx_regime_start ON regime_history(start_time);
"""


class FeedbackCollector:
    """
    Collects and routes feedback to storage systems.

    Provides hooks for:
    - Development: sensitivity analysis, backtests, parameter sweeps
    - Trading: trade outcomes, signal accuracy, regime changes
    - Models: predictions, drift, feature importance

    Routes to:
    - SQLite: Structured storage for queries and analysis
    - ChromaDB: Semantic search and RAG retrieval
    - LearningEngine: Model training pipeline

    Includes:
    - CircuitBreakerMonitor: Real-time error rate monitoring and trading halt
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        chroma_client: ChromaClient | None = None,
        learning_engine: LearningEngine | None = None,
        feedback_collection_name: str = "learning_feedback",
        circuit_breaker: CircuitBreakerMonitor | None = None,
    ):
        """Initialize feedback collector.

        Args:
            db_manager: SQLite database manager
            chroma_client: ChromaDB client for vector storage
            learning_engine: Learning engine for model training
            feedback_collection_name: ChromaDB collection for feedback
            circuit_breaker: Real-time error monitoring (creates default if None)
        """
        self.db_manager = db_manager
        self.chroma_client = chroma_client
        self.learning_engine = learning_engine
        self.feedback_collection_name = feedback_collection_name

        # Circuit breaker for real-time error monitoring
        # This would have caught 12/17 after first ~10-100 failures
        self.circuit_breaker = circuit_breaker or CircuitBreakerMonitor()

        self._initialized = False
        self._pending_records: list[FeedbackRecord] = []
        self._feedback_collection = None

    async def initialize(self) -> None:
        """Initialize storage and create schema if needed."""
        if self._initialized:
            return

        # Initialize SQLite schema
        if self.db_manager and self.db_manager.is_connected:
            await self._init_sqlite_schema()

        # Initialize ChromaDB collection
        if self.chroma_client:
            self._feedback_collection = self.chroma_client.client.get_or_create_collection(
                name=self.feedback_collection_name,
                metadata={"description": "Learning feedback for RAG retrieval"},
            )
            logger.info(f"Feedback collection '{self.feedback_collection_name}' ready")

        self._initialized = True
        logger.info("FeedbackCollector initialized")

    async def _init_sqlite_schema(self) -> None:
        """Create SQLite tables for feedback storage."""
        if not self.db_manager or not self.db_manager._connection:
            return

        async with self.db_manager._lock:
            await self.db_manager._connection.executescript(FEEDBACK_SCHEMA)
            await self.db_manager._connection.commit()
            logger.info("Feedback schema created/verified")

    # -------------------------------------------------------------------------
    # Circuit Breaker Checks (per SDR GovernanceEngine)
    # -------------------------------------------------------------------------
    # These methods should be called BEFORE any engine takes action
    # This is the "pre-flight" check pattern from the spec

    def should_allow_signals(self) -> tuple[bool, str]:
        """
        Check if signal generation should proceed.

        Per SDR GovernanceEngine: "Policy Checks (Pre-flight): Before any
        critical action is taken by another engine..."

        Call this before generating/processing signals.

        Returns:
            (allowed, reason): If not allowed, reason explains why
        """
        # Check ALL engines - any tripped breaker blocks signals
        # This is the holistic protection that was missing on 12/17
        for engine in self.circuit_breaker.engine_states:
            allowed, reason = self.circuit_breaker.should_allow_signal(engine)
            if not allowed:
                return False, f"{engine}: {reason}"
        return True, ""

    def should_allow_execution(self) -> tuple[bool, str]:
        """
        Check if order execution should proceed.

        Per SDR GovernanceEngine: "Trading is allowed under current market
        conditions... Risk limits haven't been exceeded..."

        Call this before submitting orders.

        Returns:
            (allowed, reason): If not allowed, reason explains why
        """
        # Check ALL engines - risk/portfolio issues should block execution
        # This prevents orders when capital is exhausted or positions mismatched
        for engine in self.circuit_breaker.engine_states:
            allowed, reason = self.circuit_breaker.should_allow_signal(engine)
            if not allowed:
                return False, f"{engine}: {reason}"
        return True, ""

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """
        Get current status of all circuit breakers.

        Per SDR GovernanceEngine: "Audit Trail: Maintain a log of actions
        and decisions made by the GovernanceEngine."

        Returns:
            Status dict with state, error counts, thresholds
        """
        return self.circuit_breaker.get_status()

    def reset_circuit_breaker(self, engine: str) -> None:
        """
        Manually reset a circuit breaker.

        Per SDR GovernanceEngine: "Escalation Path: Provide alerts or halt
        trading if needed, with manual override capabilities."

        Args:
            engine: Engine to reset (signal_engine, execution_engine, etc.)
        """
        if engine in self.circuit_breaker.engine_states:
            self.circuit_breaker.engine_states[engine] = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker manually reset for {engine}")

    # -------------------------------------------------------------------------
    # Recording Hooks
    # -------------------------------------------------------------------------

    async def record_sensitivity_analysis(
        self,
        strategy: str,
        parameters: dict[str, Any],
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
        overfitting_score: float,
        symbols: list[str] | None = None,
        timeframe: str = "daily",
        summary: str = "",
    ) -> FeedbackRecord:
        """
        Record results from sensitivity analysis.

        Args:
            strategy: Strategy name
            parameters: Parameter configuration tested
            train_metrics: In-sample metrics (sharpe, pnl, etc.)
            test_metrics: Out-of-sample metrics
            overfitting_score: Degree of overfitting (0-1)
            symbols: Symbols analyzed
            timeframe: Data timeframe
            summary: Human-readable summary

        Returns:
            Created FeedbackRecord
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.SENSITIVITY_ANALYSIS,
            source="sensitivity_analysis",
            strategy=strategy,
            timeframe=timeframe,
            parameters=parameters,
            metrics={
                "train_sharpe": train_metrics.get("sharpe", 0),
                "test_sharpe": test_metrics.get("sharpe", 0),
                "train_pnl": train_metrics.get("total_pnl", 0),
                "test_pnl": test_metrics.get("total_pnl", 0),
            },
            train_performance=train_metrics,
            test_performance=test_metrics,
            is_overfit=overfitting_score > 0.3,
            overfitting_score=overfitting_score,
            labels={
                "type": "sensitivity",
                "strategy": strategy,
            },
            tags=["sensitivity", "walk_forward", strategy],
            summary=summary
            or self._generate_sensitivity_summary(
                strategy, parameters, train_metrics, test_metrics, overfitting_score
            ),
        )

        await self._store_record(record)
        return record

    async def record_parameter_sweep(
        self,
        strategy: str,
        parameter_name: str,
        sweep_results: list[dict[str, Any]],
        optimal_value: Any,
        stability_score: float,
    ) -> FeedbackRecord:
        """
        Record results from parameter sweep.

        Args:
            strategy: Strategy name
            parameter_name: Parameter being swept
            sweep_results: List of {value, train_metric, test_metric}
            optimal_value: Best parameter value
            stability_score: How stable across values (0-1)
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.PARAMETER_SWEEP,
            source="parameter_sweep",
            strategy=strategy,
            parameters={
                "swept_param": parameter_name,
                "optimal_value": optimal_value,
                "stability_score": stability_score,
                "n_values_tested": len(sweep_results),
            },
            metrics={
                "stability_score": stability_score,
            },
            labels={
                "type": "param_sweep",
                "strategy": strategy,
                "parameter": parameter_name,
            },
            tags=["parameter_sweep", strategy, parameter_name],
            summary=f"Parameter sweep for {strategy}.{parameter_name}: optimal={optimal_value}, stability={stability_score:.2f}",
        )

        await self._store_record(record)
        return record

    async def record_trade_outcome(
        self,
        strategy: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        signal_probability: float,
        parameters: dict[str, Any] | None = None,
        market_regime: str | None = None,
    ) -> FeedbackRecord:
        """
        Record outcome of a completed trade.

        Args:
            strategy: Strategy that generated the signal
            symbol: Traded symbol
            side: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            pnl: Absolute P&L
            pnl_pct: Percentage P&L
            signal_probability: Original signal probability
            parameters: Strategy parameters used
            market_regime: Market regime at trade time
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.TRADE_OUTCOME,
            priority=FeedbackPriority.HIGH,
            source="live_trading",
            strategy=strategy,
            symbol=symbol,
            parameters=parameters or {},
            metrics={
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "signal_probability": signal_probability,
                "entry_price": entry_price,
                "exit_price": exit_price,
            },
            market_regime=market_regime,
            labels={
                "type": "trade",
                "strategy": strategy,
                "symbol": symbol,
                "side": side,
                "outcome": "win" if pnl > 0 else "loss",
            },
            tags=["trade", strategy, symbol, "win" if pnl > 0 else "loss"],
            summary=f"{strategy} {side} {symbol}: {pnl_pct:+.2f}% (signal prob: {signal_probability:.2f})",
        )

        await self._store_record(record)
        return record

    async def record_signal_accuracy(
        self,
        strategy: str,
        n_signals: int,
        n_correct: int,
        accuracy: float,
        by_regime: dict[str, float] | None = None,
        time_period: str = "daily",
    ) -> FeedbackRecord:
        """
        Record signal accuracy metrics.

        Args:
            strategy: Strategy name
            n_signals: Total signals generated
            n_correct: Correct signals (profitable)
            accuracy: Hit rate (0-1)
            by_regime: Accuracy broken down by regime
            time_period: Period covered
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.SIGNAL_ACCURACY,
            priority=FeedbackPriority.NORMAL,
            source="signal_analytics",
            strategy=strategy,
            timeframe=time_period,
            metrics={
                "n_signals": n_signals,
                "n_correct": n_correct,
                "accuracy": accuracy,
                **(by_regime or {}),
            },
            labels={
                "type": "signal_accuracy",
                "strategy": strategy,
            },
            tags=["signal_accuracy", strategy],
            summary=f"{strategy} accuracy: {accuracy:.1%} ({n_correct}/{n_signals})",
        )

        await self._store_record(record)
        return record

    async def record_regime_change(
        self,
        regime_type: str,
        old_value: str,
        new_value: str,
        symbol: str | None = None,
        confidence: float = 0.5,
        indicators: dict[str, float] | None = None,
    ) -> FeedbackRecord:
        """
        Record market regime change.

        Args:
            regime_type: Type of regime (market, volatility, momentum)
            old_value: Previous regime value
            new_value: New regime value
            symbol: Symbol (None for market-wide)
            confidence: Confidence in regime detection
            indicators: Indicator values used for detection
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.REGIME_CHANGE,
            priority=FeedbackPriority.HIGH,
            source="regime_detector",
            symbol=symbol,
            market_regime=new_value if regime_type == "market" else None,
            volatility_regime=new_value if regime_type == "volatility" else None,
            parameters=indicators or {},
            metrics={
                "confidence": confidence,
            },
            labels={
                "type": "regime_change",
                "regime_type": regime_type,
                "old": old_value,
                "new": new_value,
            },
            tags=["regime", regime_type, new_value],
            summary=f"Regime change: {regime_type} {old_value} → {new_value} (conf: {confidence:.2f})",
        )

        await self._store_record(record)

        # Also store in regime_history table
        if self.db_manager and self.db_manager.is_connected:
            await self._store_regime_history(regime_type, new_value, symbol, confidence, indicators)

        return record

    # -------------------------------------------------------------------------
    # ExecutionEngine Feedback Hooks (per SDR Section 4)
    # -------------------------------------------------------------------------
    # "Feedback Loop: Publish execution results (fills, trade confirmations)
    #  back to the system... Execution events can also be published on the
    #  StreamingBus if other components need to know."

    async def record_execution_failure(
        self,
        order_id: str,
        symbol: str,
        error_type: str,
        error_message: str,
        order_details: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> tuple[FeedbackRecord, bool]:
        """
        Record execution failure from ExecutionEngine.

        Per SDR: "Order State Management: Track orders until completion –
        handle acknowledgments, partial fills, cancellations, and rejections."

        This would have caught the 6,123 "insufficient buying power" errors on 12/17.

        Args:
            order_id: Order identifier
            symbol: Symbol being traded
            error_type: Type of error (insufficient_buying_power, rejected, etc.)
            error_message: Full error message
            order_details: Order parameters (quantity, side, etc.)
            strategy: Strategy that generated the signal

        Returns:
            (FeedbackRecord, circuit_breaker_tripped): Record and whether breaker tripped
        """
        # Check circuit breaker BEFORE processing
        # This is the key feedback loop that was missing on 12/17
        tripped, trip_reason = self.circuit_breaker.record_error(
            error_type=error_type,
            engine="execution_engine",
        )

        if tripped:
            # Get error window stats
            error_window = self.circuit_breaker.error_windows.get(error_type, ErrorWindow())
            error_rate = error_window.get_rate_per_minute()
            error_count = len(error_window._timestamps)

            # Record the circuit breaker event
            await self.record_circuit_breaker_triggered(
                breaker_type="error_rate",
                trigger_reason=trip_reason,
                error_count=error_count,
                error_rate=error_rate,
                time_window_seconds=60,
                action_taken="halt_signals",
                affected_engines=["signal_engine", "execution_engine"],
            )
            logger.critical(f"Circuit breaker tripped: {trip_reason}")

        # Determine priority based on error type
        priority = (
            FeedbackPriority.CRITICAL
            if error_type in ["insufficient_buying_power", "margin_call", "account_restricted"]
            else FeedbackPriority.HIGH
        )

        record = FeedbackRecord(
            feedback_type=FeedbackType.EXECUTION_FAILURE,
            priority=priority,
            source="execution_engine",
            strategy=strategy,
            symbol=symbol,
            parameters=order_details or {},
            metrics={
                "error_count": 1,
                "circuit_breaker_tripped": 1 if tripped else 0,
            },
            labels={
                "type": "execution_failure",
                "error_type": error_type,
                "order_id": order_id,
                "circuit_breaker_state": self.circuit_breaker.engine_states.get(
                    "execution_engine", CircuitBreakerState.CLOSED
                ).value,
            },
            tags=["execution", "failure", error_type, symbol],
            summary=f"Execution failed for {symbol}: {error_type} - {error_message[:100]}",
        )

        await self._store_record(record)
        return record, tripped

    async def record_order_rejected(
        self,
        order_id: str,
        symbol: str,
        rejection_reason: str,
        broker_response: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """
        Record order rejection from broker.

        Per SDR: "Governance Hooks: Ensure compliance at execution time...
        Every executed trade generates an audit record."
        """
        # Notify circuit breaker about the rejection
        self.circuit_breaker.record_error("order_rejected", "execution_engine")

        record = FeedbackRecord(
            feedback_type=FeedbackType.ORDER_REJECTED,
            priority=FeedbackPriority.HIGH,
            source="execution_engine",
            symbol=symbol,
            parameters=broker_response or {},
            labels={
                "type": "order_rejected",
                "reason": rejection_reason,
                "order_id": order_id,
            },
            tags=["execution", "rejected", symbol],
            summary=f"Order {order_id} rejected for {symbol}: {rejection_reason}",
        )

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # RiskEngine Feedback Hooks (per SDR Section 3)
    # -------------------------------------------------------------------------
    # "Signal Adjustment or Blocking: Analyze each incoming signal... Output
    #  either an approval (possibly with adjusted size) or a rejection."

    async def record_risk_breach(
        self,
        breach_type: str,
        rule_id: str,
        current_value: float,
        threshold: float,
        action_taken: str,
        symbol: str | None = None,
        strategy: str | None = None,
        portfolio_state: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """
        Record risk policy breach from RiskEngine.

        Per SDR: "Policy Enforcement: Define and apply a set of risk rules...
        Exposure Limits, Leverage/Margin Checks, Stop-Loss/Drawdown Control..."

        This would have caught the margin exhaustion on 12/17.

        Args:
            breach_type: Type of breach (exposure_limit, margin, drawdown, etc.)
            rule_id: ID of the rule that was breached
            current_value: Current metric value
            threshold: Threshold that was exceeded
            action_taken: What the system did (blocked, resized, alert)
            symbol: Related symbol if applicable
            strategy: Strategy involved
            portfolio_state: Current portfolio snapshot
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.RISK_EVENT,
            priority=FeedbackPriority.CRITICAL,
            source="risk_engine",
            strategy=strategy,
            symbol=symbol,
            parameters=portfolio_state or {},
            metrics={
                "current_value": current_value,
                "threshold": threshold,
                "breach_severity": abs(current_value - threshold) / threshold if threshold else 0,
            },
            labels={
                "type": "risk_breach",
                "breach_type": breach_type,
                "rule_id": rule_id,
                "action": action_taken,
            },
            tags=["risk", "breach", breach_type, action_taken],
            summary=f"Risk breach: {breach_type} ({rule_id}) - {current_value:.2f} vs threshold {threshold:.2f}. Action: {action_taken}",
        )

        await self._store_record(record)
        return record

    async def record_insufficient_capital(
        self,
        required_capital: float,
        available_capital: float,
        buying_power: float,
        signal_count_blocked: int,
        portfolio_state: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """
        Record insufficient capital event.

        Per SDR RiskEngine: "Leverage/Margin Checks: For futures or leveraged
        instruments, ensure margin requirements are met."

        This is the PRIMARY feedback that was missing on 12/17 -
        6,123 signals failed because buying_power was 0.

        Args:
            required_capital: Capital needed for pending signals
            available_capital: Cash available
            buying_power: Broker-reported buying power
            signal_count_blocked: Number of signals that couldn't execute
            portfolio_state: Current portfolio snapshot
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.INSUFFICIENT_CAPITAL,
            priority=FeedbackPriority.CRITICAL,
            source="risk_engine",
            parameters=portfolio_state or {},
            metrics={
                "required_capital": required_capital,
                "available_capital": available_capital,
                "buying_power": buying_power,
                "signals_blocked": signal_count_blocked,
                "shortfall": required_capital - available_capital,
            },
            labels={
                "type": "capital_exhausted",
                "severity": "critical" if buying_power <= 0 else "warning",
            },
            tags=["risk", "capital", "buying_power", "margin"],
            summary=f"Insufficient capital: need ${required_capital:,.0f}, have ${available_capital:,.0f} (buying power: ${buying_power:,.0f}). {signal_count_blocked} signals blocked.",
        )

        # CRITICAL: Notify circuit breaker - this would have stopped 12/17 disaster
        self.circuit_breaker.record_error("insufficient_capital", "risk_engine")

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # PortfolioEngine Feedback Hooks (per SDR Section 5)
    # -------------------------------------------------------------------------
    # "Position Tracking: Maintain up-to-date positions for each asset...
    #  As execution reports come in, update positions accordingly."

    async def record_position_mismatch(
        self,
        symbol: str,
        internal_quantity: int,
        broker_quantity: int,
        internal_cost: float,
        broker_cost: float,
    ) -> FeedbackRecord:
        """
        Record position mismatch between internal tracking and broker.

        Per SDR PortfolioEngine: "Position Tracking: Maintain up-to-date
        positions for each asset (quantities held, average cost, unrealized P&L)."

        On 12/17, internal tracking showed 0 positions while broker had 21.

        Args:
            symbol: Symbol with mismatch
            internal_quantity: Quantity per internal tracking
            broker_quantity: Quantity per broker
            internal_cost: Average cost per internal tracking
            broker_cost: Average cost per broker
        """
        quantity_diff = broker_quantity - internal_quantity
        cost_diff = broker_cost - internal_cost

        # CRITICAL: Position mismatch trips circuit breaker immediately (threshold=1)
        # This prevents further trading when reconciliation has failed
        self.circuit_breaker.record_error("position_mismatch", "portfolio_engine")

        record = FeedbackRecord(
            feedback_type=FeedbackType.POSITION_MISMATCH,
            priority=FeedbackPriority.CRITICAL,
            source="portfolio_engine",
            symbol=symbol,
            metrics={
                "internal_quantity": internal_quantity,
                "broker_quantity": broker_quantity,
                "quantity_diff": quantity_diff,
                "internal_cost": internal_cost,
                "broker_cost": broker_cost,
                "cost_diff": cost_diff,
            },
            labels={
                "type": "position_mismatch",
                "symbol": symbol,
                "severity": "critical" if abs(quantity_diff) > 0 else "warning",
            },
            tags=["portfolio", "mismatch", "reconciliation", symbol],
            summary=f"Position mismatch for {symbol}: internal={internal_quantity} vs broker={broker_quantity} (diff: {quantity_diff})",
        )

        await self._store_record(record)
        return record

    async def record_portfolio_state_snapshot(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        position_count: int,
        total_exposure: float,
        margin_used: float,
        unrealized_pnl: float,
        daily_pnl: float,
    ) -> FeedbackRecord:
        """
        Record periodic portfolio state snapshot.

        Per SDR PortfolioEngine: "Performance and P&L: Calculate portfolio
        performance metrics in real-time or for each trading cycle."

        This creates a continuous record for the LearningEngine to analyze.
        """
        # Detect concerning states
        is_over_leveraged = total_exposure > equity * 2
        is_capital_exhausted = buying_power <= 0
        is_in_drawdown = daily_pnl < -equity * 0.02  # >2% daily loss

        priority = (
            FeedbackPriority.CRITICAL
            if (is_capital_exhausted or is_over_leveraged)
            else FeedbackPriority.NORMAL
        )

        record = FeedbackRecord(
            feedback_type=FeedbackType.ACCOUNT_STATE_CHANGE,
            priority=priority,
            source="portfolio_engine",
            metrics={
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
                "position_count": position_count,
                "total_exposure": total_exposure,
                "margin_used": margin_used,
                "unrealized_pnl": unrealized_pnl,
                "daily_pnl": daily_pnl,
                "leverage_ratio": total_exposure / equity if equity > 0 else 0,
            },
            labels={
                "type": "portfolio_snapshot",
                "is_over_leveraged": str(is_over_leveraged),
                "is_capital_exhausted": str(is_capital_exhausted),
                "is_in_drawdown": str(is_in_drawdown),
            },
            tags=["portfolio", "snapshot", "state"],
            summary=f"Portfolio: equity=${equity:,.0f}, buying_power=${buying_power:,.0f}, positions={position_count}, exposure=${total_exposure:,.0f}",
        )

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # GovernanceEngine Feedback Hooks (per SDR Section 9)
    # -------------------------------------------------------------------------
    # "Policy Checks (Pre-flight): Before any critical action is taken by
    #  another engine, the GovernanceEngine can perform checks."

    async def record_circuit_breaker_triggered(
        self,
        breaker_type: str,
        trigger_reason: str,
        error_count: int,
        error_rate: float,
        time_window_seconds: int,
        action_taken: str,
        affected_engines: list[str] | None = None,
    ) -> FeedbackRecord:
        """
        Record circuit breaker activation.

        Per SDR GovernanceEngine: Should have error rate monitoring
        that triggers trading halt when error rate exceeds threshold.

        On 12/17, this would have stopped trading after first 100 failures.

        Args:
            breaker_type: Type of circuit breaker (error_rate, drawdown, etc.)
            trigger_reason: Why it triggered
            error_count: Number of errors in window
            error_rate: Error rate (errors/minute or similar)
            time_window_seconds: Monitoring window
            action_taken: What system did (pause, halt, throttle)
            affected_engines: Which engines were affected
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.CIRCUIT_BREAKER_TRIGGERED,
            priority=FeedbackPriority.CRITICAL,
            source="governance_engine",
            parameters={
                "breaker_type": breaker_type,
                "affected_engines": affected_engines or [],
            },
            metrics={
                "error_count": error_count,
                "error_rate": error_rate,
                "time_window_seconds": time_window_seconds,
            },
            labels={
                "type": "circuit_breaker",
                "breaker_type": breaker_type,
                "action": action_taken,
            },
            tags=["governance", "circuit_breaker", breaker_type, action_taken],
            summary=f"Circuit breaker triggered: {breaker_type} - {error_count} errors ({error_rate:.1f}/min) in {time_window_seconds}s. Action: {action_taken}",
        )

        await self._store_record(record)
        return record

    async def record_signal_throttle(
        self,
        strategy: str,
        signals_generated: int,
        signals_allowed: int,
        throttle_reason: str,
        throttle_duration_seconds: int,
    ) -> FeedbackRecord:
        """
        Record signal throttling event.

        Per SDR GovernanceEngine: Trading should be throttled "if needed
        (to avoid excessive frequency or notional volume)."

        On 12/17, 6,000+ signals should have been throttled after capital exhausted.
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.SIGNAL_THROTTLE,
            priority=FeedbackPriority.HIGH,
            source="governance_engine",
            strategy=strategy,
            metrics={
                "signals_generated": signals_generated,
                "signals_allowed": signals_allowed,
                "signals_blocked": signals_generated - signals_allowed,
                "throttle_duration_seconds": throttle_duration_seconds,
            },
            labels={
                "type": "signal_throttle",
                "strategy": strategy,
                "reason": throttle_reason,
            },
            tags=["governance", "throttle", strategy],
            summary=f"Signal throttle: {strategy} generated {signals_generated}, allowed {signals_allowed}. Reason: {throttle_reason}",
        )

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # SignalEngine Feedback Hooks (per SDR Section 2)
    # -------------------------------------------------------------------------
    # "Governance Checks: Before releasing signals, consult the GovernanceEngine
    #  for any policy constraints."

    async def record_signal_batch(
        self,
        strategy: str,
        signal_count: int,
        long_count: int,
        short_count: int,
        avg_confidence: float,
        symbols: list[str],
        passed_risk_check: int,
        blocked_by_risk: int,
    ) -> FeedbackRecord:
        """
        Record signal generation batch from SignalEngine.

        Per SDR SignalEngine: "Multi-Model Ensemble... combine them to form
        a final signal" and "Governance Checks: Before releasing signals..."

        This tracks signal generation patterns for the LearningEngine.
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.PREDICTION_ACCURACY,
            priority=FeedbackPriority.NORMAL,
            source="signal_engine",
            strategy=strategy,
            parameters={
                "symbols": symbols[:20],  # Limit for storage
            },
            metrics={
                "signal_count": signal_count,
                "long_count": long_count,
                "short_count": short_count,
                "avg_confidence": avg_confidence,
                "passed_risk_check": passed_risk_check,
                "blocked_by_risk": blocked_by_risk,
                "pass_rate": passed_risk_check / signal_count if signal_count > 0 else 0,
            },
            labels={
                "type": "signal_batch",
                "strategy": strategy,
            },
            tags=["signal", "batch", strategy],
            summary=f"Signal batch: {strategy} generated {signal_count} signals ({long_count}L/{short_count}S), {passed_risk_check} passed risk, {blocked_by_risk} blocked",
        )

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # OrchestrationEngine Feedback Hooks (per SDR Section 1)
    # -------------------------------------------------------------------------
    # "Manage context propagation... Implement tracing and timing for each
    #  step to feed into performance analytics."

    async def record_trading_cycle(
        self,
        cycle_id: str,
        duration_ms: float,
        signals_generated: int,
        orders_submitted: int,
        orders_filled: int,
        orders_rejected: int,
        errors: list[dict[str, Any]] | None = None,
    ) -> FeedbackRecord:
        """
        Record trading cycle metrics from OrchestrationEngine.

        Per SDR OrchestrationEngine: "run_cycle(event): Processes one market
        event or tick through the pipeline, returns a composite result."

        This provides cycle-level visibility for the LearningEngine.
        """
        success_rate = orders_filled / orders_submitted if orders_submitted > 0 else 1.0
        error_count = len(errors) if errors else 0

        priority = (
            FeedbackPriority.HIGH
            if error_count > 0 or success_rate < 0.5
            else FeedbackPriority.NORMAL
        )

        record = FeedbackRecord(
            feedback_type=FeedbackType.EXECUTION_QUALITY,
            priority=priority,
            source="orchestration_engine",
            parameters={
                "cycle_id": cycle_id,
                "errors": errors[:10] if errors else [],  # Limit for storage
            },
            metrics={
                "duration_ms": duration_ms,
                "signals_generated": signals_generated,
                "orders_submitted": orders_submitted,
                "orders_filled": orders_filled,
                "orders_rejected": orders_rejected,
                "success_rate": success_rate,
                "error_count": error_count,
            },
            labels={
                "type": "trading_cycle",
                "cycle_id": cycle_id,
                "has_errors": str(error_count > 0),
            },
            tags=["orchestration", "cycle", "metrics"],
            summary=f"Cycle {cycle_id}: {signals_generated} signals → {orders_submitted} orders → {orders_filled} fills, {orders_rejected} rejected, {error_count} errors",
        )

        await self._store_record(record)
        return record

    async def record_error_rate_spike(
        self,
        error_type: str,
        error_count: int,
        time_window_seconds: int,
        error_rate_per_minute: float,
        threshold_per_minute: float,
        sample_errors: list[str] | None = None,
    ) -> FeedbackRecord:
        """
        Record error rate spike detection.

        Per SDR GovernanceEngine: Should monitor error patterns and
        trigger circuit breakers when rate exceeds threshold.

        On 12/17, error rate was ~100+/minute but no adaptation occurred.
        """
        record = FeedbackRecord(
            feedback_type=FeedbackType.ERROR_RATE_SPIKE,
            priority=FeedbackPriority.CRITICAL,
            source="orchestration_engine",
            parameters={
                "sample_errors": sample_errors[:5] if sample_errors else [],
            },
            metrics={
                "error_count": error_count,
                "time_window_seconds": time_window_seconds,
                "error_rate_per_minute": error_rate_per_minute,
                "threshold_per_minute": threshold_per_minute,
                "rate_vs_threshold": error_rate_per_minute / threshold_per_minute
                if threshold_per_minute > 0
                else 0,
            },
            labels={
                "type": "error_rate_spike",
                "error_type": error_type,
                "severity": "critical"
                if error_rate_per_minute > threshold_per_minute * 2
                else "warning",
            },
            tags=["error", "spike", error_type],
            summary=f"Error rate spike: {error_type} at {error_rate_per_minute:.1f}/min (threshold: {threshold_per_minute:.1f}/min). {error_count} errors in {time_window_seconds}s.",
        )

        await self._store_record(record)
        return record

    # -------------------------------------------------------------------------
    # Storage Operations
    # -------------------------------------------------------------------------

    async def _store_record(self, record: FeedbackRecord) -> None:
        """Store feedback record in all configured backends."""
        # Store in SQLite
        if self.db_manager and self.db_manager.is_connected:
            await self._store_sqlite(record)

        # Store in ChromaDB (with embedding)
        if self._feedback_collection is not None:
            await self._store_chroma(record)

        # Send to LearningEngine
        if self.learning_engine:
            await self._notify_learning_engine(record)

        logger.debug(f"Stored feedback: {record.feedback_id} ({record.feedback_type.value})")

    async def _store_sqlite(self, record: FeedbackRecord) -> None:
        """Store record in SQLite."""
        if not self.db_manager or not self.db_manager._connection:
            return

        sql = """
        INSERT INTO learning_feedback (
            feedback_id, feedback_type, priority, source, timestamp,
            strategy, symbol, timeframe, parameters, metrics,
            train_performance, test_performance, is_overfit, overfitting_score,
            market_regime, volatility_regime, data_quality_score,
            labels, tags, summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            record.feedback_id,
            record.feedback_type.value,
            record.priority.value,
            record.source,
            record.timestamp.isoformat(),
            record.strategy,
            record.symbol,
            record.timeframe,
            json.dumps(record.parameters),
            json.dumps(record.metrics),
            json.dumps(record.train_performance),
            json.dumps(record.test_performance),
            1 if record.is_overfit else 0,
            record.overfitting_score,
            record.market_regime,
            record.volatility_regime,
            record.data_quality_score,
            json.dumps(record.labels),
            json.dumps(record.tags),
            record.summary,
        )

        async with self.db_manager._lock:
            await self.db_manager._connection.execute(sql, values)
            await self.db_manager._connection.commit()

    async def _store_chroma(self, record: FeedbackRecord) -> None:
        """Store record in ChromaDB for semantic search."""
        if not self._feedback_collection:
            return

        # Generate embedding text
        text = record.to_embedding_text()

        # Store with metadata
        self._feedback_collection.add(
            documents=[text],
            metadatas=[
                {
                    "feedback_type": record.feedback_type.value,
                    "strategy": record.strategy or "",
                    "symbol": record.symbol or "",
                    "source": record.source,
                    "timestamp": record.timestamp.isoformat(),
                    "is_overfit": record.is_overfit,
                }
            ],
            ids=[record.feedback_id],
        )

    async def _notify_learning_engine(self, record: FeedbackRecord) -> None:
        """Notify LearningEngine of new feedback."""
        if not self.learning_engine:
            return

        from ordinis.engines.learning.core.models import EventType, LearningEvent

        # Map feedback type to learning event type
        event_type_map = {
            FeedbackType.SENSITIVITY_ANALYSIS: EventType.METRIC_RECORDED,
            FeedbackType.PARAMETER_SWEEP: EventType.METRIC_RECORDED,
            FeedbackType.TRADE_OUTCOME: EventType.POSITION_CLOSED,
            FeedbackType.SIGNAL_ACCURACY: EventType.SIGNAL_ACCURACY,
            FeedbackType.REGIME_CHANGE: EventType.DRIFT_DETECTED,
        }

        event_type = event_type_map.get(record.feedback_type, EventType.METRIC_RECORDED)

        event = LearningEvent(
            event_type=event_type,
            source_engine=record.source,
            symbol=record.symbol,
            payload=record.to_dict(),
            labels=record.labels,
        )

        self.learning_engine.record_event(event)

    async def _store_regime_history(
        self,
        regime_type: str,
        regime_value: str,
        symbol: str | None,
        confidence: float,
        indicators: dict[str, float] | None,
    ) -> None:
        """Store regime change in history table."""
        if not self.db_manager or not self.db_manager._connection:
            return

        sql = """
        INSERT INTO regime_history (
            regime_type, regime_value, symbol, start_time, confidence, indicators
        ) VALUES (?, ?, ?, ?, ?, ?)
        """

        values = (
            regime_type,
            regime_value,
            symbol,
            datetime.now(UTC).isoformat(),
            confidence,
            json.dumps(indicators or {}),
        )

        async with self.db_manager._lock:
            await self.db_manager._connection.execute(sql, values)
            await self.db_manager._connection.commit()

    # -------------------------------------------------------------------------
    # Optimal Parameters Management
    # -------------------------------------------------------------------------

    async def update_optimal_parameters(
        self,
        strategy: str,
        parameters: dict[str, Any],
        train_sharpe: float,
        test_sharpe: float,
        overfitting_score: float,
        symbol: str | None = None,
        timeframe: str | None = None,
        confidence: float = 0.5,
        source_feedback_id: str | None = None,
    ) -> None:
        """
        Update optimal parameters for a strategy.

        Args:
            strategy: Strategy name
            parameters: Optimal parameter configuration
            train_sharpe: In-sample Sharpe ratio
            test_sharpe: Out-of-sample Sharpe ratio
            overfitting_score: Degree of overfitting
            symbol: Symbol-specific (None for universal)
            timeframe: Timeframe-specific
            confidence: Confidence in these parameters
            source_feedback_id: ID of feedback record that triggered update
        """
        if not self.db_manager or not self.db_manager._connection:
            logger.warning("Cannot update optimal params: DB not connected")
            return

        # Deactivate previous optimal params
        deactivate_sql = """
        UPDATE optimal_parameters
        SET is_active = 0, valid_until = ?
        WHERE strategy = ? AND (symbol = ? OR (symbol IS NULL AND ? IS NULL))
            AND (timeframe = ? OR (timeframe IS NULL AND ? IS NULL))
            AND is_active = 1
        """

        now = datetime.now(UTC).isoformat()

        async with self.db_manager._lock:
            await self.db_manager._connection.execute(
                deactivate_sql,
                (now, strategy, symbol, symbol, timeframe, timeframe),
            )

            # Insert new optimal params
            insert_sql = """
            INSERT INTO optimal_parameters (
                strategy, symbol, timeframe, parameters,
                train_sharpe, test_sharpe, overfitting_score, confidence_score,
                valid_from, source_feedback_id, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """

            await self.db_manager._connection.execute(
                insert_sql,
                (
                    strategy,
                    symbol,
                    timeframe,
                    json.dumps(parameters),
                    train_sharpe,
                    test_sharpe,
                    overfitting_score,
                    confidence,
                    now,
                    source_feedback_id,
                ),
            )

            await self.db_manager._connection.commit()

        logger.info(f"Updated optimal params for {strategy}: {parameters}")

    async def get_optimal_parameters(
        self,
        strategy: str,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get current optimal parameters for a strategy.

        Args:
            strategy: Strategy name
            symbol: Symbol-specific params (falls back to universal)
            timeframe: Timeframe-specific params

        Returns:
            Optimal parameters dict or None
        """
        if not self.db_manager or not self.db_manager._connection:
            return None

        # Try symbol-specific first, then universal
        sql = """
        SELECT parameters, train_sharpe, test_sharpe, confidence_score
        FROM optimal_parameters
        WHERE strategy = ? AND is_active = 1
            AND (symbol = ? OR symbol IS NULL)
            AND (timeframe = ? OR timeframe IS NULL)
        ORDER BY
            CASE WHEN symbol IS NOT NULL THEN 0 ELSE 1 END,
            CASE WHEN timeframe IS NOT NULL THEN 0 ELSE 1 END,
            confidence_score DESC
        LIMIT 1
        """

        async with self.db_manager._lock:
            cursor = await self.db_manager._connection.execute(sql, (strategy, symbol, timeframe))
            row = await cursor.fetchone()

        if row:
            return {
                "parameters": json.loads(row[0]),
                "train_sharpe": row[1],
                "test_sharpe": row[2],
                "confidence": row[3],
            }

        return None

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    async def query_feedback(
        self,
        feedback_type: FeedbackType | None = None,
        strategy: str | None = None,
        symbol: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query feedback records from SQLite.

        Args:
            feedback_type: Filter by type
            strategy: Filter by strategy
            symbol: Filter by symbol
            since: Only records after this time
            limit: Maximum records to return

        Returns:
            List of feedback records as dicts
        """
        if not self.db_manager or not self.db_manager._connection:
            return []

        conditions = []
        params = []

        if feedback_type:
            conditions.append("feedback_type = ?")
            params.append(feedback_type.value)

        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        sql = f"""
        SELECT feedback_id, feedback_type, source, timestamp, strategy, symbol,
               parameters, metrics, train_performance, test_performance,
               is_overfit, overfitting_score, summary
        FROM learning_feedback
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ?
        """

        async with self.db_manager._lock:
            cursor = await self.db_manager._connection.execute(sql, params)
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "feedback_id": row[0],
                    "feedback_type": row[1],
                    "source": row[2],
                    "timestamp": row[3],
                    "strategy": row[4],
                    "symbol": row[5],
                    "parameters": json.loads(row[6]) if row[6] else {},
                    "metrics": json.loads(row[7]) if row[7] else {},
                    "train_performance": json.loads(row[8]) if row[8] else {},
                    "test_performance": json.loads(row[9]) if row[9] else {},
                    "is_overfit": bool(row[10]),
                    "overfitting_score": row[11],
                    "summary": row[12],
                }
            )

        return results

    def search_feedback(
        self,
        query: str,
        n_results: int = 10,
        feedback_type: str | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search for feedback in ChromaDB.

        Args:
            query: Natural language query
            n_results: Number of results
            feedback_type: Filter by type
            strategy: Filter by strategy

        Returns:
            List of matching feedback records
        """
        if not self._feedback_collection:
            return []

        where = {}
        if feedback_type:
            where["feedback_type"] = feedback_type
        if strategy:
            where["strategy"] = strategy

        results = self._feedback_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None,
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append(
                    {
                        "id": results["ids"][0][i] if results["ids"] else None,
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else None,
                    }
                )

        return output

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_sensitivity_summary(
        self,
        strategy: str,
        parameters: dict[str, Any],
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
        overfitting_score: float,
    ) -> str:
        """Generate human-readable summary of sensitivity analysis."""
        param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        train_sharpe = train_metrics.get("sharpe", 0)
        test_sharpe = test_metrics.get("sharpe", 0)
        train_pnl = train_metrics.get("total_pnl", 0)
        test_pnl = test_metrics.get("total_pnl", 0)

        overfit_status = "OVERFIT" if overfitting_score > 0.3 else "OK"

        return (
            f"Sensitivity: {strategy} with {param_str}\n"
            f"Train: Sharpe={train_sharpe:.2f}, PnL={train_pnl:.2f}%\n"
            f"Test: Sharpe={test_sharpe:.2f}, PnL={test_pnl:.2f}%\n"
            f"Overfitting: {overfitting_score:.2f} ({overfit_status})"
        )
