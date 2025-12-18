"""
Metrics collectors for Ordinis engines.

Collects metrics from OrchestrationEngine, FlowRoute, and other engines
for export to Prometheus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

from prometheus_client import Counter, Gauge, Histogram, Info

if TYPE_CHECKING:
    from ordinis.engines.orchestration.core.models import CycleResult

logger = logging.getLogger(__name__)


# Prometheus metric definitions
# Throughput counters
CYCLES_TOTAL = Counter(
    "ordinis_cycles_total",
    "Total trading cycles executed",
    ["status"],
)

SIGNALS_TOTAL = Counter(
    "ordinis_signals_total",
    "Total signals generated",
    ["status"],  # approved, rejected
)

ORDERS_TOTAL = Counter(
    "ordinis_orders_total",
    "Total orders processed",
    ["status"],  # submitted, filled, rejected
)

# Latency histograms
CYCLE_DURATION = Histogram(
    "ordinis_cycle_duration_seconds",
    "Trading cycle duration",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

STAGE_DURATION = Histogram(
    "ordinis_stage_duration_seconds",
    "Pipeline stage duration",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Workflow health gauges
CIRCUIT_BREAKER_STATE = Gauge(
    "ordinis_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["engine"],
)

PIPELINE_HEALTH = Gauge(
    "ordinis_pipeline_health",
    "Pipeline health status (0=unhealthy, 1=degraded, 2=healthy)",
)

ERROR_RATE = Gauge(
    "ordinis_error_rate",
    "Current error rate",
    ["error_type"],
)

# Paper trading specific
PAPER_FILLS_TOTAL = Counter(
    "ordinis_paper_fills_total",
    "Total paper trade fills",
)

PAPER_SLIPPAGE = Histogram(
    "ordinis_paper_slippage_bps",
    "Paper trade slippage in basis points",
    buckets=[0, 1, 2, 5, 10, 25, 50, 100],
)

PAPER_COMMISSION = Counter(
    "ordinis_paper_commission_total",
    "Total paper trade commissions",
)

# Portfolio metrics
EQUITY = Gauge(
    "ordinis_equity",
    "Current portfolio equity",
)

CASH = Gauge(
    "ordinis_cash",
    "Current cash balance",
)

BUYING_POWER = Gauge(
    "ordinis_buying_power",
    "Current buying power",
)

POSITION_COUNT = Gauge(
    "ordinis_position_count",
    "Number of open positions",
)

# Engine info
ENGINE_INFO = Info(
    "ordinis_engine",
    "Engine information",
)


class EngineProtocol(Protocol):
    """Protocol for engines that expose metrics."""

    def get_metrics(self) -> Any:
        """Get current metrics."""
        ...


@dataclass
class CollectorState:
    """Internal state for the metrics collector."""

    last_cycle_count: int = 0
    last_signal_count: int = 0
    last_order_count: int = 0
    last_fill_count: int = 0
    last_collection_time: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collects metrics from Ordinis engines for Prometheus export.

    Usage:
        collector = MetricsCollector()
        collector.register_orchestration_engine(engine)
        collector.start()  # Starts background collection
    """

    def __init__(self, collection_interval: float = 1.0) -> None:
        """
        Initialize metrics collector.

        Args:
            collection_interval: How often to collect metrics (seconds)
        """
        self._collection_interval = collection_interval
        self._orchestration_engine: Any | None = None
        self._flowroute_engine: Any | None = None
        self._state = CollectorState()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Set initial engine info
        ENGINE_INFO.info(
            {
                "version": "1.0.0",
                "mode": "paper",
                "environment": "development",
            }
        )

    def register_orchestration_engine(self, engine: Any) -> None:
        """Register the orchestration engine for metrics collection."""
        with self._lock:
            self._orchestration_engine = engine
            logger.info("Registered OrchestrationEngine for metrics collection")

    def register_flowroute_engine(self, engine: Any) -> None:
        """Register the FlowRoute engine for metrics collection."""
        with self._lock:
            self._flowroute_engine = engine
            logger.info("Registered FlowRouteEngine for metrics collection")

    def start(self) -> None:
        """Start background metrics collection."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="metrics-collector",
        )
        self._thread.start()
        logger.info("Started metrics collection thread")

    def stop(self) -> None:
        """Stop background metrics collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Stopped metrics collection thread")

    def _collection_loop(self) -> None:
        """Background loop for collecting metrics."""
        while self._running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            time.sleep(self._collection_interval)

    def _collect_metrics(self) -> None:
        """Collect metrics from all registered engines."""
        with self._lock:
            if self._orchestration_engine:
                self._collect_orchestration_metrics()
            if self._flowroute_engine:
                self._collect_flowroute_metrics()

    def _collect_orchestration_metrics(self) -> None:
        """Collect metrics from OrchestrationEngine."""
        if not self._orchestration_engine:
            return

        try:
            # Get pipeline metrics
            metrics = self._orchestration_engine.get_metrics()
            if not metrics:
                return

            # Update cycle counters (delta)
            new_successful = metrics.successful_cycles - self._state.last_cycle_count
            if new_successful > 0:
                CYCLES_TOTAL.labels(status="completed").inc(new_successful)

            new_failed = metrics.failed_cycles
            if new_failed > 0:
                CYCLES_TOTAL.labels(status="failed").inc(new_failed)

            self._state.last_cycle_count = metrics.successful_cycles

            # Update signal counters
            new_approved = metrics.approved_signals - self._state.last_signal_count
            if new_approved > 0:
                SIGNALS_TOTAL.labels(status="approved").inc(new_approved)

            SIGNALS_TOTAL.labels(status="rejected")._value.set(metrics.rejected_signals)
            self._state.last_signal_count = metrics.approved_signals

            # Update order counters
            ORDERS_TOTAL.labels(status="submitted")._value.set(metrics.total_orders)
            ORDERS_TOTAL.labels(status="filled")._value.set(metrics.filled_orders)
            ORDERS_TOTAL.labels(status="rejected")._value.set(metrics.rejected_orders)

            # Get last cycle for detailed metrics
            last_cycle = self._orchestration_engine.get_last_cycle()
            if last_cycle and last_cycle.total_duration_ms > 0:
                # Record cycle duration
                CYCLE_DURATION.observe(last_cycle.total_duration_ms / 1000.0)

                # Record stage durations
                if last_cycle.data_latency_ms > 0:
                    STAGE_DURATION.labels(stage="data_fetch").observe(
                        last_cycle.data_latency_ms / 1000.0
                    )
                if last_cycle.signal_latency_ms > 0:
                    STAGE_DURATION.labels(stage="signal_generation").observe(
                        last_cycle.signal_latency_ms / 1000.0
                    )
                if last_cycle.risk_latency_ms > 0:
                    STAGE_DURATION.labels(stage="risk_evaluation").observe(
                        last_cycle.risk_latency_ms / 1000.0
                    )
                if last_cycle.execution_latency_ms > 0:
                    STAGE_DURATION.labels(stage="order_execution").observe(
                        last_cycle.execution_latency_ms / 1000.0
                    )
                if last_cycle.analytics_latency_ms > 0:
                    STAGE_DURATION.labels(stage="analytics").observe(
                        last_cycle.analytics_latency_ms / 1000.0
                    )

            # Update health status
            health = getattr(self._orchestration_engine, "health_status", None)
            if health:
                health_value = {"healthy": 2, "degraded": 1, "unhealthy": 0}.get(
                    health.level.value if hasattr(health.level, "value") else str(health.level),
                    0,
                )
                PIPELINE_HEALTH.set(health_value)

        except Exception as e:
            logger.error(f"Error collecting orchestration metrics: {e}")

    def _collect_flowroute_metrics(self) -> None:
        """Collect metrics from FlowRouteEngine."""
        if not self._flowroute_engine:
            return

        try:
            # Get circuit breaker state
            circuit_breaker = getattr(self._flowroute_engine, "_circuit_breaker", None)
            if circuit_breaker:
                state_value = {
                    "closed": 0,
                    "open": 1,
                    "half_open": 2,
                }.get(circuit_breaker.state.value, 0)
                CIRCUIT_BREAKER_STATE.labels(engine="flowroute").set(state_value)

                # Get error rate from circuit breaker metrics
                cb_metrics = getattr(circuit_breaker, "metrics", None)
                if cb_metrics:
                    ERROR_RATE.labels(error_type="execution").set(cb_metrics.current_error_rate)

            # Get account state
            account = getattr(self._flowroute_engine, "_account_state", None)
            if account:
                EQUITY.set(float(getattr(account, "equity", 0)))
                CASH.set(float(getattr(account, "cash", 0)))
                BUYING_POWER.set(float(getattr(account, "buying_power", 0)))

            # Get positions
            positions = getattr(self._flowroute_engine, "_positions", {})
            POSITION_COUNT.set(len(positions))

        except Exception as e:
            logger.error(f"Error collecting flowroute metrics: {e}")

    def record_cycle(self, result: CycleResult) -> None:
        """
        Record metrics from a completed cycle.

        This can be called directly from OrchestrationEngine for real-time updates.
        """
        try:
            # Record cycle status
            status = "completed" if result.status.value == "completed" else "failed"
            CYCLES_TOTAL.labels(status=status).inc()

            # Record duration
            if result.total_duration_ms > 0:
                CYCLE_DURATION.observe(result.total_duration_ms / 1000.0)

            # Record signals
            SIGNALS_TOTAL.labels(status="approved").inc(result.signals_approved)
            rejected = result.signals_generated - result.signals_approved
            if rejected > 0:
                SIGNALS_TOTAL.labels(status="rejected").inc(rejected)

            # Record orders
            ORDERS_TOTAL.labels(status="submitted").inc(result.orders_submitted)
            ORDERS_TOTAL.labels(status="filled").inc(result.orders_filled)
            ORDERS_TOTAL.labels(status="rejected").inc(result.orders_rejected)

            # Record stage durations
            for stage in result.stages:
                if stage.duration_ms > 0:
                    STAGE_DURATION.labels(stage=stage.stage.value).observe(
                        stage.duration_ms / 1000.0
                    )

        except Exception as e:
            logger.error(f"Error recording cycle metrics: {e}")

    def record_paper_fill(
        self,
        slippage_bps: float,
        commission: float,
    ) -> None:
        """Record a paper trade fill."""
        PAPER_FILLS_TOTAL.inc()
        PAPER_SLIPPAGE.observe(slippage_bps)
        PAPER_COMMISSION.inc(commission)

    def update_portfolio(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        position_count: int,
    ) -> None:
        """Update portfolio metrics."""
        EQUITY.set(equity)
        CASH.set(cash)
        BUYING_POWER.set(buying_power)
        POSITION_COUNT.set(position_count)

    def set_circuit_breaker_state(self, engine: str, state: str) -> None:
        """Update circuit breaker state."""
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
        CIRCUIT_BREAKER_STATE.labels(engine=engine).set(state_value)

    def set_error_rate(self, error_type: str, rate: float) -> None:
        """Update error rate gauge."""
        ERROR_RATE.labels(error_type=error_type).set(rate)
