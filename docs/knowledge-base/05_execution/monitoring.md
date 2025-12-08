# Monitoring for Algorithmic Trading Systems

## Overview

Comprehensive monitoring is essential for algorithmic trading systems to ensure reliability, detect anomalies, track performance, and maintain regulatory compliance. This document covers monitoring strategies, metrics, alerting, and dashboards.

**Last Updated**: December 8, 2025

---

## 1. Monitoring Architecture

### 1.1 Monitoring Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MONITORING ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     APPLICATION METRICS                                │ │
│  │   Trading Performance | Signal Quality | Order Execution | P&L        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     SYSTEM METRICS                                     │ │
│  │   CPU | Memory | Disk | Network | Latency | Throughput                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    INFRASTRUCTURE METRICS                              │ │
│  │   Container Health | DB Connections | Queue Depth | API Rates         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐              │
│  │   Prometheus  │───▶│   Grafana     │    │   AlertMgr    │              │
│  │   (Metrics)   │    │  (Dashboards) │    │  (Alerts)     │              │
│  └───────────────┘    └───────────────┘    └───────────────┘              │
│                                                     │                       │
│                                                     ▼                       │
│                                            ┌───────────────┐               │
│                                            │  PagerDuty /  │               │
│                                            │  Slack / SMS  │               │
│                                            └───────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Monitoring Components

| Component | Purpose | Tools |
|-----------|---------|-------|
| **Metrics Collection** | Gather numerical time-series data | Prometheus, InfluxDB, StatsD |
| **Log Aggregation** | Centralize and search logs | ELK Stack, Loki, CloudWatch |
| **Tracing** | Track request flows | Jaeger, Zipkin, OpenTelemetry |
| **Visualization** | Dashboards and graphs | Grafana, Kibana, Datadog |
| **Alerting** | Notify on anomalies | AlertManager, PagerDuty, OpsGenie |

---

## 2. Trading-Specific Metrics

### 2.1 Performance Metrics

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import numpy as np


@dataclass
class TradingMetrics:
    """Core trading performance metrics."""

    # P&L Metrics
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    daily_pnl: float

    # Return Metrics
    total_return: float
    daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk Metrics
    current_drawdown: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float

    # Trade Metrics
    num_trades_today: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Position Metrics
    num_positions: int
    gross_exposure: float
    net_exposure: float
    buying_power_used: float

    # Execution Metrics
    avg_slippage: float
    avg_fill_time_ms: float
    order_rejection_rate: float


class MetricsCalculator:
    """Calculate trading performance metrics."""

    def __init__(self, returns: List[float], trades: List[dict]):
        self.returns = np.array(returns)
        self.trades = trades

    def calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate annualized Sharpe ratio.
        """
        if len(self.returns) < 2:
            return 0.0

        excess_returns = self.returns - (risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_sortino(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        """
        if len(self.returns) < 2:
            return 0.0

        excess_returns = self.returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        """
        if len(self.returns) < 2:
            return 0.0

        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return float(np.min(drawdowns))

    def calculate_win_rate(self) -> float:
        """Calculate percentage of winning trades."""
        if not self.trades:
            return 0.0

        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        """
        gross_profit = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t.get("pnl", 0) < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss
```

### 2.2 Signal Quality Metrics

```python
@dataclass
class SignalMetrics:
    """Metrics for evaluating signal quality."""

    # Accuracy
    signal_accuracy: float  # % of profitable signals
    direction_accuracy: float  # % correct direction

    # Timing
    avg_bars_to_profit: float
    avg_bars_held: float

    # Strength
    avg_signal_confidence: float
    signal_volatility: float

    # Decay
    signal_decay_rate: float  # How quickly edge decays

    # Correlation
    signal_autocorrelation: float
    inter_signal_correlation: float


class SignalAnalyzer:
    """Analyze signal quality and decay."""

    def __init__(self, signals: List[dict], outcomes: List[dict]):
        self.signals = signals
        self.outcomes = outcomes

    def calculate_signal_accuracy(self) -> float:
        """Calculate overall signal accuracy."""
        if not self.outcomes:
            return 0.0

        correct = sum(
            1 for o in self.outcomes
            if o.get("signal_direction") == o.get("actual_direction")
        )
        return correct / len(self.outcomes)

    def analyze_signal_decay(self, window: int = 30) -> dict:
        """
        Analyze how signal effectiveness decays over time.

        Returns:
            Dictionary with decay analysis
        """
        # Group outcomes by signal age
        recent_accuracy = self._accuracy_for_period(-window, 0)
        older_accuracy = self._accuracy_for_period(-2 * window, -window)

        decay_rate = (older_accuracy - recent_accuracy) / window if window > 0 else 0

        return {
            "recent_accuracy": recent_accuracy,
            "older_accuracy": older_accuracy,
            "decay_rate": decay_rate,
            "estimated_half_life_days": self._estimate_half_life(decay_rate)
        }

    def _accuracy_for_period(self, start_days: int, end_days: int) -> float:
        """Calculate accuracy for a specific time period."""
        # Implementation
        pass

    def _estimate_half_life(self, decay_rate: float) -> float:
        """Estimate signal half-life from decay rate."""
        if decay_rate <= 0:
            return float("inf")
        return 0.693 / decay_rate  # ln(2) / decay_rate
```

### 2.3 Execution Quality Metrics

```python
@dataclass
class ExecutionMetrics:
    """Order execution quality metrics."""

    # Latency
    order_to_ack_ms: float  # Time to broker acknowledgment
    order_to_fill_ms: float  # Time to complete fill
    market_data_latency_ms: float

    # Slippage
    avg_slippage_bps: float  # Basis points
    slippage_vs_arrival: float  # vs. price at decision
    slippage_vs_vwap: float  # vs. VWAP benchmark

    # Fill Quality
    fill_rate: float  # % orders fully filled
    partial_fill_rate: float
    rejection_rate: float

    # Market Impact
    realized_spread_bps: float
    price_impact_bps: float

    # Comparison
    implementation_shortfall: float


class ExecutionAnalyzer:
    """Analyze execution quality."""

    def __init__(self, executions: List[dict]):
        self.executions = executions

    def calculate_slippage(self, execution: dict) -> float:
        """
        Calculate slippage for a single execution.

        Slippage = (Actual Price - Expected Price) / Expected Price
        Positive = unfavorable, Negative = favorable
        """
        expected = execution["expected_price"]
        actual = execution["fill_price"]
        side = execution["side"]

        if side == "buy":
            return (actual - expected) / expected * 10000  # in bps
        else:
            return (expected - actual) / expected * 10000

    def calculate_vwap_deviation(self, execution: dict) -> float:
        """
        Calculate deviation from VWAP.

        VWAP is a common benchmark for execution quality.
        """
        vwap = execution["vwap"]
        fill_price = execution["fill_price"]
        side = execution["side"]

        if side == "buy":
            return (fill_price - vwap) / vwap * 10000
        else:
            return (vwap - fill_price) / vwap * 10000

    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        fill_price: float,
        side: str,
        shares: int
    ) -> float:
        """
        Calculate implementation shortfall.

        IS = (Fill Price - Decision Price) * Shares
        Measures total cost vs. if we could execute at decision price.
        """
        if side == "buy":
            return (fill_price - decision_price) * shares
        else:
            return (decision_price - fill_price) * shares
```

---

## 3. System Health Metrics

### 3.1 Core System Metrics

```python
import psutil
from dataclasses import dataclass
from typing import Dict


@dataclass
class SystemHealthMetrics:
    """System-level health metrics."""

    # CPU
    cpu_percent: float
    cpu_count: int
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float

    # Memory
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    swap_percent: float

    # Disk
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

    # Network
    network_bytes_sent: int
    network_bytes_recv: int
    network_connections: int

    # Process
    process_cpu_percent: float
    process_memory_mb: float
    process_threads: int
    process_open_files: int


class SystemMonitor:
    """Collect system health metrics."""

    def collect_metrics(self) -> SystemHealthMetrics:
        """Collect current system metrics."""
        cpu_times = psutil.cpu_times_percent()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()
        network = psutil.net_io_counters()

        # Get current process metrics
        process = psutil.Process()

        load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)

        return SystemHealthMetrics(
            cpu_percent=psutil.cpu_percent(),
            cpu_count=psutil.cpu_count(),
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2],
            memory_total_gb=memory.total / (1024 ** 3),
            memory_used_gb=memory.used / (1024 ** 3),
            memory_percent=memory.percent,
            swap_percent=swap.percent,
            disk_total_gb=disk.total / (1024 ** 3),
            disk_used_gb=disk.used / (1024 ** 3),
            disk_percent=disk.percent,
            disk_io_read_mb=disk_io.read_bytes / (1024 ** 2),
            disk_io_write_mb=disk_io.write_bytes / (1024 ** 2),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            network_connections=len(psutil.net_connections()),
            process_cpu_percent=process.cpu_percent(),
            process_memory_mb=process.memory_info().rss / (1024 ** 2),
            process_threads=process.num_threads(),
            process_open_files=len(process.open_files())
        )
```

### 3.2 Connection Health

```python
from enum import Enum
from typing import Dict, Optional
import asyncio


class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ConnectionHealth:
    """Health status of external connections."""
    status: ConnectionStatus
    latency_ms: Optional[float]
    last_check: datetime
    error_message: Optional[str] = None


class ConnectionMonitor:
    """Monitor external service connections."""

    def __init__(self):
        self.connections: Dict[str, ConnectionHealth] = {}

    async def check_broker_connection(self, broker_client) -> ConnectionHealth:
        """Check broker API connection."""
        start = datetime.utcnow()

        try:
            # Attempt lightweight API call
            account = await broker_client.get_account()

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            return ConnectionHealth(
                status=ConnectionStatus.CONNECTED,
                latency_ms=latency,
                last_check=datetime.utcnow()
            )

        except Exception as e:
            return ConnectionHealth(
                status=ConnectionStatus.DISCONNECTED,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def check_database_connection(self, db_pool) -> ConnectionHealth:
        """Check database connection."""
        start = datetime.utcnow()

        try:
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            return ConnectionHealth(
                status=ConnectionStatus.CONNECTED,
                latency_ms=latency,
                last_check=datetime.utcnow()
            )

        except Exception as e:
            return ConnectionHealth(
                status=ConnectionStatus.DISCONNECTED,
                latency_ms=None,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )

    async def check_all_connections(self) -> Dict[str, ConnectionHealth]:
        """Check all monitored connections."""
        tasks = {
            "broker": self.check_broker_connection(self.broker_client),
            "database": self.check_database_connection(self.db_pool),
            "redis": self.check_redis_connection(self.redis_client),
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        return dict(zip(tasks.keys(), results))
```

---

## 4. Prometheus Integration

### 4.1 Custom Metrics

```python
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server


class TradingPrometheusMetrics:
    """Prometheus metrics for trading system."""

    def __init__(self):
        # Counters (monotonically increasing)
        self.orders_total = Counter(
            "trading_orders_total",
            "Total number of orders submitted",
            ["symbol", "side", "type", "status"]
        )

        self.signals_total = Counter(
            "trading_signals_total",
            "Total signals generated",
            ["strategy", "direction"]
        )

        self.errors_total = Counter(
            "trading_errors_total",
            "Total errors by type",
            ["error_type", "component"]
        )

        # Gauges (can go up and down)
        self.portfolio_value = Gauge(
            "trading_portfolio_value_usd",
            "Current portfolio value"
        )

        self.unrealized_pnl = Gauge(
            "trading_unrealized_pnl_usd",
            "Current unrealized P&L"
        )

        self.position_count = Gauge(
            "trading_position_count",
            "Number of open positions"
        )

        self.drawdown_percent = Gauge(
            "trading_drawdown_percent",
            "Current drawdown percentage"
        )

        self.buying_power = Gauge(
            "trading_buying_power_usd",
            "Available buying power"
        )

        # Histograms (for latency distributions)
        self.order_latency = Histogram(
            "trading_order_latency_seconds",
            "Order execution latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        self.signal_latency = Histogram(
            "trading_signal_generation_seconds",
            "Signal generation latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self.slippage_bps = Histogram(
            "trading_slippage_basis_points",
            "Order slippage distribution",
            buckets=[-10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50]
        )

    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        latency: float,
        slippage: float
    ):
        """Record order metrics."""
        self.orders_total.labels(
            symbol=symbol,
            side=side,
            type=order_type,
            status=status
        ).inc()

        self.order_latency.observe(latency)
        self.slippage_bps.observe(slippage)

    def update_portfolio(
        self,
        value: float,
        pnl: float,
        positions: int,
        drawdown: float,
        buying_power: float
    ):
        """Update portfolio gauges."""
        self.portfolio_value.set(value)
        self.unrealized_pnl.set(pnl)
        self.position_count.set(positions)
        self.drawdown_percent.set(drawdown)
        self.buying_power.set(buying_power)


# Start metrics server
def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
```

### 4.2 Prometheus Configuration

```yaml
# prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'trading_rules.yml'

scrape_configs:
  - job_name: 'trading-engine'
    static_configs:
      - targets: ['trading:9090']
    scrape_interval: 5s  # More frequent for trading

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

---

## 5. Alerting Rules

### 5.1 Trading Alerts

```yaml
# trading_rules.yml

groups:
  - name: trading_critical
    rules:
      # Risk alerts
      - alert: MaxDrawdownExceeded
        expr: trading_drawdown_percent > 0.15
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Maximum drawdown exceeded"
          description: "Drawdown is {{ $value }}%, exceeding 15% limit"

      - alert: DailyLossLimitApproaching
        expr: trading_daily_pnl_usd < -5000
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Approaching daily loss limit"
          description: "Daily P&L is ${{ $value }}"

      # Connection alerts
      - alert: BrokerConnectionDown
        expr: trading_broker_connection_status != 1
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Broker connection lost"
          description: "Cannot connect to broker API"

      - alert: HighOrderLatency
        expr: histogram_quantile(0.95, trading_order_latency_seconds_bucket) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order latency detected"
          description: "95th percentile order latency is {{ $value }}s"

      # Execution alerts
      - alert: HighRejectionRate
        expr: rate(trading_orders_total{status="rejected"}[5m]) / rate(trading_orders_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order rejection rate"
          description: "{{ $value }}% of orders being rejected"

      - alert: HighSlippage
        expr: histogram_quantile(0.50, trading_slippage_basis_points_bucket) > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Elevated slippage detected"
          description: "Median slippage is {{ $value }} bps"

  - name: trading_system
    rules:
      # System health
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Trading process using {{ $value }}GB memory"

      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "Trading process CPU usage is {{ $value }}"

      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_activity_count > pg_settings_max_connections * 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### 5.2 Alert Manager Configuration

```yaml
# alertmanager.yml

global:
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alerts@ordinis.local'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'trading-team'

  routes:
    # Critical trading alerts - immediate
    - match:
        severity: critical
      receiver: 'trading-critical'
      group_wait: 0s
      repeat_interval: 1h

    # Warning alerts
    - match:
        severity: warning
      receiver: 'trading-team'

receivers:
  - name: 'trading-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
    slack_configs:
      - api_url: '<slack-webhook>'
        channel: '#trading-alerts'
        send_resolved: true

  - name: 'trading-team'
    email_configs:
      - to: 'trading-team@example.com'
    slack_configs:
      - api_url: '<slack-webhook>'
        channel: '#trading-alerts'
```

---

## 6. Grafana Dashboards

### 6.1 Trading Dashboard JSON

```json
{
  "dashboard": {
    "title": "Trading System Dashboard",
    "panels": [
      {
        "title": "Portfolio Value",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_portfolio_value_usd",
            "legendFormat": "Portfolio Value"
          }
        ]
      },
      {
        "title": "Daily P&L",
        "type": "stat",
        "targets": [
          {
            "expr": "trading_daily_pnl_usd",
            "legendFormat": "Daily P&L"
          }
        ]
      },
      {
        "title": "Current Drawdown",
        "type": "gauge",
        "targets": [
          {
            "expr": "trading_drawdown_percent * 100",
            "legendFormat": "Drawdown %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "max": 20,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 5},
                {"color": "orange", "value": 10},
                {"color": "red", "value": 15}
              ]
            }
          }
        }
      },
      {
        "title": "Order Latency (p95)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(trading_order_latency_seconds_bucket[5m]))",
            "legendFormat": "p95 Latency"
          }
        ]
      },
      {
        "title": "Orders by Status",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(trading_orders_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Slippage Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "trading_slippage_basis_points_bucket",
            "legendFormat": "Slippage"
          }
        ]
      }
    ]
  }
}
```

---

## 7. Logging Best Practices

### 7.1 Structured Logging

```python
import structlog
from datetime import datetime


def configure_logging():
    """Configure structured logging for trading system."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger()


# Usage examples
def log_order_submitted(order: dict):
    """Log order submission with full context."""
    logger.info(
        "order_submitted",
        order_id=order["id"],
        symbol=order["symbol"],
        side=order["side"],
        quantity=order["quantity"],
        price=order.get("limit_price"),
        order_type=order["type"],
        strategy=order.get("strategy"),
        signal_id=order.get("signal_id")
    )


def log_order_filled(order: dict, fill: dict):
    """Log order fill with execution details."""
    logger.info(
        "order_filled",
        order_id=order["id"],
        symbol=order["symbol"],
        side=order["side"],
        fill_price=fill["price"],
        fill_quantity=fill["quantity"],
        slippage_bps=calculate_slippage(order, fill),
        latency_ms=fill["latency_ms"]
    )


def log_risk_alert(alert_type: str, details: dict):
    """Log risk management alerts."""
    logger.warning(
        "risk_alert",
        alert_type=alert_type,
        **details
    )
```

### 7.2 Audit Trail

```python
class AuditLogger:
    """
    Immutable audit trail for regulatory compliance.
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.logger = structlog.get_logger("audit")

    def log_decision(
        self,
        decision_type: str,
        decision: str,
        context: dict,
        approved_by: str = "system"
    ):
        """
        Log a trading decision with full context.
        Immutable for regulatory compliance.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": decision_type,
            "decision": decision,
            "context": context,
            "approved_by": approved_by,
            "system_state": self._capture_system_state()
        }

        # Log to structured logger
        self.logger.info("decision", **entry)

        # Also append to immutable file
        self._append_to_audit_file(entry)

    def _capture_system_state(self) -> dict:
        """Capture current system state for audit."""
        return {
            "portfolio_value": get_portfolio_value(),
            "open_positions": get_position_count(),
            "daily_pnl": get_daily_pnl(),
            "drawdown": get_current_drawdown()
        }

    def _append_to_audit_file(self, entry: dict):
        """Append entry to append-only audit file."""
        # Implementation with file locking and integrity checks
        pass
```

---

## 8. Health Checks

### 8.1 Health Check Endpoint

```python
from fastapi import FastAPI, Response
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


app = FastAPI()


@app.get("/health")
async def health_check() -> dict:
    """
    Comprehensive health check endpoint.
    Returns overall system health status.
    """
    checks = {
        "broker_connection": await check_broker_health(),
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "market_data": await check_market_data_health(),
        "trading_engine": check_trading_engine_health()
    }

    # Determine overall status
    if all(c["status"] == "healthy" for c in checks.values()):
        overall = HealthStatus.HEALTHY
    elif any(c["status"] == "unhealthy" for c in checks.values()):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED

    return {
        "status": overall.value,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/ready")
async def readiness_check() -> Response:
    """
    Kubernetes readiness probe.
    Returns 200 if ready to receive traffic.
    """
    health = await health_check()

    if health["status"] in ["healthy", "degraded"]:
        return Response(status_code=200)
    else:
        return Response(status_code=503)


@app.get("/live")
async def liveness_check() -> Response:
    """
    Kubernetes liveness probe.
    Returns 200 if process is alive.
    """
    # Simple check that process is responsive
    return Response(status_code=200)
```

---

## 9. References

- Google SRE Book: https://sre.google/sre-book/monitoring-distributed-systems/
- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- OpenTelemetry: https://opentelemetry.io/docs/
- The Art of Monitoring (Turnbull, 2016)
