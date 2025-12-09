# Monitoring and Logging

Comprehensive monitoring and logging system for observability, debugging, and performance tracking.

## Overview

The monitoring system provides:
- **Structured Logging** - Using loguru with rotation and multiple formats
- **Metrics Collection** - Performance and trading metrics tracking
- **Health Checks** - System health monitoring and alerting
- **Observability** - Complete visibility into system operations

## Components

### 1. Logger (`monitoring.logger`)

Centralized logging with structured output and automatic rotation.

**Setup:**
```python
from monitoring import setup_logging, get_logger

# Configure logging
setup_logging(
    log_level="INFO",
    log_file="logs/ordinis.log",
    rotation="100 MB",
    retention="30 days",
    json_format=False  # Set True for structured JSON logs
)

# Get logger for your module
logger = get_logger(__name__)

# Use it
logger.info("Starting backtest")
logger.warning("Low confidence signal")
logger.error("API call failed")
```

**Features:**
- Color-coded console output
- Automatic file rotation by size/time
- JSON format for log aggregation
- Structured logging with context
- Exception tracking with stack traces

**Decorators:**
```python
from monitoring.logger import log_execution_time, log_exception

@log_execution_time
def run_backtest():
    # Automatically logs execution time
    pass

@log_exception()
def risky_operation():
    # Automatically logs exceptions
    pass
```

### 2. Metrics Collector (`monitoring.metrics`)

Tracks performance metrics and system statistics.

**Usage:**
```python
from monitoring import get_metrics_collector

# Get global collector
metrics = get_metrics_collector()

# Record operations
metrics.record_operation(success=True, execution_time=0.5)
metrics.record_operation(success=False)

# Record signals
metrics.record_signal(generated=True)
metrics.record_signal(executed=True)
metrics.record_signal(rejected=True)

# Record API calls
metrics.record_api_call(success=True, response_time=0.2)

# Custom metrics
metrics.increment_counter("trades_today", 1)
metrics.set_gauge("portfolio_value", 125000.0)

# Get metrics
current_metrics = metrics.get_metrics()
print(f"Success rate: {current_metrics.success_rate:.2%}")
print(f"Signal execution rate: {current_metrics.signal_execution_rate:.2%}")

# Get summary
summary = metrics.get_summary()
```

**Tracked Metrics:**
- Operation counts (total, successful, failed)
- Execution times (avg, min, max)
- Signal statistics (generated, executed, rejected)
- API call metrics (count, errors, response times)
- Custom counters and gauges

**Performance Metrics Object:**
```python
@dataclass
class PerformanceMetrics:
    # Execution
    total_operations: int
    successful_operations: int
    failed_operations: int

    # Timing
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float

    # Trading
    signals_generated: int
    signals_executed: int
    signals_rejected: int

    # API
    api_calls: int
    api_errors: int
    api_avg_response_time: float

    # Custom
    custom_metrics: dict[str, Any]

    # Calculated properties
    success_rate: float
    error_rate: float
    signal_execution_rate: float
```

### 3. Health Checks (`monitoring.health`)

Monitors system component health and overall status.

**Usage:**
```python
from monitoring import get_health_check, HealthCheck, HealthStatus

# Get global health check
health = get_health_check()

# Register custom check
def my_custom_check() -> HealthCheckResult:
    try:
        # Check something
        if everything_ok:
            return HealthCheckResult(
                name="my_check",
                status=HealthStatus.HEALTHY,
                message="All systems operational"
            )
        else:
            return HealthCheckResult(
                name="my_check",
                status=HealthStatus.DEGRADED,
                message="Performance degraded"
            )
    except Exception as e:
        return HealthCheckResult(
            name="my_check",
            status=HealthStatus.UNHEALTHY,
            message=f"Check failed: {e}"
        )

health.register_check("my_check", my_custom_check)

# Run specific check
result = health.run_check("disk_space")
print(f"Status: {result.status.value}")
print(f"Message: {result.message}")

# Run all checks
all_results = health.run_all_checks()

# Get overall status
overall = health.get_overall_status()
if overall == HealthStatus.UNHEALTHY:
    print("ALERT: System unhealthy!")

# Get comprehensive report
report = health.get_health_report()
```

**Health Status Levels:**
- `HEALTHY` - Everything working normally
- `DEGRADED` - Partial functionality, issues detected
- `UNHEALTHY` - Critical issues, immediate attention needed
- `UNKNOWN` - Cannot determine status

**Built-in Health Checks:**
- `disk_space` - Disk space availability
- `memory` - Memory usage
- `database` - Database connectivity (when implemented)
- `api` - External API connectivity (when implemented)

## Integration Examples

### Backtesting with Monitoring

```python
from monitoring import setup_logging, get_logger, get_metrics_collector
from engines.proofbench import SimulationEngine, SimulationConfig

# Setup
setup_logging(log_level="INFO", log_file="logs/backtest.log")
logger = get_logger(__name__)
metrics = get_metrics_collector()

# Run backtest
logger.info("Starting backtest")
config = SimulationConfig(initial_capital=100000.0)
sim = SimulationEngine(config=config)

# Track operations
def on_bar(engine, symbol, bar):
    start_time = time.perf_counter()

    try:
        # Generate signal
        signal = strategy.generate_signal(data, bar.timestamp)
        metrics.record_signal(generated=True)

        if signal:
            # Execute
            execute_signal(signal)
            metrics.record_signal(executed=True)

        # Record success
        exec_time = time.perf_counter() - start_time
        metrics.record_operation(success=True, execution_time=exec_time)

    except Exception as e:
        logger.exception(f"Error in on_bar: {e}")
        metrics.record_operation(success=False)

sim.on_bar = on_bar
results = sim.run()

# Log summary
summary = metrics.get_summary()
logger.info(f"Backtest complete: {summary['total_operations']} operations")
logger.info(f"Success rate: {summary['success_rate']:.2%}")
logger.info(f"Signal execution rate: {summary['signal_execution_rate']:.2%}")
```

### CLI with Health Checks

```python
from monitoring import get_health_check

def cli_health_command():
    """CLI command to check system health."""
    health = get_health_check()
    report = health.get_health_report()

    print(f"Overall Status: {report['overall_status'].upper()}")
    print("\nComponent Health:")

    for name, check in report['checks'].items():
        status_emoji = {
            'healthy': '',
            'degraded': '',
            'unhealthy': '',
            'unknown': '?'
        }
        emoji = status_emoji.get(check['status'], '?')
        print(f"  {emoji} {name}: {check['message']}")
```

### Production Monitoring

```python
from monitoring import setup_logging, get_metrics_collector, get_health_check
import time

# Setup with JSON logging for log aggregation
setup_logging(
    log_level="INFO",
    log_file="logs/production.log",
    json_format=True  # For ELK/Splunk/etc
)

# Periodic health checks
def monitor_health():
    """Run health checks every minute."""
    health = get_health_check()

    while True:
        report = health.get_health_report()

        if report['overall_status'] != 'healthy':
            # Send alert
            send_alert(report)

        time.sleep(60)

# Periodic metrics reporting
def report_metrics():
    """Report metrics every 5 minutes."""
    metrics = get_metrics_collector()

    while True:
        summary = metrics.get_summary()
        # Send to monitoring system
        send_to_prometheus(summary)

        time.sleep(300)
```

## Configuration

### Log Levels

- `DEBUG` - Detailed debugging information
- `INFO` - General informational messages
- `WARNING` - Warning messages for potential issues
- `ERROR` - Error messages for failures
- `CRITICAL` - Critical failures requiring immediate attention

### Log Rotation

```python
setup_logging(
    rotation="100 MB",  # Rotate when file reaches 100 MB
    # OR
    rotation="1 day",   # Rotate daily at midnight
    # OR
    rotation="1 week",  # Rotate weekly

    retention="30 days" # Keep logs for 30 days
)
```

### JSON Logging

For structured logging and log aggregation:

```python
setup_logging(json_format=True)
```

Output format:
```json
{
    "text": "Starting backtest",
    "record": {
        "time": {"timestamp": 1234567890.123},
        "level": {"name": "INFO"},
        "message": "Starting backtest",
        "name": "backtest",
        "function": "run_backtest",
        "line": 42
    }
}
```

## Best Practices

### 1. Log Appropriately

```python
#  Good - Informative messages
logger.info(f"Backtest started: {symbol}, {len(data)} bars")
logger.warning(f"Low confidence signal: {signal.probability:.2%}")
logger.error(f"API call failed: {status_code}")

#  Bad - Too verbose or not useful
logger.debug("Entering function")  # Unless actually debugging
logger.info("Data processed")      # Too vague
```

### 2. Use Structured Context

```python
# Add context to logs
logger.bind(symbol="SPY", strategy="RSI").info("Signal generated")
logger.bind(user_id=123).warning("API rate limit approaching")
```

### 3. Track Important Metrics

```python
# Record key events
metrics.record_signal(generated=True)
metrics.increment_counter("profitable_trades")
metrics.set_gauge("current_drawdown", -0.05)
```

### 4. Monitor Health Proactively

```python
# Check health before critical operations
health = get_health_check()
if health.get_overall_status() != HealthStatus.HEALTHY:
    logger.warning("System unhealthy, skipping operation")
    return
```

### 5. Handle Exceptions

```python
# Always log exceptions with context
try:
    result = risky_operation()
except Exception as e:
    logger.exception(f"Operation failed for {symbol}")
    metrics.record_operation(success=False)
    raise
```

## Troubleshooting

### Logs Not Appearing

```python
# Ensure logging is configured
setup_logging(log_level="INFO")

# Check file permissions
log_file = "logs/app.log"
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
```

### Metrics Not Updating

```python
# Use global collector
metrics = get_metrics_collector()  # Not MetricsCollector()

# Check if operations are recorded
metrics.record_operation(success=True)
print(metrics.get_summary())
```

### Health Checks Failing

```python
# Run individual check to debug
health = get_health_check()
result = health.run_check("disk_space")
print(f"Status: {result.status.value}")
print(f"Details: {result.details}")
```

## Future Enhancements

Planned improvements:
- Prometheus metrics export
- Grafana dashboard templates
- Sentry error tracking integration
- CloudWatch/Datadog integration
- Distributed tracing support
- Custom alerting rules

---

**Version:** 1.0.0
**Last Updated:** 2025-11-29
