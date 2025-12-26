# L2-06 — OBS — Observability

## 1. Identifier
- **ID:** `L2-06`
- **Component:** OBS — Observability
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Structured logging (JSON) with mandatory trace_id/correlation_id propagation.
- Metrics collection and export (Prometheus endpoint in DB1).
- Distributed tracing (OpenTelemetry) across engine boundaries.
- Alert rule definitions and SLO tracking for DB1.

### Out of scope
- Managed observability vendors (Datadog/NewRelic) — adapters can be added later.
- On-call paging integration (PagerDuty) — can be integrated later.

## 3. Responsibilities
- Provide logging, metrics, and tracing primitives used by all components.
- Enforce standard log/event fields (trace/correlation/component/action).
- Provide SLO definitions and alert thresholds (see L3-14).
- Guarantee redaction is applied via SEC before log persistence/emission.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class LogContext:
    component_id: str
    correlation_id: str
    trace_id: str
    span_id: str
    tags: Dict[str,str]

def get_logger(name: str): ...
def log_info(msg: str, ctx: LogContext, **fields: Any) -> None: ...
def log_error(msg: str, ctx: LogContext, **fields: Any) -> None: ...

class Metrics:
    def counter(self, name: str, labels: Dict[str,str] = None): ...
    def histogram(self, name: str, buckets: list[float] = None, labels: Dict[str,str] = None): ...
    def gauge(self, name: str, labels: Dict[str,str] = None): ...

class Tracing:
    def start_span(self, name: str, ctx: LogContext): ...
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.alert.raised
- ordinis.ops.slo.breached

### Schemas
- Log schema: mandatory fields + freeform fields (redacted)
- Metric naming schema
- Trace span attributes standard

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_OBS_EXPORT_FAILED` | Metrics endpoint/export failure | Yes |
| `E_OBS_REDACTION_REQUIRED` | Attempted to log unredacted data | No |

## 5. Dependencies
### Upstream
- SEC for redaction policies
- CFG for enabling exporters and log levels

### Downstream
- All engines/services
- L3-14 OBS-SLOS for SLO definitions and alert rules

### External services / libraries
- OpenTelemetry SDK
- Prometheus exposition format (HTTP endpoint)

## 6. Data & State
### Storage
- DB1: logs to stdout + optional file sink; metrics in-memory until scraped; traces to OTLP collector (optional) or local exporter.

### Caching
- Logger instances cached per name.
- Metric objects cached by name+labelset.

### Retention
- Log retention is deployment-defined (file rotation).
- Trace retention is deployment-defined (collector).

### Migrations / Versioning
- Log schema additions must be backward compatible; consumers tolerate unknown fields.

## 7. Algorithms / Logic
### Trace propagation
- ORCH creates correlation_id + root trace_id at cycle start.
- SB publishes events with trace_id; consumers continue spans.

### Logging rules
- All logs must be structured and must include trace_id + correlation_id.
- Sensitive fields are redacted via SEC; logging unredacted sensitive fields is a hard error in prod.

### Metrics rules
- Component-level counters/histograms are mandatory for latency and error rates.
- KPI metrics (system + trading) are emitted by ANA and BENCH, but exported through OBS.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| OBS_LOG_LEVEL | string | INFO | P0 | DEBUG/INFO/WARN/ERROR |
| OBS_LOG_FORMAT | string | json | P0 | `json` required in prod. |
| OBS_METRICS_ENABLE | bool | true | P0 | Enable /metrics. |
| OBS_METRICS_PORT | int | 9100 | P0 | Local port for Prometheus scrape. |
| OBS_TRACING_ENABLE | bool | true | P0 | Enable OpenTelemetry. |
| OBS_TRACING_EXPORTER | string | otlp | P0 | `otlp` or `console`. |

**Environment variables**
- `ORDINIS_OBS_LOG_LEVEL`

## 9. Non-Functional Requirements
- Logging overhead: p95 < 2ms per log event at INFO under 1k logs/sec.
- Metrics export must not block engine execution; scrape endpoint is lock-free where possible.
- Trace spans must not exceed 1% CPU overhead under typical load.

## 10. Observability
**Standard metrics (mandatory per component)**
- `component_requests_total{component,action,result}`
- `component_latency_ms{component,action}`
- `component_errors_total{component,action,error_code}`

**Alerting**
- See [L3-14 OBS-SLOS — SLOs, Alerts & Dashboards](../L3/L3-14_OBS-SLOS_SLOs_Alerts_and_Dashboards.md).

## 11. Failure Modes & Recovery
- Metrics exporter down: continue execution; log error; raise ops alert if persistent.
- Tracing exporter down: fall back to local sampling; continue.
- Redaction failure: fail closed for prod; dev may warn.

## 12. Test Plan
### Unit tests
- Logger includes mandatory fields.
- Redaction applied before serialization.
- Metrics registry returns stable metric objects.

### Integration tests
- End-to-end trace_id propagated from ORCH to SIG to RISK to EXEC.
- /metrics endpoint exposes expected metrics.

### Acceptance criteria
- All DB1 components emit required metrics and structured logs.

## 13. Open Questions / Risks
- Do we ship an embedded OTLP collector for DB1 or require external collector?
- What log sink is default on Windows (stdout vs file) for dev ergonomics?

## 14. Traceability
### Parent
- DB1 System SRS: Reliability and governance require end-to-end observability.

### Children / Related
- [L3-14 OBS-SLOS — SLOs, Alerts & Dashboards](../L3/L3-14_OBS-SLOS_SLOs_Alerts_and_Dashboards.md)
- [L2-04 SEC — Security](L2-04_SEC_Security_Identity_and_Secrets.md)

### Originating system requirements
- OBS-FR-001, OBS-FR-002, OBS-FR-003
