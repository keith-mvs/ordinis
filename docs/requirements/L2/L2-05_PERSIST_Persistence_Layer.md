# L2-05 — PERSIST — Persistence Layer

## 1. Identifier
- **ID:** `L2-05`
- **Component:** PERSIST — Persistence Layer
- **Level:** L2

## 2. Purpose / Scope
### In scope
- SQLite storage (WAL mode) for DB1 system state: orders, fills, positions, runs, audit, configs, artifacts.
- Repository interfaces with transactional semantics.
- Schema migrations and versioning.
- Append-only audit chain storage support.

### Out of scope
- Distributed databases and horizontal sharding (P2).
- Long-term cold archival to object storage (P2).

## 3. Responsibilities
- Own database schema definitions and migrations.
- Provide repository interfaces used by engines and services.
- Guarantee atomic writes for multi-entity updates (orders+fills+portfolio).
- Support append-only audit records with hash chaining and integrity checks.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Protocol, Optional, Sequence, Mapping, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class DBHealth:
    ok: bool
    latency_ms: float
    wal_mode: bool
    user_version: int

class Database(Protocol):
    def begin(self): ...
    def commit(self): ...
    def rollback(self): ...
    def execute(self, sql: str, params: Sequence[Any] = ()): ...
    def health(self) -> DBHealth: ...

class OrdersRepo(Protocol):
    def upsert_intent(self, intent: Mapping[str, Any]) -> None: ...
    def set_submitted(self, order_id: str, broker_order_id: str, status: str) -> None: ...
    def get_by_idempotency_key(self, key: str) -> Optional[Mapping[str, Any]]: ...

class AuditRepo(Protocol):
    def append(self, audit_record: Mapping[str, Any]) -> None: ...
    def verify_chain(self, since_id: Optional[int] = None) -> bool: ...
```

### Events
**Consumes**
- ordinis.* (engines emit events; PERSIST stores selected ones as state/audit)

**Emits**
- ordinis.ops.persist.error
- ordinis.ops.persist.migration.applied

### Schemas
- DB schema: `orders`, `fills`, `positions`, `runs`, `audit_log`, `run_config_snapshot`, `artifacts`, `dlq`

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_DB_UNAVAILABLE` | Database file locked/unavailable | Yes |
| `E_DB_MIGRATION_FAILED` | Migration failure at startup | No |
| `E_DB_CONSTRAINT` | Constraint violation (e.g., duplicate primary key) | No |
| `E_AUDIT_CHAIN_BROKEN` | Audit hash chain verification failed | No |

## 5. Dependencies
### Upstream
- CFG for database path/DSN
- SEC for encryption/redaction policy (if encrypting fields later)

### Downstream
- ORCH (run tracking)
- EXEC (orders/fills persistence)
- PORT/LEDG (positions/ledger state)
- GOV (audit sink)
- SB (DLQ persistence optional)
- LEARN/BENCH (artifacts + eval reports)

### External services / libraries
- SQLite 3 with WAL mode support
- Migration runner (e.g., Alembic-like or custom)

## 6. Data & State
### Storage
- Single SQLite database file (DB1) with WAL enabled.
- Optional separate file for large artifacts (artifact store) with content-addressing.

### Caching
- Connection pool (bounded) in-process.
- Prepared statement caching where supported.

### Retention
- DB1 defaults (override via CFG):
- Audit log: 365 days (dev), 2 years (paper), 7 years (prod/regulatory) — configurable.
- Artifacts: 90 days (dev), 365 days (paper/prod) — configurable.
- DLQ: 30 days — configurable.

### Migrations / Versioning
- Migrations applied at startup; fail hard if migration fails.
- Schema version stored in SQLite `PRAGMA user_version` and `schema_migrations` table.

## 7. Algorithms / Logic
### Transaction boundaries (DB1)
- EXEC submit: persist OrderIntent (tx) → broker submit (out-of-tx) → persist submission status (tx).
- Fill apply: persist fill (tx) + ledger postings + position update (tx).
- GOV audit: append-only insert with hash chaining (tx).

### Audit hash chaining
- `hash_self = H(record_json + hash_prev)`
- `hash_prev` points to prior record's hash.
- Verification scans chain and recomputes.

### Edge cases
- SQLite busy/locked: retry with backoff up to budget; if still locked, trigger kill-switch (prod) or degrade (dev).

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| DB_PATH | string | ./data/ordinis.db | P0 | SQLite file path. |
| DB_WAL | bool | true | P0 | Enable WAL mode. |
| DB_BUSY_TIMEOUT_MS | int | 5000 | P0 | SQLite busy timeout. |
| DB_RETENTION_AUDIT_DAYS | int | 365 | P0 | Audit retention. |
| DB_RETENTION_DLQ_DAYS | int | 30 | P0 | DLQ retention. |

**Environment variables**
- `ORDINIS_DB_PATH`

## 9. Non-Functional Requirements
- Durability: commits survive process crash (WAL).
- Write throughput: sustain 1k inserts/sec on commodity SSD for typical event sizes.
- Integrity: audit chain verification must detect tampering with > 0.999 probability (cryptographic hash).

## 10. Observability
**Metrics**
- `db_query_total{op,result}`
- `db_query_latency_ms{op}`
- `db_lock_contention_total`
- `audit_chain_verify_total{result}`

**Logs**
- Migration apply logs include version transitions.

**Alerts**
- Persistent DB lock contention
- Audit chain broken
- Migration failure

## 11. Failure Modes & Recovery
- DB locked: retry with backoff; if exceeds budget, fail action and emit alert.
- Corrupt DB: fail startup; require manual restore from backup (DB1) or rebuild from ledger/audit where possible.
- Disk full: fail writes; trigger kill-switch; alert.

## 12. Test Plan
### Unit tests
- Repo upsert semantics and idempotency lookups.
- Audit chain hash creation + verification.

### Integration tests
- Simulate crash between order intent persist and broker submit; ensure no duplicate on restart.
- WAL mode enabled check.

### Acceptance criteria
- All side-effecting actions are recoverable after crash without duplication.

## 13. Open Questions / Risks
- Do we implement a separate artifact store in DB1 or store blobs in SQLite?
- Backup strategy for SQLite (hot backup vs periodic copy).

## 14. Traceability
### Parent
- DB1 System SRS: Durable audit and state persistence.

### Related
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)
- [L2-13 EXEC — ExecutionEngine](L2-13_EXEC_ExecutionEngine_FlowRoute.md)

### Originating system requirements
- PERSIST-FR-001, PERSIST-FR-002, PERSIST-FR-003
