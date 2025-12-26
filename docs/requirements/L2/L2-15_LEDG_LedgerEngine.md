# L2-15 — LEDG — LedgerEngine

## 1. Identifier
- **ID:** `L2-15`
- **Component:** LEDG — LedgerEngine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Authoritative accounting for cash/positions/fees/PnL via postings.
- Atomic posting of fills into ledger entries.
- Invariant checks: equity consistency, PnL consistency.
- Rebuild/replay support to reconstruct portfolio from ledger.

### Out of scope
- Tax lots, wash sale, and jurisdiction-specific accounting (P2).
- Complex corporate actions (splits, mergers) automation (P1+).

## 3. Responsibilities
- Own ledger entry schema and posting API.
- Guarantee atomicity and ordering of postings per correlation_id and order_id.
- Expose invariant checks and raise kill-switch on invariant breach per policy.
- Support replay to reconstruct portfolio state deterministically.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

EntryType = Literal["cash","position","fee","pnl_realized","pnl_unrealized","mtm"]

@dataclass(frozen=True)
class LedgerEntry:
    entry_id: str
    ts: str
    correlation_id: str
    entry_type: EntryType
    symbol: Optional[str]
    amount: float
    meta: Dict[str, Any]

class LedgerEngine:
    def post_fill(self, fill: Dict[str, Any], correlation_id: str) -> List[LedgerEntry]: ...
    def post_mtm(self, symbol: str, price: float, ts: str) -> LedgerEntry: ...
    def verify_invariants(self, snapshot: Dict[str, Any]) -> Dict[str, Any]: ...  # returns status + violations
    def replay(self, since_ts: Optional[str] = None) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.executions.report (fills)
- ordinis.market.* (MTM)

**Emits**
- ordinis.ledger.entry.posted
- ordinis.ops.ledger.invariant.violation

### Schemas
- LedgerEntry schema
- Invariant violation report schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_LEDG_POST_FAILED` | Ledger posting transaction failed | Yes |
| `E_LEDG_INVARIANT_BREACH` | Ledger invariants violated | No |
| `E_LEDG_REPLAY_FAILED` | Ledger replay failed | No |

## 5. Dependencies
### Upstream
- PERSIST for ledger storage and transactions
- CFG for invariant thresholds and kill-switch policy

### Downstream
- PORT uses ledger as authoritative source
- ANA uses ledger for realized/unrealized calculations (optional)
- GOV audit references ledger postings for trade justification

### External services / libraries
- None required in DB1

## 6. Data & State
### Storage
- SQLite tables: `ledger_entries` indexed by (ts, symbol, correlation_id), plus `ledger_state` optional.

### Caching
- Optional in-memory aggregates for quick snapshot, derived from ledger stream.

### Retention
- Ledger entries retained per audit retention (default aligns with audit log).

### Migrations / Versioning
- Ledger schema changes require migration; entry_type enum is append-only in DB1.

## 7. Algorithms / Logic
### Posting model (DB1)
- For each fill:
  - Position delta entry (qty * sign)
  - Cash delta entry (notional + fees)
  - Fee entry (fees)
  - Realized PnL entry if closing/offsetting positions (DB1 simplified; full tax lots later)
- For MTM updates:
  - MTM entry records latest price and derived unrealized PnL delta (optional).

### Invariants (minimum)
- `equity == cash + Σ(qty * mtm_price)` within tolerance
- No NaN/inf in amounts
- Positions reconstructed from entries match PORT snapshot

### Edge cases
- Partial fills: each fill posts independently; idempotent posting requires unique fill_id (from broker) or derived hash.
- Missing fill_id: derive `fill_hash = H(order_id, ts, qty, price)`.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| LEDG_INVARIANT_TOLERANCE | float | 1e-6 | P0 | Numeric tolerance. |
| LEDG_FAIL_ON_INVARIANT | bool | true | P0 | Prod should be true. |
| LEDG_ENABLE_MTM_ENTRIES | bool | true | P0 | Emit MTM entries. |

**Environment variables**
- `ORDINIS_LEDG_FAIL_ON_INVARIANT`

## 9. Non-Functional Requirements
- Atomicity: posting is all-or-nothing for each fill batch.
- Replay determinism: replay yields identical state given same ledger entries.
- Invariant checks run every cycle with p95 < 20ms for typical portfolios.

## 10. Observability
**Metrics**
- `ledg_entries_total{type}`
- `ledg_post_latency_ms`
- `ledg_invariant_violations_total{type}`

**Alerts**
- Any invariant violation triggers immediate alert; may trigger kill-switch.

## 11. Failure Modes & Recovery
- DB write failure: retry; if persistent, block trading via kill-switch.
- Invariant breach: emit violation event; trigger kill-switch depending on config.
- Replay failure: fail startup; require manual restore/rebuild.

## 12. Test Plan
### Unit tests
- Posting correctness for buy/sell, partial fills.
- Invariant checker catches injected inconsistencies.

### Integration tests
- Replay ledger and compare to live PORT snapshot across a benchmark run.

### Acceptance criteria
- Ledger replay reproduces portfolio state exactly for deterministic backtests.

## 13. Open Questions / Risks
- DB1 realized PnL rules: FIFO vs average cost; pick one and make it explicit.
- Corporate action handling in DB1: manual adjustment events vs automatic ingestion.

## 14. Traceability
### Parent
- DB1 System SRS: Ledger authority for portfolio accounting and reproducibility.

### Children / Related
- [L3-07 LEDG-ACCOUNTING — Ledger Accounting Rules & Invariants](../L3/L3-07_LEDG-ACCOUNTING_Ledger_Accounting_Rules_and_Invariants.md)
- [L2-14 PORT — PortfolioEngine](L2-14_PORT_PortfolioEngine.md)

### Originating system requirements
- LEDG-FR-001, LEDG-FR-002
