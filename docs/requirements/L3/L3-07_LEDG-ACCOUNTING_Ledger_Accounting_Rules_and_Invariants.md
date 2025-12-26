# L3-07 — LEDG-ACCOUNTING — Ledger Accounting Rules & Invariants

## 1. Identifier
- **ID:** `L3-07`
- **Component:** LEDG-ACCOUNTING — Ledger Accounting Rules & Invariants
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Double-entry (or disciplined single-entry) posting rules for fills and fees.
- Realized/unrealized PnL computation rules (DB1 baseline).
- Invariant definitions and tolerance rules.
- Replay rules to reconstruct portfolio state.

### Out of scope
- Tax lot accounting and jurisdiction-specific rules (P2).
- Complex corporate actions automation (P1+).

## 3. Responsibilities
- Define ledger entry types and posting templates.
- Define invariant checks and what constitutes a breach vs warning.
- Define replay algorithm and ordering guarantees.

## 4. Interfaces
### Public APIs / SDKs
Ledger posting and invariant APIs are exposed by [L2-15 LEDG](../L2/L2-15_LEDG_LedgerEngine.md). This spec defines the rules behind those APIs.

**Entry types (DB1 baseline)**
- `cash` — cash delta
- `position` — position qty delta
- `fee` — fee expense
- `pnl_realized` — realized PnL delta
- `mtm` — mark-to-market price update (optional)

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- LedgerEntry schema (EVT internal)
- InvariantViolationReport schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_LEDG_INVARIANT_BREACH` | Invariant breach detected | No |
| `E_LEDG_POST_TEMPLATE_INVALID` | Posting template produced invalid entries | No |

## 5. Dependencies
### Upstream
- CFG for tolerance and accounting mode

### Downstream
- LEDG engine runtime
- PORT (portfolio state)
- ANA (metrics)

### External services / libraries
_None._

## 6. Data & State
### Storage
- Ledger entries persisted in SQLite `ledger_entries` with strict constraints (non-null ts, amount, entry_type).

### Caching
- Optional: incremental aggregates (cash balance, positions) stored as derived state for faster snapshot.

### Retention
- Ledger entries retained aligned with audit retention.

### Migrations / Versioning
- EntryType enum is append-only; existing meanings never change.

## 7. Algorithms / Logic
### Posting templates (simplified DB1)

**Buy fill (qty > 0)**
- position: +qty
- cash: -(qty * price) - fees
- fee: -fees

**Sell fill (qty < 0)**
- position: -qty
- cash: +(qty * price) - fees   (note qty positive in notional)
- fee: -fees

**Realized PnL**
- DB1 baseline: average cost method
  - maintain avg_cost per symbol in derived state
  - realized_pnl = (fill_price - avg_cost) * qty_closed (sign-adjusted)
- Emit `pnl_realized` entry when closing part/all of position.

### Invariants (minimum)
- Cash balance equals sum(cash entries) within tolerance.
- Position qty equals sum(position entries) per symbol.
- Equity consistency: equity == cash + Σ(qty * last_mtm_price) within tolerance.
- No negative cash if margin disabled (configurable).

### Replay
- Sort ledger entries by ts then entry_id.
- Apply templates to reconstruct:
  - cash
  - positions
  - avg_cost (if used)
  - realized/unrealized series (optional)

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| LEDG_ACCOUNTING_METHOD | string | avg_cost | P0 | avg_cost|fifo (fifo later). |
| LEDG_ALLOW_MARGIN | bool | false | P0 | If false, negative cash is invariant breach. |
| LEDG_INVARIANT_TOLERANCE | float | 1e-6 | P0 | Numeric tolerance. |

## 9. Non-Functional Requirements
- Reproducibility: replay yields identical derived state.
- Invariant checks are O(entries) for scan or O(positions) for incremental mode.

## 10. Observability
LEDG emits `ledger.invariant.violation` events and metrics (see L2-15). Violations should include:
- invariant_id
- expected vs actual
- tolerance
- correlation_id

## 11. Failure Modes & Recovery
- Invariant breach: trigger kill-switch if configured; require manual investigation.
- Corrupt ledger entries: fail startup and require restore.

## 12. Test Plan
### Unit tests
- Posting templates produce correct cash/position deltas.
- Realized PnL calculation under avg_cost with partial close.
- Invariant checker detects injected inconsistency.

### Integration tests
- Run a benchmark backtest and replay ledger; compare to live portfolio state.

## 13. Open Questions / Risks
- Average cost vs FIFO decision (avg_cost simplest for DB1).
- Do we store mtm prices in ledger entries or in separate price table?

## 14. Traceability
### Parent
- [L2-15 LEDG — LedgerEngine](../L2/L2-15_LEDG_LedgerEngine.md)

### Originating system requirements
- LEDG-FR-001, LEDG-FR-002
