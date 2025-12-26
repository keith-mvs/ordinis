# L2-14 — PORT — PortfolioEngine

## 1. Identifier
- **ID:** `L2-14`
- **Component:** PORT — PortfolioEngine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Maintain portfolio positions, cash, equity, and exposures.
- Apply executions (fills) to update positions via LedgerEngine.
- Provide rebalance planning based on targets and constraints.
- Provide portfolio snapshots to RISK/SIG/ANA.

### Out of scope
- Optimization (OPT) (provides targets but not computed here).
- Broker-side account truth (EXEC handles reconciliation with broker).

## 3. Responsibilities
- Own the authoritative in-process portfolio snapshot API.
- Translate fills into ledger postings and position updates (via LEDG).
- Emit PositionUpdate events and portfolio snapshots.
- Provide rebalance order intents that pass constraints (basic).

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class PortfolioSnapshot:
    ts: str
    cash: float
    equity: float
    positions: List[Dict[str, Any]]
    exposures: Dict[str, float]
    pnl: Dict[str, float]

class PortfolioEngine:
    def get_snapshot(self) -> PortfolioSnapshot: ...
    def apply_execution(self, execution_report: Dict[str, Any]) -> PortfolioSnapshot: ...
    def rebalance(self, targets: Dict[str, float], constraints: Dict[str, Any]) -> List[Dict[str, Any]]: ...
```

### Events
**Consumes**
- ordinis.executions.report
- ordinis.market.* (for MTM)
- ordinis.opt.weights (optional targets)

**Emits**
- ordinis.portfolio.position.updated
- ordinis.portfolio.snapshot
- ordinis.orders.intent (rebalance intents)

### Schemas
- EVT-DATA-018 PositionUpdate
- PortfolioSnapshot schema (artifact/event payload)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_PORT_LEDGER_POST_FAILED` | Ledger failed to post execution | No |
| `E_PORT_SNAPSHOT_INVALID` | Snapshot invariant violated | No |
| `E_PORT_REBALANCE_INFEASIBLE` | Targets infeasible under constraints | No |

## 5. Dependencies
### Upstream
- LEDG for postings and invariants
- EXEC for fills
- CFG for valuation methods and constraints
- PERSIST for position state persistence

### Downstream
- RISK consumes snapshots
- ANA consumes positions for attribution
- LEARN uses portfolio state for labeling

### External services / libraries
- None required in DB1

## 6. Data & State
### Storage
- SQLite tables: `positions`, `cash`, `ledger_entries` (LEDG-owned) and snapshot artifacts.

### Caching
- In-memory snapshot updated per execution and per market price tick/bar.

### Retention
- Position history retention configurable (default 365 days).

### Migrations / Versioning
- Position schema changes require migration; snapshot schema versioned.

## 7. Algorithms / Logic
### Apply execution
1. Receive ExecutionReport.
2. For each fill:
   - Post ledger entries (cash delta, position delta, fees).
3. Update in-memory position state and compute realized/unrealized PnL.
4. Emit PositionUpdate and PortfolioSnapshot.

### MTM valuation
- Update MTM price per symbol on each bar close (or tick if enabled).
- Recompute unrealized PnL and equity.

### Rebalance planning (DB1 baseline)
- Input: target weights by symbol and constraints.
- Compute desired notional per symbol = target_weight * equity.
- Translate delta notional to order qty using latest price.
- Apply basic rounding and min-order constraints.
- Output OrderIntents; RISK will still evaluate.

### Edge cases
- Missing price: do not trade that symbol; emit `ops.price.missing`.
- Corporate actions/dividends: DB1 optional; if missing, treat as cash adjustments when provided by data source.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| PORT_VALUATION_PRICE | string | close | P0 | `close` from bars. |
| PORT_REBALANCE_MIN_NOTIONAL | float | 50.0 | P0 | Skip tiny trades. |
| PORT_ROUND_LOT_SIZE | bool | true | P0 | Round qty to lot size (if known). |
| PORT_SNAPSHOT_EVERY_CYCLE | bool | true | P0 | Emit snapshot per cycle. |

**Environment variables**
- `ORDINIS_PORT_SNAPSHOT_EVERY_CYCLE`

## 9. Non-Functional Requirements
- Apply execution latency: p95 < 10ms per fill batch.
- Ledger invariants enforced every cycle; violations trigger kill-switch (configurable).
- Portfolio snapshot retrieval O(positions) with bounded allocations.

## 10. Observability
**Metrics**
- `port_positions_count`
- `port_apply_exec_latency_ms`
- `port_equity`
- `port_drawdown`

**Alerts**
- Snapshot invariant violation
- Reconcile mismatches vs broker

## 11. Failure Modes & Recovery
- Ledger post failure: fail closed; do not mutate state; emit alert.
- Price missing: degrade valuation; block trading on that symbol via RISK.
- DB persistence failure: state still in-memory; must reconcile on restart; alert.

## 12. Test Plan
### Unit tests
- Fill application updates positions and cash correctly.
- PnL computations for long/short scenarios.
- Rebalance planner produces correct intents given simple targets.

### Integration tests
- EXEC execution reports applied to ledger; ANA metrics reflect state changes.

### Acceptance criteria
- Portfolio state is reproducible from ledger replay and matches live state after restart.

## 13. Open Questions / Risks
- Do we implement full double-entry ledger in DB1 or single-entry with invariants? (LEDG spec favors disciplined postings).
- How to source instrument metadata (lot size, tick size) for rounding in DB1.

## 14. Traceability
### Parent
- DB1 System SRS: PortfolioEngine maintains positions/allocations; applies rebalance logic.

### Related
- [L2-15 LEDG — LedgerEngine](L2-15_LEDG_LedgerEngine.md)
- [L2-13 EXEC — ExecutionEngine](L2-13_EXEC_ExecutionEngine_FlowRoute.md)

### Originating system requirements
- PORT-IF-001, PORT-FR-001..003
