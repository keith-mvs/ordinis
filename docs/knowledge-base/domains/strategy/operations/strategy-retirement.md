# Strategy Retirement Criteria and Workflow

## Purpose

Decide when to pause or retire a strategy, unwind safely, and capture lessons.

## Quantitative Triggers (examples)

- Sharpe < target (e.g., 1.0) or Sortino < target over rolling N days.
- Max drawdown > limit (e.g., 20%) or recovery time > limit.
- Alpha half-life below threshold; info ratio < target.
- Slippage/commission above budget for N periods.
- Regime incompatibility: performance fails across required regimes.

## Qualitative/Operational Triggers

- Data/infra issues that can’t be resolved within SLA.
- Model validity concerns (overfit, stale features, drift, compliance issues).
- Strategy duplicates exposure already covered more efficiently elsewhere.

## Workflow

1) Flag: auto alert on trigger; PM/Risk acknowledge.
2) Investigate: root cause (data, execution, signal drift, regime shift).
3) Decide: fix (tune/patch), pause (shadow only), or retire (unwind + archive).
4) Execute:
   - Reduce size, then unwind positions; cancel orders.
   - Turn on shadow mode if continuing research.
   - Update routing/allocations to prevent new flow.
5) Record:
   - Version/config, data set, metrics at decision time.
   - Root cause, actions, next review date.
6) Review: post-mortem and add to strategy graveyard; track action items.

## Safe Unwind

- Prefer staged exits to avoid market impact; respect liquidity/volume bands.
- Preserve hedges until underlying positions are closed.
- Verify positions, PnL, and cash after unwind; reconcile.

## Documentation

- Summary: why retired, when, who approved.
- Performance snapshot: key metrics, charts, regimes tested.
- Operational notes: incidents, data/execution issues.
- Action items: what to fix if reconsidered; where to monitor for reuse.

## Governance

- Approvals: PM + Risk for pause/retire; Ops for unwind plan.
- Audit: keep artifacts (reports, configs, code tag, data manifest).
- Re-entry: require new version/tag and fresh validation before redeploy.
